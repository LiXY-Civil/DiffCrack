import torch
import torch.nn as nn
from diffusers import UNet2DModel
from diffusers.configuration_utils import register_to_config

class MeanBypassBranch(nn.Module):
    """
    ECS风格旁路分支：多层1x1卷积+多次时间嵌入+可选归一化。
    - 输入: mean_in [B, C, 1]（每通道空间均值），t_emb [B, embed_dim]（时间嵌入）
    - 输出: [B, C]，每通道均值噪声
    - 结构特点：每层卷积后都加一次时间嵌入（Dense），可选GroupNorm，激活为ReLU
    """
    def __init__(self, in_channels, embed_dim, hidden_dim=128, use_gnorm=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.gn1 = nn.GroupNorm(32, hidden_dim) if use_gnorm else nn.Identity()
        self.dense1 = nn.Linear(embed_dim, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.gn2 = nn.GroupNorm(32, hidden_dim) if use_gnorm else nn.Identity()
        self.dense2 = nn.Linear(embed_dim, hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, in_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, mean_in, t_emb):
        # mean_in: [B, C, 1], t_emb: [B, embed_dim]
        x = self.conv1(mean_in)
        x = self.gn1(x)
        x = x + self.dense1(t_emb).unsqueeze(-1)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = x + self.dense2(t_emb).unsqueeze(-1)
        x = self.relu(x)
        x = self.conv3(x)
        return x.squeeze(-1)  # [B, C]

class MeanBypassUNet(UNet2DModel):
    """
    主干/旁路分离的UNet模型，ECS风格旁路分支。
    - 主干预测去均值后的噪声细节。
    - 旁路预测均值噪声（多层1x1卷积+多次时间嵌入+可选归一化）。
    - 融合输出主干和旁路的合成噪声。
    """
    @register_to_config
    def __init__(
        self,
        sample_size=128,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        hidden_dim=128,
        use_gnorm=True,
        **kwargs
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            **kwargs
        )
        # 获取时间嵌入维度
        t_emb_dim = self.time_embedding.linear_2.out_features

        # 旁路分支：ECS风格
        self.mean_bypass = MeanBypassBranch(
            in_channels=self.config.in_channels,
            embed_dim=t_emb_dim,
            hidden_dim=hidden_dim,
            use_gnorm=use_gnorm
        )

    def forward(self, sample, timestep, class_labels=None, return_dict=True):
        """
        Args:
            sample: [B, C, H, W]
            timestep: [B] or scalar
            class_labels: 可选
            return_dict: 是否返回字典
        Returns:
            dict(sample=..., mean_pred=..., residual_pred=...)
        """
        dtype = sample.dtype
        device = sample.device
        B, C, H, W = sample.shape

        # 1. 计算每通道空间均值
        spatial_mean = sample.mean(dim=(2, 3)).to(dtype=dtype, device=device)  # [B, C]

        # 2. 去均值残差
        residual_x = sample - spatial_mean[:, :, None, None]  # [B, C, H, W]

        # 3. 时间嵌入
        t_proj = self.time_proj(timestep)                # [B, time_proj_dim]
        t_emb = self.time_embedding(t_proj)              # [B, t_emb_dim]

        # 4. 旁路分支
        mean_in = spatial_mean.unsqueeze(-1)  # [B, C, 1]
        mean_pred = self.mean_bypass(mean_in, t_emb).to(dtype=dtype)  # [B, C]

        # 5. 主干分支（原UNet2DModel）
        residual_pred = super().forward(residual_x, timestep, class_labels=class_labels, return_dict=True).sample  # [B, C, H, W]
        residual_pred = residual_pred - residual_pred.mean(dim=(2, 3), keepdim=True)

        # 6. 融合输出
        full_pred = residual_pred + mean_pred[:, :, None, None]  # [B, C, H, W]

        if not return_dict:
            return (full_pred, mean_pred, residual_pred)
        return {
            "sample": full_pred,
            "mean_pred": mean_pred,
            "residual_pred": residual_pred,
        }