import torch
import torch.nn as nn

class DiffSegNet(nn.Module):
    """
    DiffSegNet: 端到端扩散特征分割网络
    - 冻结的扩散U-Net主干
    - 可训练的分割头
    - 支持指定block和timestep的特征提取
    """
    def __init__(self, unet, seg_head, probe_block_name, probe_timestep, scheduler):
        """
        Args:
            unet: 冻结的MeanBypassUNet实例
            seg_head: 可训练的分割头实例
            probe_block_name: str, 需要hook的block名（如 'up_blocks.2.upsamplers.last'）
            probe_timestep: int, 扩散时间步
            scheduler: 扩散调度器实例
        """
        super().__init__()
        self.unet = unet
        self.seg_head = seg_head
        self.probe_block_name = probe_block_name
        self.probe_timestep = probe_timestep
        self.scheduler = scheduler

        # 注册hook
        self._feature = None
        self._hook_handle = self._register_hook()

        # 冻结unet参数
        for p in self.unet.parameters():
            p.requires_grad = False

    def _get_module_by_name(self, module, name):
        """递归获取子模块，支持 'up_blocks.2.upsamplers.last' 形式"""
        parts = name.split('.')
        for idx, part in enumerate(parts):
            if part.isdigit():
                module = module[int(part)]
            elif part == 'last':
                # 取ModuleList最后一层
                if isinstance(module, nn.ModuleList):
                    module = module[-1]
                else:
                    raise ValueError(f"Cannot use 'last' on non-ModuleList: {'.'.join(parts[:idx+1])}")
            else:
                module = getattr(module, part)
        return module

    def _hook_fn(self, module, input, output):
        self._feature = output

    def _register_hook(self):
        # 递归定位到目标模块并注册hook
        target_module = self._get_module_by_name(self.unet, self.probe_block_name)
        return target_module.register_forward_hook(self._hook_fn)

    def remove_hooks(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def forward(self, x):
        """
        x: [B, 3, H, W] 输入原始图像
        返回: [B, num_classes, H, W] 分割logits
        """
        self._feature = None
        B = x.shape[0]
        device = x.device

        # 构造扩散噪声
        t = torch.tensor([self.probe_timestep] * B, device=device)
        # noise = torch.randn_like(x)
        noise = torch.zeros_like(x)  # 使用零噪声以便捕获特征
        # 使用scheduler添加噪声
        noisy_x = self.scheduler.add_noise(x, noise, t)

        # 前向传播，hook自动捕获特征
        _ = self.unet(noisy_x, t)
        feat = self._feature
        if feat is None:
            raise RuntimeError(f"Feature not captured at {self.probe_block_name}")

        # 分割头
        logits = self.seg_head(feat)
        return logits

    def extract_feature(self, x):
        """
        仅提取指定block的特征（不经过分割头）
        """
        self._feature = None
        B = x.shape[0]
        device = x.device
        t = torch.tensor([self.probe_timestep] * B, device=device)
        noise = torch.randn_like(x)
        noisy_x = self.scheduler.add_noise(x, noise, t)
        _ = self.unet(noisy_x, t)
        feat = self._feature
        if feat is None:
            raise RuntimeError(f"Feature not captured at {self.probe_block_name}")
        return feat

    def set_probe(self, probe_block_name=None, probe_timestep=None):
        """
        动态切换hook的block或timestep
        """
        if probe_block_name is not None and probe_block_name != self.probe_block_name:
            self.remove_hooks()
            self.probe_block_name = probe_block_name
            self._hook_handle = self._register_hook()
        if probe_timestep is not None:
            self.probe_timestep = probe_timestep