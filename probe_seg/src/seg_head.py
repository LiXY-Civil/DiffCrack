import torch
import torch.nn as nn
import torch.nn.functional as F

# 简单MLP分割头
class MLPSimple(nn.Module):
    def __init__(self, num_classes, in_channels, output_size=None):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.output_size = output_size  # (H, W) or None

    def forward(self, x):
        # 支持 [B, C, H, W]、[B, L, C]
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, C)
            logits = self.layers(x)
            logits = logits.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B, num_classes, H, W]
        elif x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.reshape(-1, C)
            logits = self.layers(x)
            logits = logits.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("Input shape不支持")
        if self.output_size is not None:
            logits = F.interpolate(logits, size=self.output_size, mode='bilinear', align_corners=False)
        return logits

# 深层MLP分割头
class MLPDeep(nn.Module):
    def __init__(self, num_classes, in_channels, output_size=None):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, num_classes)
        )
        self.output_size = output_size

    def forward(self, x):
        # 支持 [B, C, H, W]、[B, L, C]
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, C)
            logits = self.layers(x)
            logits = logits.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        elif x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.reshape(-1, C)
            logits = self.layers(x)
            logits = logits.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("Input shape不支持")
        if self.output_size is not None:
            logits = F.interpolate(logits, size=self.output_size, mode='bilinear', align_corners=False)
        return logits

# FCN全卷积头（轻量原型）
class FCNLightHead(nn.Module):
    def __init__(self, num_classes, in_channels, output_size=None):
        super().__init__()
        self.fc_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.output_size = output_size

    def forward(self, x):
        # 支持 [B, C, H, W]、[B, L, C]
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).view(B, C, H, W)
        logits = self.fc_conv(x)
        if self.output_size is not None:
            logits = F.interpolate(logits, size=self.output_size, mode='bilinear', align_corners=False)
        return logits

# Segmenter MLP解码器
class SegmenterMLPDecoder(nn.Module):
    def __init__(self, num_classes, in_channels, mlp_dim=256, output_size=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mlp_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mlp_dim, in_channels, kernel_size=1)
        )
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.output_size = output_size

    def forward(self, x):
        # 支持 [B, L, C] or [B, C, H, W]
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.mlp(x)
        logits = self.classifier(x)
        if self.output_size is not None:
            logits = F.interpolate(logits, size=self.output_size, mode='bilinear', align_corners=False)
        return logits

# 轻量FPN金字塔头
class LightFPNHead(nn.Module):
    def __init__(self, num_classes, in_channels_list, out_channels=64, output_size=None):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.final_conv = nn.Conv2d(out_channels, num_classes, 1)
        self.output_size = output_size

    def forward(self, features):
        # features: list of [B, C, H, W], from high to low resolution
        p = self.lateral_convs[0](features[0])
        for i in range(1, len(features)):
            p = F.interpolate(p, scale_factor=2, mode='bilinear', align_corners=False)
            p = p + self.lateral_convs[i](features[i])
            p = F.conv2d(p, weight=torch.ones((p.shape[1], p.shape[1], 3, 3), device=p.device), padding=1, groups=p.shape[1]) / 9.0
        logits = self.final_conv(p)
        if self.output_size is not None:
            logits = F.interpolate(logits, size=self.output_size, mode='bilinear', align_corners=False)
        return logits

# 轻量级U-Net解码器
class LightUnetDecoder(nn.Module):
    def __init__(self, num_classes, in_channels, output_size=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.final_conv = nn.Conv2d(32, num_classes, 1)
        self.output_size = output_size

    def forward(self, x):
        # 支持 [B, C, H, W]、[B, L, C]
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        logits = self.final_conv(x)
        if self.output_size is not None:
            logits = F.interpolate(logits, size=self.output_size, mode='bilinear', align_corners=False)
        return logits

# LR-ASPP分割头
class LRASPP(nn.Module):
    def __init__(self, num_classes, in_channels, output_size=None):
        super().__init__()
        self.aspp1 = nn.Conv2d(in_channels, 64, 1)
        self.aspp2 = nn.Conv2d(in_channels, 64, 3, dilation=6, padding=6)
        self.aspp3 = nn.Conv2d(in_channels, 64, 3, dilation=12, padding=12)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(192, 192, 1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(192, num_classes, 1)
        self.output_size = output_size

    def forward(self, x):
        # 支持 [B, C, H, W]、[B, L, C]
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).view(B, C, H, W)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        se = self.se(x_cat)
        x_se = x_cat * se
        logits = self.final_conv(x_se)
        if self.output_size is not None:
            logits = F.interpolate(logits, size=self.output_size, mode='bilinear', align_corners=False)
        return logits
    
class DeepLabV3Head(nn.Module):
    def __init__(self, num_classes, in_channels, output_size=None):
        super().__init__()
        # ASPP模块：多尺度空洞卷积
        self.aspp1 = nn.Conv2d(in_channels, 256, 1)
        self.aspp2 = nn.Conv2d(in_channels, 256, 3, padding=6, dilation=6)
        self.aspp3 = nn.Conv2d(in_channels, 256, 3, padding=12, dilation=12)
        self.aspp4 = nn.Conv2d(in_channels, 256, 3, padding=18, dilation=18)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, 1),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(256 * 5, num_classes, 1)
        self.output_size = output_size

    def forward(self, x):
        # 支持 [B, C, H, W]、[B, L, C]
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).view(B, C, H, W)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        logits = self.final_conv(x_cat)
        if self.output_size is not None:
            logits = F.interpolate(logits, size=self.output_size, mode='bilinear', align_corners=False)
        return logits


class ResidualSegHead(nn.Module):
    def __init__(self, num_classes, in_channels, output_size=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.res_conv = nn.Conv2d(in_channels, 64, 1)  # 匹配通道数
        self.final_conv = nn.Conv2d(64, num_classes, 1)
        self.output_size = output_size

    def forward(self, x):
        # 支持 [B, C, H, W]、[B, L, C]
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).view(B, C, H, W)
        identity = self.res_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 残差连接
        out = self.relu(out)
        logits = self.final_conv(out)
        if self.output_size is not None:
            logits = F.interpolate(logits, size=self.output_size, mode='bilinear', align_corners=False)
        return logits

# 注意力增强模块（SE）
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# 注意力增强模块（ECA）
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

# 模型注册表
MODEL_REGISTRY = {
    "mlp_simple": MLPSimple,
    "mlp_deep": MLPDeep,
    "conv1x1": FCNLightHead,
    "segmenter_mlp": SegmenterMLPDecoder,
    "light_fpn": LightFPNHead,
    "light_unet": LightUnetDecoder,
    "lraspp": LRASPP,
    "se_block": SEBlock,
    "eca_block": ECABlock,
    "residual_seg": ResidualSegHead,
    "deeplabv3": DeepLabV3Head,
}

def create_model(model_name, *args, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    return MODEL_REGISTRY[model_name](*args, **kwargs)