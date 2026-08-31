import torch

def compute_bypass_total_loss(
    residual_pred,  # [B, C, H, W]，主干UNet预测的去均值噪声分量
    noise_var,      # [B, C, H, W]，真实噪声的去均值分量
    mean_pred,      # [B, C]，旁路分支预测的每通道噪声均值
    noise_mu,       # [B, C]，真实噪声的每通道均值
    lambda_=1.0,    # 旁路分支损失权重
    gamma=0.1       # 主干分支空间均值约束权重
):
    """
    标准化旁路分支DDPM损失函数，主损失为标量，辅助分量用于日志记录。
    返回:
        total_loss: 标量，可直接 backward
        loss_dict:  辅助分量日志（含主干、旁路、均值约束损失，均为 float）
    """
    dtype = residual_pred.dtype
    noise_var = noise_var.to(dtype)
    mean_pred = mean_pred.to(dtype)
    noise_mu = noise_mu.to(dtype)

    # 主干分支损失（各通道独立MSE）
    loss_var = ((residual_pred - noise_var) ** 2).mean(dim=(0, 2, 3))  # [C]
    # 旁路分支损失（各通道独立MSE）
    loss_mu = ((mean_pred - noise_mu) ** 2).mean(dim=0)                # [C]
    # 主干分支空间均值约束损失（各通道独立MSE）
    residual_pred_mean = residual_pred.mean(dim=(2, 3))                # [B, C]
    loss_var_mean = (residual_pred_mean ** 2).mean(dim=0)              # [C]
    # 总损失（各通道加权求和）
    total_loss = loss_var.sum() + lambda_ * loss_mu.sum() + gamma * loss_var_mean.sum()

    # 日志分量（转为 float，便于 tensorboard/wandb 记录）
    loss_dict = {
        "loss_var": loss_var.detach().mean().item(),
        "loss_mu": loss_mu.detach().mean().item(),
        "loss_var_mean": loss_var_mean.detach().mean().item(),
        "loss_var_sum": loss_var.sum().item(),
        "loss_mu_sum": loss_mu.sum().item(),
        "loss_var_mean_sum": loss_var_mean.sum().item(),
    }
    return total_loss, loss_dict