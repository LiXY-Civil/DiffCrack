import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    """
    多类别Dice Loss
    pred: [B, C, H, W] logits
    target: [B, H, W] label
    """
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    dice = 0
    for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        dice_c = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice += 1 - dice_c
    return dice / num_classes

def combined_loss(pred, target, criterion_ce, dice_weight=1.0):
    """
    组合加权交叉熵和Dice Loss
    pred: [B, C, H, W] logits
    target: [B, H, W] label
    criterion_ce: nn.CrossEntropyLoss实例
    dice_weight: Dice Loss权重
    """
    ce = criterion_ce(pred, target)
    dice = dice_loss(pred, target)
    return ce + dice_weight * dice, ce, dice