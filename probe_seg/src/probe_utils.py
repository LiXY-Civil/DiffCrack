import os
import numpy as np
import torch
from PIL import Image

def compute_iou(pred, target, num_classes, ignore_index=None):
    """
    计算每个类别的IoU和mean IoU
    pred, target: [N] 或 [H, W]，int类型
    num_classes: 类别数
    ignore_index: 忽略的标签（如255），可为None
    返回: per_class_iou, mean_iou
    """
    pred = np.asarray(pred).flatten()
    target = np.asarray(target).flatten()
    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    mean_iou = np.nanmean(ious)
    return ious, mean_iou

def compute_precision_recall(pred, target, num_classes, ignore_index=None):
    """
    计算二分类的Precision、Recall
    """
    pred = np.asarray(pred).flatten()
    target = np.asarray(target).flatten()
    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    # 只适用于二分类
    TP = np.logical_and(pred == 1, target == 1).sum()
    FP = np.logical_and(pred == 1, target == 0).sum()
    FN = np.logical_and(pred == 0, target == 1).sum()
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    return precision, recall

def compute_ap(pred, target, ignore_index=None):
    """
    计算AP（仅二分类，简单实现）
    """
    from sklearn.metrics import average_precision_score
    pred = np.asarray(pred).flatten()
    target = np.asarray(target).flatten()
    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    return average_precision_score(target, pred)

def compute_ods_ois(pred_probs, targets, ignore_index=None, num_thresholds=100):
    """
    计算ODS（全局最佳F1）和OIS（每张图片最佳F1的均值）
    pred_probs: [N, H, W]，每像素属于前景的概率（如sigmoid输出）
    targets: [N, H, W]，0/1标签
    返回: ods, ois
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    all_f1s = []
    for t in thresholds:
        preds = (pred_probs >= t).astype(np.uint8)
        precision, recall = compute_precision_recall(preds, targets, num_classes=2, ignore_index=ignore_index)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        all_f1s.append(f1)
    ods = np.max(all_f1s)
    # OIS: 每张图片单独找最佳F1再平均
    if pred_probs.ndim == 3:  # [N, H, W]
        N = pred_probs.shape[0]
        ois_list = []
        for i in range(N):
            f1s = []
            for t in thresholds:
                preds = (pred_probs[i] >= t).astype(np.uint8)
                precision, recall = compute_precision_recall(preds, targets[i], num_classes=2, ignore_index=ignore_index)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                f1s.append(f1)
            ois_list.append(np.max(f1s))
        ois = np.mean(ois_list)
    else:
        ois = ods
    return ods, ois

def get_palette(num_classes):
    """
    生成调色板（每个类别一个RGB颜色）
    """
    palette = []
    for i in range(num_classes):
        r = (i * 123) % 256
        g = (i * 231) % 256
        b = (i * 321) % 256
        palette.extend([r, g, b])
    # 补齐到768长度
    palette += [0] * (768 - len(palette))
    return palette

def colorize_mask(mask, palette):
    """
    将单通道mask转为彩色PIL图像
    """
    mask_img = Image.fromarray(mask.astype(np.uint8), mode='P')
    mask_img.putpalette(palette)
    return mask_img

def save_predictions(pred_mask, save_path, palette=None):
    """
    保存预测mask为彩色图片
    pred_mask: [H, W]，int类型
    save_path: 保存路径
    palette: 调色板（可用get_palette生成）
    """
    if palette is None:
        num_classes = int(pred_mask.max()) + 1
        palette = get_palette(num_classes)
    mask_img = colorize_mask(pred_mask, palette)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mask_img.save(save_path)

def set_seed(seed):
    """
    设置随机种子，保证实验可复现
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    """
    自动创建目录
    """
    if not os.path.exists(path):
        os.makedirs(path)