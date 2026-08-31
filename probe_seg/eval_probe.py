import os
import sys
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler

# 将项目根目录加入路径，便于导入 src.*
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.diff_seg_net import DiffSegNet
from src.seg_head import create_model
from src.probe_dataset import ImageMaskDataset, load_file_list
from src.probe_utils import (
    compute_iou, compute_precision_recall, compute_ap, compute_ods_ois,
    save_predictions, set_seed, ensure_dir, get_palette, colorize_mask
)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="评估所用的yaml配置文件路径")
    args = parser.parse_args()

    # 1) 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) 数据
    img_dir = config["data"]["img_dir"]
    mask_dir = config["data"]["mask_dir"]
    test_list_path = config["data"]["test_list"]
    test_list = load_file_list(test_list_path)
    img_size = config["data"].get("img_size", 256)
    mask_threshold = config["data"].get("mask_threshold", 128)
    ignore_label = config["model"].get("ignore_label", None)

    test_dataset = ImageMaskDataset(
        img_dir, mask_dir, test_list,
        img_size=img_size,
        ignore_label=ignore_label,
        mask_threshold=mask_threshold
    )
    batch_size = config.get("batch_size", 4)
    num_workers = config.get("num_workers", 4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 3) 预训练 U-Net（冻结）
    from bypassunet.bypass_unet import MeanBypassUNet
    unet_ckpt = config["model"]["unet_ckpt"]
    if not os.path.exists(unet_ckpt):
        raise FileNotFoundError(f"未找到unet_ckpt: {unet_ckpt}")
    if os.path.isdir(unet_ckpt):
        unet = MeanBypassUNet.from_pretrained(unet_ckpt)
    else:
        raise ValueError("unet_ckpt 应为 save_pretrained 保存的目录，例如 .../model_epoch_xxx/unet")
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False
    unet = unet.to(device)

    # 4) 分割头
    seg_cfg = config["model"]["seg_head"]
    in_channels = seg_cfg.get("in_channels", seg_cfg.get("in_dim"))
    num_classes = seg_cfg["num_classes"]
    seg_head = create_model(
        seg_cfg["name"],
        num_classes=num_classes,
        in_channels=in_channels,
        output_size=(img_size, img_size)
    ).to(device)
    if "ckpt" in seg_cfg and seg_cfg["ckpt"]:
        seg_state = torch.load(seg_cfg["ckpt"], map_location=device)
        seg_head.load_state_dict(seg_state)
    seg_head.eval()

    # 5) DiffSegNet 与调度器
    # 与 train_probe_wono 保持一致，默认使用 squaredcos_cap_v2
    beta_schedule = config.get("ddpm_beta_schedule", "squaredcos_cap_v2")  # 可选: "linear" 或 "squaredcos_cap_v2"
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule=beta_schedule)
    model = DiffSegNet(
        unet=unet,
        seg_head=seg_head,
        probe_block_name=config["model"]["probe_block_name"],
        probe_timestep=config["model"]["probe_timestep"],
        scheduler=scheduler
    ).to(device)
    model.eval()

    # 6) 推理与评估（对齐 train_probe_wono 的验证循环）
    output_dir = config["output_dir"]
    ensure_dir(output_dir)
    pred_dir = os.path.join(output_dir, "predictions")
    vis_dir = os.path.join(output_dir, "predictions_color")
    ensure_dir(pred_dir)
    ensure_dir(vis_dir)

    all_preds, all_gts, all_probs = [], [], []
    palette = get_palette(num_classes)

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)  # [B, C, H, W]
            # 概率用于 AP/ODS/OIS
            if logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=1)[:, 1]  # 取前景通道概率
            else:
                probs = torch.sigmoid(logits[:, 0])
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.append(preds)
            all_gts.append(masks.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    # 指标：mIoU、Precision、Recall、AP、ODS、OIS
    iou_per_class, miou = compute_iou(all_preds, all_gts, num_classes, ignore_label)
    precision, recall = compute_precision_recall(all_preds, all_gts, num_classes, ignore_label)
    ap = compute_ap(all_probs, all_gts, ignore_index=ignore_label)
    ods, ois = compute_ods_ois(all_probs, all_gts, ignore_index=ignore_label)

    print(f"mIoU={miou:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  AP={ap:.4f}  ODS={ods:.4f}  OIS={ois:.4f}")

    # 保存指标
    metrics = dict(
        mIoU=float(miou),
        Precision=float(precision),
        Recall=float(recall),
        AP=float(ap),
        ODS=float(ods),
        OIS=float(ois),
        IoU_per_class=[float(x) for x in (iou_per_class.tolist() if hasattr(iou_per_class, "tolist") else iou_per_class)]
    )
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 额外保存到可读的TXT文件
    txt_path = os.path.join(output_dir, "metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"mIoU: {miou:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"AP: {ap:.4f}\n")
        f.write(f"ODS: {ods:.4f}\n")
        f.write(f"OIS: {ois:.4f}\n")
        if iou_per_class is not None:
            for ci, v in enumerate(iou_per_class):
                f.write(f"IoU[class_{ci}]: {float(v):.4f}\n")

    # 7) 保存预测（灰度与彩色）
    for idx, name in enumerate(test_list):
        mask = all_preds[idx]
        gray_path = os.path.join(pred_dir, f"{name}.png")
        save_predictions(mask, gray_path, palette)  # 若save_predictions已着色，这里可直接保存；否则保存灰度
        # 彩色可视化（确保 colorize_mask 存在）
        try:
            vis = colorize_mask(mask, palette).convert("RGB")
            vis.save(os.path.join(vis_dir, f"{name}.png"))
        except Exception:
            pass

    print(f"Predictions saved to: {pred_dir}")
    print(f"Color visualizations saved to: {vis_dir}")
    print(f"Metrics saved to: {os.path.join(output_dir, 'metrics.json')}")
    print(f"Metrics (txt) saved to: {txt_path}")

if __name__ == "__main__":
    main()