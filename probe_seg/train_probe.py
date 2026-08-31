import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

from diffusers import DDPMScheduler

from src.diff_seg_net import DiffSegNet
from src.seg_head import create_model
from src.probe_dataset import ImageMaskDataset, load_file_list
from src.probe_utils import (
    compute_iou, compute_precision_recall, compute_ap, compute_ods_ois,
    save_predictions, set_seed, ensure_dir, get_palette, colorize_mask
)
from src.probe_loss import dice_loss, combined_loss

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchvision.utils import make_grid
import torchvision.transforms as T

from accelerate import Accelerator

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # 1. 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"].get("seed", 42))

    # 2. 加载数据
    img_dir = config["data"]["img_dir"]
    mask_dir = config["data"]["mask_dir"]
    train_list = load_file_list(config["data"]["train_list"])
    val_list = load_file_list(config["data"]["val_list"]) if config["training"].get("use_validation", True) else []

    img_size = config["data"].get("img_size", 256)
    mask_threshold = config["data"].get("mask_threshold", 128)
    ignore_label = config["model"].get("ignore_label", None)

    train_dataset = ImageMaskDataset(img_dir, mask_dir, train_list, img_size=img_size, ignore_label=ignore_label, mask_threshold=mask_threshold)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4)
    if val_list:
        val_dataset = ImageMaskDataset(img_dir, mask_dir, val_list, img_size=img_size, ignore_label=ignore_label, mask_threshold=mask_threshold)
        val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)
    else:
        val_loader = None

    # 3. 加载冻结的U-Net
    from bypassunet.bypass_unet import MeanBypassUNet
    unet = MeanBypassUNet.from_pretrained(config["model"]["unet_ckpt"])
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    # 4. 创建分割头
    seg_head = create_model(
        config["model"]["seg_head"]["name"],
        num_classes=config["model"]["seg_head"]["num_classes"],
        in_channels=config["model"]["seg_head"]["in_channels"],
        output_size=(img_size, img_size)
    )

    # 5. 构建DiffSegNet
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    model = DiffSegNet(
        unet=unet,
        seg_head=seg_head,
        probe_block_name=config["model"]["probe_block_name"],
        probe_timestep=config["model"]["probe_timestep"],
        scheduler=scheduler
    )

    # 7. Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # 6. 损失与优化器
    weights = torch.tensor([1.0, 1.5], device=device)  # [背景, 裂缝] 可根据实际调整
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1 if ignore_label is not None else None)
    optimizer = Adam(model.seg_head.parameters(), lr=config["training"]["lr"])

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader:
        val_loader = accelerator.prepare(val_loader)

    output_dir = config["output_dir"]
    ensure_dir(output_dir)
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs"))
    else:
        writer = None

    # 8. 训练与验证循环
    best_miou = 0
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        train_loss = 0
        train_ce_loss = 0
        train_dice_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=not accelerator.is_local_main_process):
            images, masks = images.to(device), masks.to(device)
            if epoch == 1 and train_loss == 0 and accelerator.is_main_process:
                feats = model.extract_feature(images)
                print("extract_feature shape:", feats.shape)
            logits = model(images)
            loss, ce, dice = combined_loss(logits, masks, criterion, dice_weight=0.5)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_ce_loss += ce.item() * images.size(0)
            train_dice_loss += dice.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        train_ce_loss /= len(train_loader.dataset)
        train_dice_loss /= len(train_loader.dataset)
        if writer and accelerator.is_main_process:
            writer.add_scalar("Train/Loss", train_loss, epoch)
            writer.add_scalar("Train/CE_Loss", train_ce_loss, epoch)
            writer.add_scalar("Train/Dice_Loss", train_dice_loss, epoch)

        # 验证
        if val_loader and (epoch % config["training"].get("val_interval", 1) == 0):
            model.eval()
            all_preds, all_gts, all_probs = [], [], []
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", disable=not accelerator.is_local_main_process):
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)[:, 1] if logits.shape[1] > 1 else torch.sigmoid(logits[:, 0])
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.append(preds)
                    all_gts.append(masks.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
            all_preds = np.concatenate(all_preds)
            all_gts = np.concatenate(all_gts)
            all_probs = np.concatenate(all_probs)
            _, miou = compute_iou(all_preds, all_gts, config["model"]["seg_head"]["num_classes"], ignore_label)
            precision, recall = compute_precision_recall(all_preds, all_gts, config["model"]["seg_head"]["num_classes"], ignore_label)
            if writer and accelerator.is_main_process:
                writer.add_scalar("Val/mIoU", miou, epoch)
                writer.add_scalar("Val/Precision", precision, epoch)
                writer.add_scalar("Val/Recall", recall, epoch)
            if accelerator.is_main_process:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, CE={train_ce_loss:.4f}, Dice={train_dice_loss:.4f}, Val mIoU={miou:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

            # 展示部分验证图片及预测
            if writer and accelerator.is_main_process:
                N = 8  # 展示前N张
                val_images = []
                val_preds = []
                val_gts = []
                val_feats = []
                palette = get_palette(config["model"]["seg_head"]["num_classes"])
                to_tensor = T.ToTensor()
                with torch.no_grad():
                    val_iter = iter(val_loader)
                    for i in range(N):
                        try:
                            images, masks = next(val_iter)
                        except StopIteration:
                            break
                        images = images.to(device)
                        # 提取特征
                        feats = model.extract_feature(images)  # [B, C, H, W]
                        logits = model(images)
                        preds = logits.argmax(dim=1).cpu().numpy()
                        # 原图
                        img = images[0].cpu()
                        if img.shape[0] == 1:
                            img = img.repeat(3, 1, 1)
                        # 预测mask可视化
                        pred_mask = preds[0]
                        pred_vis = colorize_mask(pred_mask, palette).convert("RGB")
                        # GT mask可视化
                        gt_mask = masks[0].cpu().numpy()
                        gt_vis = colorize_mask(gt_mask, palette).convert("RGB")
                        # 特征可视化（灰度图，支持任意通道数）
                        feat = feats[0]  # [C, H, W]
                        feat_gray = feat.mean(dim=0, keepdim=True)  # [1, H, W]
                        feat_gray = (feat_gray - feat_gray.min()) / (feat_gray.max() - feat_gray.min() + 1e-8)
                        val_feats.append(feat_gray.cpu())
                        val_images.append(img)
                        val_preds.append(to_tensor(pred_vis))
                        val_gts.append(to_tensor(gt_vis))
                if val_images:
                    img_grid = make_grid(val_images, nrow=N)
                    pred_grid = make_grid(val_preds, nrow=N)
                    gt_grid = make_grid(val_gts, nrow=N)
                    feat_grid = make_grid(val_feats, nrow=N)
                    writer.add_image("Val/Image", img_grid, epoch)
                    writer.add_image("Val/Pred", pred_grid, epoch)
                    writer.add_image("Val/GT", gt_grid, epoch)
                    writer.add_image("Val/Feature", feat_grid, epoch)

            # 保存最佳模型
            if accelerator.is_main_process and miou > best_miou:
                best_miou = miou
                torch.save(model.seg_head.state_dict(), os.path.join(output_dir, "seg_head_best.pth"))
        else:
            if accelerator.is_main_process:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, CE={train_ce_loss:.4f}, Dice={train_dice_loss:.4f}")

    if writer and accelerator.is_main_process:
        writer.close()

    # 9. 最终预测与保存
    if accelerator.is_main_process:
        print("Loading best seg_head and predicting on validation set...")
        model.seg_head.load_state_dict(torch.load(os.path.join(output_dir, "seg_head_best.pth"), map_location=device))
        model.eval()
        if val_loader:
            all_preds = []
            all_gts = []
            all_probs = []
            for images, masks in tqdm(val_loader, desc="Final Prediction"):
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1] if logits.shape[1] > 1 else torch.sigmoid(logits[:, 0])
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_gts.append(masks.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
            all_preds = np.concatenate(all_preds)
            all_gts = np.concatenate(all_gts)
            all_probs = np.concatenate(all_probs)
            # 保存可视化
            palette = get_palette(config["model"]["seg_head"]["num_classes"])
            ensure_dir(os.path.join(output_dir, "predictions"))
            for idx, name in enumerate(val_list):
                mask = all_preds[idx]
                save_path = os.path.join(output_dir, "predictions", name + ".png")
                save_predictions(mask, save_path, palette)
            # 最终AP、ODS、OIS
            ap = compute_ap(all_probs, all_gts, ignore_index=ignore_label)
            ods, ois = compute_ods_ois(all_probs, all_gts, ignore_index=ignore_label)
            print(f"Final AP: {ap:.4f}, Final ODS: {ods:.4f}, Final OIS: {ois:.4f}")
        print("Done.")

if __name__ == "__main__":
    import sys
    if "--config" not in sys.argv:
        sys.argv += ["--config", "probe_seg/train_probe.yaml"]
    main()