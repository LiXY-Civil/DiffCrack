import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

class ImageMaskDataset(Dataset):
    """
    用于扩散分割任务的标准图片-掩码数据集
    每个样本返回：(image_tensor, mask_tensor)
    """
    def __init__(self, img_dir, mask_dir, file_list, img_size=256, ignore_label=None, mask_threshold=128):
        """
        Args:
            img_dir: 图片文件夹路径
            mask_dir: 掩码文件夹路径
            file_list: 图片文件名列表（如 ['xxx.jpg', ...]）
            img_size: 图片和掩码resize到的目标尺寸
            ignore_label: 可选，mask中需要忽略的像素值
            mask_threshold: 二值化阈值，默认128
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.img_size = img_size
        self.ignore_label = ignore_label
        self.mask_threshold = mask_threshold

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # 可选：标准化
            # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_list[idx])
        mask_base = os.path.join(self.mask_dir, os.path.splitext(self.file_list[idx])[0])
        # 优先找png，否则找jpg
        if os.path.exists(mask_base + ".png"):
            mask_path = mask_base + ".png"
        elif os.path.exists(mask_base + ".jpg"):
            mask_path = mask_base + ".jpg"
        else:
            raise FileNotFoundError(f"Mask file not found: {mask_base}.png or .jpg")

        img = Image.open(img_path).convert('RGB')
        # img = Image.open(img_path).convert('L')   # 如果是灰度图，使用'L'模式
        img = self.img_transform(img)

        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        mask = np.array(mask)
        mask = (mask >= self.mask_threshold).astype(np.uint8)
        mask = torch.from_numpy(mask).long()

        if self.ignore_label is not None:
            mask[mask == self.ignore_label] = -1

        return img, mask

def load_file_list(txt_path):
    """
    从txt文件读取图片名列表
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]