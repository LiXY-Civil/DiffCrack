# 裂缝分割扩散模型（Crack Diffusion Segmentation）

这是一个基于扩散模型的二阶段语义分割项目，专门用于裂缝检测。该项目利用无分类引导的扩散模型进行裂缝图像生成，并从扩散过程中提取特征用于语义分割任务。

## 技术方法

该项目采用两阶段方法：

1. **扩散模型阶段**：基于guided diffusion中的Gaussian扩散过程和UNet网络，实现无分类引导的扩散模型用于裂缝图像生成
   - 保留多分类引导接口以支持未来多类型路面损伤检测
   - 提供无引导生成接口用于裂缝或损伤图片及其掩码的直接生成

2. **分割模型阶段**：从扩散模型的中间表示中提取特征，用于语义分割
   - 提取特定时间步和UNet上采样过程中的激活值
   - 通过特征融合策略组合不同时间步和模块的特征
   - 使用现有裂缝数据集的图像和掩码进行分割训练

## 项目结构

```
├── configs/                # 配置文件
│   ├── diffusion_config.json     # 扩散模型配置
│   ├── extract_config.json       # 特征提取配置
│   └── segmentation_config.json  # 分割模型配置
├── guided_diffusion/       # 扩散模型核心代码
│   ├── gaussian_diffusion.py     # 扩散过程实现
│   ├── unet.py                   # UNet网络结构
│   └── ...                       # 其他辅助模块
├── scripts/                # 执行脚本
│   ├── extract_features.py       # 特征提取脚本
│   └── ...                       # 其他实用脚本
├── src/                    # 源代码
│   ├── datasets/                 # 数据集处理
│   ├── models/                   # 模型定义
│   ├── train/                    # 训练逻辑
│   └── utils/                    # 工具函数
├── reference/              # 参考实现（仅供参考）
├── requirements.txt        # 项目依赖
└── README.md               # 项目文档
```

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd diffusion-segmentation-project

# 安装依赖
pip install -r requirements.txt

# 准备数据集（裂缝图像及对应分割掩码）
# 放置在data/cracks目录下，按照train/val/test结构组织
```

## 配置

项目使用JSON配置文件：

- **diffusion_config.json**: 控制扩散模型训练参数
- **extract_config.json**: 定义特征提取设置（时间步、网络模块）
- **segmentation_config.json**: 配置分割模型架构和训练参数

## 使用方法

### 1. 训练扩散模型

```bash
python src/train/train_diffusion.py --config configs/diffusion_config.json
```

### 2. 提取扩散模型特征

```bash
python scripts/extract_features.py --config configs/extract_config.json
```

### 3. 训练分割模型

```bash
python src/train/train_segmentation.py --config configs/segmentation_config.json
```

### 4. 推理与可视化

```bash
python scripts/inference.py --model_path results/segmentation/best_model.pt --image_path data/test_image.png
```

## 模型说明

### 扩散模型 (DiffusionModel)
- 基于Gaussian噪声逐步去噪的生成模型
- 支持条件生成和无条件生成
- 使用UNet架构进行图像特征提取

### 特征提取器 (DiffusionFeatureExtractor)
- 从扩散模型中提取中间表示
- 可指定提取的时间步和网络模块
- 提供特征缓存机制提高处理效率

### 分割模型 (DiffusionSegmentationModel)
- 利用提取的扩散模型特征进行语义分割
- 支持多种特征融合策略（注意力机制、简单连接等）
- 包含边缘注意力机制增强分割边界精度

## 参考与致谢

本项目基于Guided Diffusion实现，特别感谢以下相关工作：
- [Guided Diffusion](https://github.com/openai/guided-diffusion)
- [Annotator-free Diffusion Semantic Segmentation](https://github.com/wl-zhao/VPD)