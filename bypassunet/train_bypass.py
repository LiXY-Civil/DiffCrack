import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import datasets
import torch
import yaml
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from bypass_unet import MeanBypassUNet
from bypass_total_loss import compute_bypass_total_loss
from bypass_pipeline import BypassDDPMPipeline  # 新增导入




# 如果未安装最低版本的 diffusers，将会报错。移除此行需自担风险。
check_min_version("0.24.0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    从一维 numpy 数组中为一批索引提取值。

    :param arr: 一维 numpy 数组。
    :param timesteps: 一个包含数组索引的 tensor。
    :param broadcast_shape: K 维更大 shape，batch 维等于 timesteps 长度。
    :return: 形状为 [batch_size, 1, ...] 的 tensor，具有 K 维。
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def load_config_from_yaml(yaml_path):
    """从YAML文件加载配置"""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="训练脚本的简单示例。")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "要用于训练的数据集名称（来自 HuggingFace hub），也可以是本地数据集路径，"
            "或 HF Datasets 能识别的文件夹。"
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="数据集的 config，如果只有一个 config 可留空。",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="要训练的 UNet 模型配置，留空则使用标准 DDPM 配置。",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "包含训练数据的文件夹。结构需参考 "
            "https://huggingface.co/docs/datasets/image_dataset#imagefolder。"
            "特别是需有 `metadata.jsonl` 文件。若指定了 `dataset_name` 则忽略此项。"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="模型预测和检查点保存的输出目录。",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="下载的模型和数据集的缓存目录。",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="输入图像分辨率，所有训练/验证图像将被调整到该分辨率。",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="是否对输入图像进行中心裁剪。如果不设置，则随机裁剪。图像会先被调整到指定分辨率。",
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="是否随机水平翻转图像",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="训练 dataloader 的每设备 batch 大小。"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="评估时生成图像的数量。"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="用于数据加载的子进程数。0 表示在主进程中加载数据。",
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10, help="训练过程中保存图像的频率（以 epoch 为单位）。")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="训练过程中保存模型的频率（以 epoch 为单位）。"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="累计多少步后再进行一次反向传播/参数更新。",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="初始学习率（可能有 warmup）。",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            '学习率调度器类型。可选 ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="学习率 warmup 步数。"
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="Adam 优化器的 beta1 参数。")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam 优化器的 beta2 参数。")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Adam 优化器的权重衰减。"
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam 优化器的 epsilon。")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="是否对最终模型权重使用指数滑动平均（EMA）。",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="EMA 衰减的逆伽马值。")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="EMA 衰减的幂值。")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="EMA 最大衰减值。")
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "实验追踪和日志记录方式，[tensorboard](https://www.tensorflow.org/tensorboard) 或 [wandb](https://www.wandb.ai)"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) 日志目录。默认为 *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***。"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练用：local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "是否使用混合精度。可选 fp16 或 bf16（bfloat16）。Bf16 需 PyTorch >= 1.10 且显卡支持。"
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="模型预测 'epsilon'/噪声残差，还是直接预测重建图像 'x0'。",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "每 X 步保存一次训练状态检查点。仅用于通过 `--resume_from_checkpoint` 恢复训练。"
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("最多保留多少个检查点。"),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "是否从之前的检查点恢复训练。可用 `--checkpointing_steps` 保存的路径，或用 'latest' 自动选择最新检查点。"
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="是否使用 xformers。"
    )
    parser.add_argument("--lambda_bypass", type=float, default=1.0, help="旁路分支损失权重")
    parser.add_argument("--gamma", type=float, default=0.1, help="主干空间均值约束损失权重")

    # 先解析一次，获取 config 路径
    args, _ = parser.parse_known_args()
    if args.config is not None:
        yaml_cfg = load_config_from_yaml(args.config)
        for k, v in yaml_cfg.items():
            if hasattr(args, k):
                old_v = getattr(args, k)
                if old_v is not None and not isinstance(v, type(old_v)):
                    try:
                        if isinstance(old_v, bool):
                            v = str(v).lower() in ("true", "1", "yes")
                        else:
                            v = type(old_v)(v)
                    except Exception:
                        pass
                setattr(args, k, v)
    # 再次解析命令行，命令行参数优先
    args = parser.parse_args()
    if args.config is not None:
        yaml_cfg = load_config_from_yaml(args.config)
        for k, v in yaml_cfg.items():
            if hasattr(args, k):
                old_v = getattr(args, k)
                if old_v is not None and not isinstance(v, type(old_v)):
                    try:
                        if isinstance(old_v, bool):
                            v = str(v).lower() in ("true", "1", "yes")
                        else:
                            v = type(old_v)(v)
                    except Exception:
                        pass
                setattr(args, k, v)
    return args


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # 大数据集或高分辨率时设置较大超时
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("如需使用 tensorboard 记录日志，请先安装 tensorboard。")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("如需使用 wandb 记录日志，请先安装 wandb。")
        import wandb

    # accelerate 0.16.0 及以上支持自定义保存
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # 注册自定义保存和加载钩子，使 accelerator.save_state(...) 以更友好的格式序列化
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    # 确保权重已弹出，避免重复保存
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), MeanBypassUNet)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # 弹出模型，避免重复加载
                model = models.pop()
                # 以 diffusers 风格加载到模型
                load_model = MeanBypassUNet.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 每个进程都记录一次日志，便于调试
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 处理输出目录创建
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 初始化自定义模型
    if args.model_config_name_or_path is None:
        model = MeanBypassUNet(
            sample_size=args.resolution,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            hidden_dim=128,
            use_gnorm=True,
        )
    else:
        config = MeanBypassUNet.load_config(args.model_config_name_or_path)
        model = MeanBypassUNet.from_config(config)

    # 创建模型的 EMA
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=MeanBypassUNet,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 在部分 GPU 上无法用于训练。如遇问题请升级到至少 0.0.17。详见 https://huggingface.co/docs/diffusers/main/en/optimization/xformers"
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("未检测到 xformers，请确保正确安装。")

    # 初始化调度器
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    # 初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 获取数据集：可指定 hub 数据集或本地文件
    # 分布式训练下，load_dataset 保证只有一个进程下载数据
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
        # 更多自定义图片加载见
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # 数据预处理与 DataLoader 创建
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    logger.info(f"数据集大小: {len(dataset)}")

    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # 初始化学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # 用 accelerator 准备所有内容
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # 初始化日志追踪器，并存储配置
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** 开始训练 *****")
    logger.info(f"  样本数 = {len(dataset)}")
    logger.info(f"  训练轮数 = {args.num_epochs}")
    logger.info(f"  每设备 batch 大小 = {args.train_batch_size}")
    logger.info(f"  总训练 batch 大小（含并行、分布式与累积） = {total_batch_size}")
    logger.info(f"  梯度累计步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # 如有需要，从之前的检查点加载权重和状态
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # 获取最新的检查点
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"检查点 '{args.resume_from_checkpoint}' 不存在，将重新开始训练。"
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"从检查点 {path} 恢复训练")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # 开始训练！
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # 跳过已恢复的步数
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["input"].to(weight_dtype)
            # 采样噪声并加到图像上
            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
            bsz = clean_images.shape[0]
            # 为每张图片采样一个随机时间步
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # 按每个时间步的噪声幅度向干净图像加噪声（前向扩散过程）
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                output = model(noisy_images, timesteps, return_dict=True)
                # output为dict，需用下标访问
                noise_mu = noise.mean(dim=(2, 3))  # [B, C]
                noise_var = noise - noise_mu[:, :, None, None]  # [B, C, H, W]
                loss, loss_dict = compute_bypass_total_loss(
                    residual_pred=output["residual_pred"],
                    noise_var=noise_var,
                    mean_pred=output["mean_pred"],
                    noise_mu=noise_mu,
                    lambda_=getattr(args, "lambda_bypass", 1.0),
                    gamma=getattr(args, "gamma", 0.1),
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 检查 accelerator 是否已完成优化步
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # 保存前检查是否超出最大检查点数
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # 保存新检查点前，最多只能有 checkpoints_total_limit-1 个
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"已有 {len(checkpoints)} 个检查点，将移除 {len(removing_checkpoints)} 个"
                                )
                                logger.info(f"移除检查点: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"已保存状态到 {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            logs.update(loss_dict)
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # 生成样本图像用于可视化
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                # 使用自定义Pipeline
                pipeline = BypassDDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                # 推理阶段运行 pipeline（采样随机噪声并去噪）
                pipeline_output = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="numpy",
                )
                images = pipeline_output.images  # 只保留完整采样

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                # 反归一化图像并保存到 tensorboard
                images_processed = (images * 255).round().astype("uint8")

                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples/full", images_processed.transpose(0, 3, 1, 2), epoch)
                elif args.logger == "wandb":
                    accelerator.get_tracker("wandb").log(
                        {
                            "test_samples/full": [wandb.Image(img) for img in images_processed],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # 保存模型
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                # 保存自定义Pipeline
                save_dir = os.path.join(args.output_dir, f"model_epoch_{epoch}")
                pipeline = BypassDDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(save_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)