import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline

from bypass_unet import MeanBypassUNet
from bypass_pipeline import BypassDDPMPipeline


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Optional[str]) -> Dict:
    if not path:
        return {}
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("读取 YAML 需要安装 pyyaml：pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _list_checkpoints(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"checkpoints_root 不存在: {root}")
    # 允许用户直接传入单个 checkpoint-xxxx 目录
    if root.is_dir() and root.name.startswith("checkpoint-") and (root / "unet").exists():
        return [root]
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]

    def _key(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except Exception:
            return 10**18

    return sorted(dirs, key=_key)


def _pick_subfolder(ckpt_dir: Path, use_ema: bool) -> str:
    if use_ema and (ckpt_dir / "unet_ema").exists():
        return "unet_ema"
    return "unet"


def _is_bypass_config(cfg: Dict) -> bool:
    return ("hidden_dim" in cfg) or ("use_gnorm" in cfg)


def _load_unet(ckpt_dir: Path, subfolder: str, device: torch.device, dtype: torch.dtype):
    cfg_path = ckpt_dir / subfolder / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少 config.json: {cfg_path}")
    cfg = _read_json(cfg_path)

    if _is_bypass_config(cfg):
        model = MeanBypassUNet.from_pretrained(str(ckpt_dir), subfolder=subfolder, local_files_only=True)
        kind = "bypassunet"
    else:
        try:
            model = UNet2DModel.from_pretrained(str(ckpt_dir), subfolder=subfolder, local_files_only=True)
            kind = "unet"
        except Exception:
            model = MeanBypassUNet.from_pretrained(str(ckpt_dir), subfolder=subfolder, local_files_only=True)
            kind = "bypassunet"

    model.to(device=device, dtype=dtype)
    model.eval()
    return model, kind, cfg


def _build_scheduler(args) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=int(args.ddpm_num_steps),
        beta_schedule=str(args.ddpm_beta_schedule),
        prediction_type=str(args.prediction_type),
    )


def _get_sample_shape(model_cfg: Dict, override_resolution: Optional[int]) -> Tuple[int, int, int]:
    sample_size = int(model_cfg.get("sample_size", 64))
    in_channels = int(model_cfg.get("in_channels", 3))
    if override_resolution is not None:
        sample_size = int(override_resolution)
    return in_channels, sample_size, sample_size


def _save_images_01(images_01: np.ndarray, out_dir: Path, indices: List[int]):
    # images_01: [B,H,W,C] in [0,1]
    out_dir.mkdir(parents=True, exist_ok=True)

    images_01 = np.asarray(images_01)
    imgs_u8 = (np.clip(images_01, 0.0, 1.0) * 255.0).round().astype("uint8")

    for i, idx in enumerate(indices):
        img = imgs_u8[i]

        # 兼容灰度：C==1 时 (H,W,1) -> (H,W)，用 L 模式保存
        if img.ndim == 3 and img.shape[-1] == 1:
            img2d = img[..., 0]
            Image.fromarray(img2d, mode="L").save(out_dir / f"{int(idx):06d}.png")
        else:
            # RGB/RGBA 等保持原样
            Image.fromarray(img).save(out_dir / f"{int(idx):06d}.png")


def _chunks(xs: List[int], bs: int):
    for i in range(0, len(xs), bs):
        yield xs[i : i + bs]


def parse_args():
    p = argparse.ArgumentParser(description="使用 accelerate 推理多个 checkpoint（UNet/MeanBypassUNet），用 pipeline 简化采样逻辑。")
    p.add_argument("--config", type=str, default=None, help="YAML 配置文件路径（命令行参数优先）")

    # 注意：不要 required=True，让 YAML 可以提供必需字段；解析后手动校验
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--checkpoints_root", type=str, default=None, help="包含 checkpoint-* 子目录的根目录，或单个 checkpoint-xxxx 目录")
    src.add_argument("--checkpoints", type=str, nargs="+", default=None, help="显式指定若干 checkpoint 目录路径")

    p.add_argument("--output_root", type=str, default=None)

    p.add_argument("--num_images", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_inference_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_ema", action="store_true")

    p.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    p.add_argument("--dtype", type=str, default=None, choices=["fp16", "bf16", "fp32"])  # 兼容旧 YAML

    p.add_argument("--ddpm_num_steps", type=int, default=1000)
    p.add_argument("--ddpm_beta_schedule", type=str, default="squaredcos_cap_v2")
    p.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "sample"])

    p.add_argument("--resolution", type=int, default=None)

    # 先解析一次拿到 --config
    args0, _ = p.parse_known_args()

    # 用 YAML 注入 defaults（命令行显式传参会覆盖这些 defaults）
    if args0.config:
        yaml_cfg = _load_yaml(args0.config) or {}
        known_dests = {a.dest for a in p._actions}
        filtered = {k: v for k, v in yaml_cfg.items() if k in known_dests}
        p.set_defaults(**filtered)

    args = p.parse_args()

    # dtype -> mixed_precision 映射（如果用户只在 YAML 里配了 dtype）
    if args.dtype and (args.mixed_precision == "no"):
        args.mixed_precision = {"fp16": "fp16", "bf16": "bf16", "fp32": "no"}[args.dtype]

    # 手动校验必需参数（允许来自 YAML）
    if not args.output_root:
        raise SystemExit("缺少 output_root：请在 YAML 里配置 output_root，或命令行传 --output_root")
    if bool(args.checkpoints_root) + bool(args.checkpoints) != 1:
        raise SystemExit("必须且只能提供 checkpoints_root 或 checkpoints 其中之一（可在 YAML 或命令行中设置）")

    return args


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=str(args.mixed_precision))

    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    device = accelerator.device
    if device.type == "cpu":
        dtype = torch.float32

    ckpts = _list_checkpoints(Path(args.checkpoints_root)) if args.checkpoints_root else [Path(x) for x in (args.checkpoints or [])]
    if not ckpts:
        raise RuntimeError("未找到任何 checkpoint")

    out_root = Path(args.output_root)
    if accelerator.is_main_process:
        out_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    rank = accelerator.process_index
    world = accelerator.num_processes

    for ckpt_dir in ckpts:
        subfolder = _pick_subfolder(ckpt_dir, use_ema=bool(args.use_ema))
        unet, kind, model_cfg = _load_unet(ckpt_dir, subfolder, device=device, dtype=dtype)
        scheduler = _build_scheduler(args)

        pipe = BypassDDPMPipeline(unet=unet, scheduler=scheduler) if kind == "bypassunet" else DDPMPipeline(unet=unet, scheduler=scheduler)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=not accelerator.is_main_process)

        c, h, w = _get_sample_shape(model_cfg, override_resolution=args.resolution)

        out_dir = out_root / ckpt_dir.name / subfolder
        if accelerator.is_main_process:
            out_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "checkpoint": str(ckpt_dir),
                "subfolder": subfolder,
                "model_kind": kind,
                "num_images": int(args.num_images),
                "batch_size_per_process": int(args.batch_size),
                "num_inference_steps": int(args.num_inference_steps),
                "seed_base": int(args.seed),
                "shape": {"in_channels": c, "height": h, "width": w},
                "accelerate": {"num_processes": int(world)},
                "scheduler": {
                    "num_train_timesteps": int(scheduler.config.num_train_timesteps),
                    "beta_schedule": str(scheduler.config.beta_schedule),
                    "prediction_type": getattr(scheduler.config, "prediction_type", None),
                },
            }
            (out_dir / "infer_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        accelerator.wait_for_everyone()

        total = int(args.num_images)
        my_indices = list(range(rank, total, world))
        chunks = list(_chunks(my_indices, max(1, int(args.batch_size))))
        pbar = tqdm(chunks, desc=f"{ckpt_dir.name}/{subfolder} (world={world})", disable=not accelerator.is_main_process)

        for indices in pbar:
            gens = [torch.Generator(device=device).manual_seed(int(args.seed) + int(i)) for i in indices]
            with accelerator.autocast():
                out = pipe(
                    batch_size=len(indices),
                    num_inference_steps=int(args.num_inference_steps),
                    generator=gens,
                    output_type="numpy",
                )
            _save_images_01(out.images, out_dir=out_dir, indices=indices)

        accelerator.wait_for_everyone()
        del pipe, unet
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()