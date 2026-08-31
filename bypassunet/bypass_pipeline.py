import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Union, Tuple

from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class BypassDDPMPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, np.ndarray] = None


class BypassDDPMPipeline(DDPMPipeline):
    """
    仅完整采样分支（主干+旁路融合），即 MeanBypassUNet 的 "sample" 输出
    """

    # 关键：不要带 **kwargs，否则 diffusers 会把它当成一个名为 'kwargs' 的组件来校验
    def __init__(self, unet, scheduler: DDPMScheduler):
        super().__init__(unet=unet, scheduler=scheduler)
        self.scheduler: DDPMScheduler = scheduler

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        generator: Optional[Union[torch.Generator, list]] = None,
        output_type: str = "numpy",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[BypassDDPMPipelineOutput, Tuple]:
        device = self.unet.device
        sample_size = int(self.unet.config.sample_size)
        in_channels = int(self.unet.config.in_channels)

        self.scheduler.set_timesteps(int(num_inference_steps), device=device)
        timesteps = self.scheduler.timesteps

        # 支持 generator 为 torch.Generator 或 list[torch.Generator]
        image = randn_tensor(
            (batch_size, in_channels, sample_size, sample_size),
            generator=generator,
            device=device,
            dtype=self.unet.dtype,
        )

        for t in self.progress_bar(timesteps):
            t_tensor = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
            out = self.unet(image, t_tensor, return_dict=True)

            sample_pred = getattr(out, "sample", None)
            if sample_pred is None and isinstance(out, dict):
                sample_pred = out.get("sample", None)
            if sample_pred is None:
                raise RuntimeError("UNet 输出不包含 'sample' 字段")

            image = self.scheduler.step(sample_pred, t, image).prev_sample

        images = self._postprocess(image, output_type=output_type)

        if not return_dict:
            return (images,)
        return BypassDDPMPipelineOutput(images=images)

    def _postprocess(self, image: torch.Tensor, output_type: str):
        if output_type == "numpy":
            image = (image.permute(0, 2, 3, 1).float().cpu().numpy() + 1.0) / 2.0
            return np.clip(image, 0.0, 1.0)
        return image