# Paper: https://arxiv.org/abs/2006.11239 (DDPM); SR3-style channel-concat conditioning
# Reference: https://github.com/ML4SCI/DeepLense/blob/main/Super_Resolution_Atal_Gupta/models/cgdpm.ipynb
"""Diffusion-based super-resolution for gravitational-lens images."""

import math

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch import nn
from torch.nn import functional

from astrogen.models import DDPM


class SuperResolutionDDPM(nn.Module):
    """Conditional DDPM that super-resolves a low-resolution lens image.

    Follows the SR3 recipe: the low-resolution image is bilinearly upsampled
    to the target resolution and concatenated, channel-wise, with the noisy
    high-resolution image at every denoising step. Low- and high-resolution
    images must have the same channel count and values normalized to
    approximately ``[-1, 1]``.
    """

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 32,
        timesteps: int = 1_000,
        unet_kwargs: dict | None = None,
        scheduler_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        unet_config = dict(
            in_channels=image_channels * 2,
            out_channels=image_channels,
            layers_per_block=1,
            block_out_channels=(base_channels, base_channels * 2, base_channels * 4),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
            norm_num_groups=math.gcd(base_channels, 32),
        )
        unet_config.update(unet_kwargs or {})
        scheduler_config = dict(num_train_timesteps=timesteps)
        scheduler_config.update(scheduler_kwargs or {})
        self.ddpm = DDPM(
            UNet2DModel(**unet_config),
            DDPMScheduler(**scheduler_config),
            image_channels=image_channels,
            condition_channels=image_channels,
        )

    @staticmethod
    def _upsample(low_resolution: torch.Tensor, size: int) -> torch.Tensor:
        return functional.interpolate(
            low_resolution, size=(size, size), mode="bilinear", align_corners=False
        )

    def forward(self, low_resolution: torch.Tensor, high_resolution: torch.Tensor) -> torch.Tensor:
        """Return the DDPM training loss for a low-/high-resolution image pair."""
        condition = self._upsample(low_resolution, high_resolution.shape[-1])
        return self.ddpm(high_resolution, condition=condition)

    @torch.no_grad()
    def sample(self, low_resolution: torch.Tensor, image_size: int) -> torch.Tensor:
        """Super-resolve ``low_resolution`` images to ``image_size`` by ``image_size``."""
        condition = self._upsample(low_resolution, image_size)
        return self.ddpm.sample(low_resolution.shape[0], sample_size=image_size, condition=condition)
