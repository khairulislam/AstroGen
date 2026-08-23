# Paper: https://arxiv.org/abs/2006.11239 (DDPM); SR3-style channel-concat conditioning
# Reference: https://github.com/ML4SCI/DeepLense/blob/main/Super_Resolution_Atal_Gupta/models/cgdpm.ipynb
"""Diffusion-based super-resolution for gravitational-lens images."""

import torch
from torch import nn
from torch.nn import functional

from astrogen.models import build_conditional_ddpm


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
        self.ddpm = build_conditional_ddpm(
            image_channels, base_channels, timesteps, "2d", unet_kwargs, scheduler_kwargs
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
