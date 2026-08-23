# Paper: https://arxiv.org/abs/2006.11239
# Reference: https://github.com/ML4SCI/DeepLense/blob/main/Difflense_Aleksandr_Duplinskii/Unconditional_diffusion/model_grav.py
"""Unconditional gravitational-lens image generation with a DDPM."""

import torch
from torch import nn

from astrogen.models import GaussianDiffusion, TimeConditionedUNet2D


class LensImageDDPM(nn.Module):
    """An unconditional DDPM baseline for normalized single-channel lens images.

    Images passed to :meth:`forward` must have shape ``(batch, 1, height, width)``
    and values normalized to approximately ``[-1, 1]``.
    """

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 32,
        timesteps: int = 1_000,
    ) -> None:
        super().__init__()
        self.denoiser = TimeConditionedUNet2D(
            in_channels=image_channels,
            out_channels=image_channels,
            base_channels=base_channels,
        )
        self.diffusion = GaussianDiffusion(timesteps=timesteps)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return the DDPM training loss for a batch of lens images."""
        return self.diffusion.loss(self.denoiser, images)

    def sample(self, count: int, image_size: int = 64) -> torch.Tensor:
        """Sample ``count`` generated single-channel lens images."""
        return self.diffusion.sample(
            self.denoiser,
            (count, self.denoiser.output.out_channels, image_size, image_size),
        )
