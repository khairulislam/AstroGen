# Paper: https://arxiv.org/abs/2006.11239 (DDPM); SR3-style channel-concat conditioning
# Corruption recipe (Gaussian noise + blur): family C's SR repos, e.g.
# https://github.com/ML4SCI/DeepLense/tree/main/Super_Resolution_Atal_Gupta
"""Conditional DDPM denoising for gravitational-lens images."""

import torch
from torch import nn

from astrogen.models import DDPM


class DenoisingDDPM(nn.Module):
    """Conditional DDPM that restores a corrupted lens image.

    Reuses :class:`DDPM`'s channel-concatenated conditioning path from
    :class:`~astrogen.tasks.super_resolution.SuperResolutionDDPM`: the
    corrupted image is concatenated, channel-wise, with the noisy clean image
    at every denoising step. Unlike super-resolution, the corrupted and clean
    images share the same spatial size, so no resizing is applied. Corrupted
    and clean images must have the same channel count and values normalized
    to approximately ``[-1, 1]``; corruption (e.g. Gaussian noise plus blur)
    is applied by the caller before training.
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
        self.ddpm = DDPM(
            image_channels=image_channels,
            base_channels=base_channels,
            timesteps=timesteps,
            condition_channels=image_channels,
            unet_kwargs=unet_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )

    def forward(self, corrupted: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Return the DDPM training loss for a corrupted/clean image pair."""
        return self.ddpm(clean, condition=corrupted)

    @torch.no_grad()
    def sample(self, corrupted: torch.Tensor) -> torch.Tensor:
        """Denoise ``corrupted`` images."""
        image_size = corrupted.shape[-1]
        return self.ddpm.sample(corrupted.shape[0], image_size=image_size, condition=corrupted)
