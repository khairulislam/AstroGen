# Paper: https://arxiv.org/abs/2006.11239 (DDPM); SR3-style channel-concat conditioning
# Corruption recipe (Gaussian noise + blur): family C's SR repos, e.g.
# https://github.com/ML4SCI/DeepLense/tree/main/Super_Resolution_Atal_Gupta
# 1D (spectra) counterpart: https://github.com/conor-horgan/spectrai
# `spectral_denoising` config
"""Conditional DDPM denoising for gravitational-lens images and spectra."""

import torch
from torch import nn

from astrogen.models import build_conditional_ddpm


class DenoisingDDPM(nn.Module):
    """Conditional DDPM that restores a corrupted image or spectrum.

    Reuses :class:`DDPM`'s channel-concatenated conditioning path from
    :class:`~astrogen.tasks.super_resolution.SuperResolutionDDPM`: the
    corrupted sample is concatenated, channel-wise, with the noisy clean
    sample at every denoising step. Unlike super-resolution, the corrupted
    and clean samples share the same spatial size, so no resizing is applied.
    Set ``dimensionality="1d"`` for spectra (shape ``(batch, channels,
    length)``); defaults to ``"2d"`` for lens images (shape ``(batch,
    channels, height, width)``). Corrupted and clean samples must have the
    same channel count and values normalized to approximately ``[-1, 1]``;
    corruption (e.g. Gaussian noise plus blur) is applied by the caller
    before training.
    """

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 32,
        timesteps: int = 1_000,
        dimensionality: str = "2d",
        unet_kwargs: dict | None = None,
        scheduler_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.ddpm = build_conditional_ddpm(
            image_channels, base_channels, timesteps, dimensionality, unet_kwargs, scheduler_kwargs
        )

    def forward(self, corrupted: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Return the DDPM training loss for a corrupted/clean sample pair."""
        return self.ddpm(clean, condition=corrupted)

    @torch.no_grad()
    def sample(self, corrupted: torch.Tensor) -> torch.Tensor:
        """Denoise ``corrupted`` images or spectra."""
        sample_size = corrupted.shape[-1]
        return self.ddpm.sample(corrupted.shape[0], sample_size=sample_size, condition=corrupted)
