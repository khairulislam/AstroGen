# Paper: https://arxiv.org/abs/2006.11239 (DDPM); SR3-style channel-concat conditioning
# Corruption recipe (Gaussian noise + blur): family C's SR repos, e.g.
# https://github.com/ML4SCI/DeepLense/tree/main/Super_Resolution_Atal_Gupta
# 1D (spectra) counterpart: https://github.com/conor-horgan/spectrai
# `spectral_denoising` config
"""Conditional DDPM denoising for gravitational-lens images and spectra."""

import math

import torch
from diffusers import DDPMScheduler, UNet1DModel, UNet2DModel
from torch import nn

from astrogen.models import DDPM


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
        if dimensionality not in ("1d", "2d"):
            raise ValueError(f"dimensionality must be '1d' or '2d', got {dimensionality!r}")
        unet_cls = UNet1DModel if dimensionality == "1d" else UNet2DModel
        block = "DownBlock1D" if dimensionality == "1d" else "DownBlock2D"
        up_block = "UpBlock1D" if dimensionality == "1d" else "UpBlock2D"
        unet_config = dict(
            in_channels=image_channels * 2,
            out_channels=image_channels,
            layers_per_block=1,
            block_out_channels=(base_channels, base_channels * 2, base_channels * 4),
            down_block_types=(block, block, block),
            up_block_types=(up_block, up_block, up_block),
            norm_num_groups=math.gcd(base_channels, 32),
        )
        unet_config.update(unet_kwargs or {})
        scheduler_config = dict(num_train_timesteps=timesteps)
        scheduler_config.update(scheduler_kwargs or {})
        self.ddpm = DDPM(
            unet_cls(**unet_config),
            DDPMScheduler(**scheduler_config),
            image_channels=image_channels,
            condition_channels=image_channels,
        )

    def forward(self, corrupted: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Return the DDPM training loss for a corrupted/clean sample pair."""
        return self.ddpm(clean, condition=corrupted)

    @torch.no_grad()
    def sample(self, corrupted: torch.Tensor) -> torch.Tensor:
        """Denoise ``corrupted`` images or spectra."""
        sample_size = corrupted.shape[-1]
        return self.ddpm.sample(corrupted.shape[0], sample_size=sample_size, condition=corrupted)
