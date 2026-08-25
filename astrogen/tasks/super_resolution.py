# Reference: https://github.com/ML4SCI/DeepLense/blob/main/Super_Resolution_Atal_Gupta/models/cgdpm.ipynb
"""Diffusion-based super-resolution for gravitational-lens images."""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional

from astrogen.models import CGDPM


class SuperResolutionDDPM(nn.Module):
    """DeepLense CGDPM that super-resolves a low-resolution lens image.

    Follows the reference conditioned U-Net, diffusion schedule, L1 noise
    objective, and reverse process. The low-resolution image is bilinearly
    upsampled and injected into every residual block. Low- and
    high-resolution images must have the same channel count and values
    normalized to ``[-1, 1]`` — diffusers' model space, and the domain
    :class:`~astrogen.models.CGDPM` samples in. Astronomical data spanning a
    wide dynamic range needs a stretch (e.g. ``asinh`` on sky-subtracted,
    noise-scaled pixels) before that mapping, not a bare min-max rescale;
    plain min-max leaves the signal narrower than the sampler's own noise
    floor. Preparing that stretch is the caller's job, as PLAN.md keeps
    dataset preparation outside the library.
    """

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        groups: int = 8,
        timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
    ) -> None:
        super().__init__()
        self.ddpm = CGDPM(
            image_channels=image_channels,
            condition_channels=image_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            groups=groups,
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
        )

    @staticmethod
    def _upsample(low_resolution: torch.Tensor, size: int) -> torch.Tensor:
        return functional.interpolate(
            low_resolution, size=(size, size), mode="bilinear", align_corners=False
        )

    def forward(self, low_resolution: torch.Tensor, high_resolution: torch.Tensor) -> torch.Tensor:
        """Return the DDPM training loss for a low-/high-resolution image pair."""
        condition = self._upsample(low_resolution, high_resolution.shape[-1])
        return self.ddpm(high_resolution, condition)

    @torch.no_grad()
    def sample(self, low_resolution: torch.Tensor, image_size: int) -> torch.Tensor:
        """Super-resolve ``low_resolution`` images to ``image_size`` by ``image_size``."""
        condition = self._upsample(low_resolution, image_size)
        return self.ddpm.sample(low_resolution.shape[0], sample_size=image_size, condition=condition)
