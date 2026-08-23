"""Convolutional blocks shared by AstroGen models."""

import torch
from torch import nn


def _group_count(channels: int) -> int:
    """Return a GroupNorm group count that divides ``channels``."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock2D(nn.Module):
    """A time-conditioned residual block for two-dimensional diffusion."""

    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.SiLU()

    def forward(self, images: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``(batch, channels, height, width)`` images."""
        hidden = self.conv1(self.activation(self.norm1(images)))
        hidden = hidden + self.time_projection(self.activation(time_embedding))[..., None, None]
        hidden = self.conv2(self.activation(self.norm2(hidden)))
        return hidden + self.skip(images)
