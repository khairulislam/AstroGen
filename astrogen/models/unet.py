"""Small U-Net denoisers for astronomy images."""

import torch
from torch import nn
from torch.nn import functional as functional

from astrogen.layers import ResidualBlock2D, SinusoidalTimeEmbedding


class TimeConditionedUNet2D(nn.Module):
    """A compact 2D U-Net that predicts diffusion noise from image and time."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        time_channels: int = 128,
    ) -> None:
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_channels),
            nn.Linear(time_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.encoder1 = ResidualBlock2D(base_channels, base_channels, time_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.encoder2 = ResidualBlock2D(base_channels * 2, base_channels * 2, time_channels)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.middle = ResidualBlock2D(base_channels * 4, base_channels * 4, time_channels)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.decoder2 = ResidualBlock2D(base_channels * 4, base_channels * 2, time_channels)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.decoder1 = ResidualBlock2D(base_channels * 2, base_channels, time_channels)
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict noise for images whose height and width are divisible by four."""
        if images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        if images.shape[-2] % 4 or images.shape[-1] % 4:
            raise ValueError("image height and width must be divisible by four")

        time_embedding = self.time_embedding(timesteps)
        first = self.encoder1(self.input(images), time_embedding)
        second = self.encoder2(self.down1(first), time_embedding)
        middle = self.middle(self.down2(second), time_embedding)
        second_decoded = self.decoder2(torch.cat((self.up2(middle), second), dim=1), time_embedding)
        first_decoded = self.decoder1(torch.cat((self.up1(second_decoded), first), dim=1), time_embedding)
        return self.output(functional.silu(first_decoded))
