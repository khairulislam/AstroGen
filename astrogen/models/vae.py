# Paper: https://arxiv.org/abs/1312.6114 (VAE), https://openreview.net/forum?id=Sy2fzU9gl (beta-VAE)
# Reference: https://github.com/ML4SCI/DeepLense/blob/main/DeepLense_Diffusion_Rishi/models/vae.py
"""Convolutional variational autoencoders for astronomy images."""

import torch
from torch import nn
from torch.nn import functional as functional


class DeepLenseVAE(nn.Module):
    """A compact VAE based on DeepLense's convolutional lens-image model.

    Training images must have shape ``(batch, image_channels, 64, 64)`` and
    values in ``[0, 1]``.
    """

    def __init__(
        self,
        image_channels: int = 1,
        latent_dimension: int = 128,
        base_channels: int = 8,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_dimension < 1:
            raise ValueError("latent_dimension must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if beta < 0:
            raise ValueError("beta must be non-negative")

        self.image_channels = image_channels
        self.latent_dimension = latent_dimension
        self.beta = beta
        encoder_channels = [
            image_channels,
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
            base_channels * 32,
        ]
        encoder_layers: list[nn.Module] = []
        for in_channels, out_channels in zip(encoder_channels, encoder_channels[1:]):
            encoder_layers.extend(
                (
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)
        self.statistics = nn.Conv2d(
            encoder_channels[-1], 2 * latent_dimension, kernel_size=3, stride=2, padding=1
        )

        decoder_channels = [
            base_channels * 32,
            base_channels * 16,
            base_channels * 8,
            base_channels * 4,
            base_channels * 2,
            base_channels,
        ]
        self.latent_projection = nn.Linear(latent_dimension, decoder_channels[0])
        decoder_layers: list[nn.Module] = []
        for in_channels, out_channels in zip(decoder_channels, decoder_channels[1:]):
            decoder_layers.extend(
                (
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        decoder_layers.append(
            nn.ConvTranspose2d(
                decoder_channels[-1],
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the latent mean and log variance for 64 by 64 images."""
        if images.ndim != 4 or images.shape[1:] != (self.image_channels, 64, 64):
            raise ValueError(
                f"images must have shape (batch, {self.image_channels}, 64, 64)"
            )
        statistics = self.statistics(self.encoder(images)).flatten(1)
        return statistics.chunk(2, dim=1)

    def reparameterize(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        """Draw differentiable samples from a diagonal Gaussian posterior."""
        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into images with values in ``[0, 1]``."""
        if latent.ndim != 2 or latent.shape[1] != self.latent_dimension:
            raise ValueError(
                f"latent must have shape (batch, {self.latent_dimension})"
            )
        features = self.latent_projection(latent)[..., None, None]
        return torch.sigmoid(self.decoder(features))

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct images and return their posterior parameters."""
        mean, log_variance = self.encode(images)
        reconstruction = self.decode(self.reparameterize(mean, log_variance))
        return reconstruction, mean, log_variance

    def loss(self, images: torch.Tensor) -> torch.Tensor:
        """Return reconstruction loss plus beta-weighted KL divergence."""
        reconstruction, mean, log_variance = self(images)
        reconstruction_loss = functional.mse_loss(reconstruction, images)
        kl_divergence = -0.5 * (1 + log_variance - mean.square() - log_variance.exp())
        return reconstruction_loss + self.beta * kl_divergence.mean()

    @torch.no_grad()
    def sample(self, count: int) -> torch.Tensor:
        """Generate ``count`` lens images from the standard normal prior."""
        if count < 1:
            raise ValueError("count must be positive")
        was_training = self.training
        self.eval()
        latent = torch.randn(
            count,
            self.latent_dimension,
            device=self.latent_projection.weight.device,
        )
        images = self.decode(latent)
        self.train(was_training)
        return images
