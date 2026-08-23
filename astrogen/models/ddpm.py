"""Denoising diffusion probabilistic model utilities."""

import torch
from torch import nn
from torch.nn import functional as functional


class GaussianDiffusion(nn.Module):
    """A Gaussian DDPM noise process with an epsilon-prediction objective."""

    def __init__(
        self,
        timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def noise_images(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to images at the given diffusion timesteps."""
        if images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        if timesteps.shape != (images.shape[0],):
            raise ValueError("timesteps must have shape (batch,)")
        if noise is None:
            noise = torch.randn_like(images)
        if noise.shape != images.shape:
            raise ValueError("noise must have the same shape as images")

        alpha_bars = self.alpha_bars[timesteps].view(-1, 1, 1, 1)
        noised_images = alpha_bars.sqrt() * images + (1 - alpha_bars).sqrt() * noise
        return noised_images, noise

    def loss(self, denoiser: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Return the epsilon-prediction loss for a batch of normalized images."""
        timesteps = torch.randint(self.timesteps, (images.shape[0],), device=images.device)
        noised_images, noise = self.noise_images(images, timesteps)
        return functional.mse_loss(denoiser(noised_images, timesteps), noise)

    @torch.no_grad()
    def sample(self, denoiser: nn.Module, shape: tuple[int, int, int, int]) -> torch.Tensor:
        """Generate images of ``shape`` by ancestral DDPM sampling."""
        if len(shape) != 4:
            raise ValueError("shape must be (batch, channels, height, width)")

        was_training = denoiser.training
        denoiser.eval()
        images = torch.randn(shape, device=self.betas.device)
        for step in range(self.timesteps - 1, -1, -1):
            timestep = torch.full((shape[0],), step, device=images.device, dtype=torch.long)
            predicted_noise = denoiser(images, timestep)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            mean = (images - (1 - alpha) / (1 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
            images = mean if step == 0 else mean + self.betas[step].sqrt() * torch.randn_like(images)
        denoiser.train(was_training)
        return images
