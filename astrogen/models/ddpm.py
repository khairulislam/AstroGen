# Paper: https://arxiv.org/abs/2006.11239
# Reference: https://github.com/ML4SCI/DeepLense/blob/main/Difflense_Aleksandr_Duplinskii/Unconditional_diffusion/model_grav.py
"""Unconditional gravitational-lens image generation with a DDPM."""

import math

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch import nn
from torch.nn import functional


class DDPM(nn.Module):
    """An unconditional DDPM baseline for normalized lens images.

    Uses a vanilla diffusers U-Net and noise scheduler, since this baseline adds
    no architecture beyond what diffusers already provides. Images passed to
    :meth:`forward` must have shape ``(batch, image_channels, height, width)``
    and values normalized to approximately ``[-1, 1]``.
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
        norm_num_groups = math.gcd(base_channels, 32)
        unet_config = dict(
            in_channels=image_channels,
            out_channels=image_channels,
            layers_per_block=1,
            block_out_channels=(base_channels, base_channels * 2, base_channels * 4),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
            norm_num_groups=norm_num_groups,
        )
        unet_config.update(unet_kwargs or {})
        self.unet = UNet2DModel(**unet_config)

        scheduler_config = dict(num_train_timesteps=timesteps)
        scheduler_config.update(scheduler_kwargs or {})
        self.scheduler = DDPMScheduler(**scheduler_config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return the DDPM training loss for a batch of lens images."""
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            self.scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device
        )
        noised_images = self.scheduler.add_noise(images, noise, timesteps)
        predicted_noise = self.unet(noised_images, timesteps).sample
        return functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, count: int, image_size: int = 64) -> torch.Tensor:
        """Sample ``count`` generated single-channel lens images."""
        was_training = self.unet.training
        self.unet.eval()
        device = next(self.unet.parameters()).device
        images = torch.randn(
            count, self.unet.config.in_channels, image_size, image_size, device=device
        )
        for timestep in self.scheduler.timesteps:
            predicted_noise = self.unet(images, timestep).sample
            images = self.scheduler.step(predicted_noise, timestep, images).prev_sample
        self.unet.train(was_training)
        return images
