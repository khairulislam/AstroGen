# Paper: https://arxiv.org/abs/2006.11239
# Reference: https://github.com/ML4SCI/DeepLense/blob/main/Difflense_Aleksandr_Duplinskii/Unconditional_diffusion/model_grav.py
"""Unconditional gravitational-lens image generation with a DDPM."""

import math

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch import nn
from torch.nn import functional


class DDPM(nn.Module):
    """A DDPM baseline for normalized lens images, optionally conditional.

    Uses a vanilla diffusers U-Net and noise scheduler, since this baseline adds
    no architecture beyond what diffusers already provides. Images passed to
    :meth:`forward` must have shape ``(batch, image_channels, height, width)``
    and values normalized to approximately ``[-1, 1]``.

    Set ``num_classes`` to condition on discrete labels (e.g. substructure type),
    passed as a ``(batch,)`` long tensor of class indices to :meth:`forward` and
    :meth:`sample`. Set ``condition_channels`` to condition on an image of the
    same spatial size (e.g. an upsampled low-resolution image for
    super-resolution), concatenated channel-wise, passed as ``condition`` to
    :meth:`forward` and :meth:`sample`. Leave both ``None``/``0`` for
    unconditional generation.
    """

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 32,
        timesteps: int = 1_000,
        num_classes: int | None = None,
        condition_channels: int = 0,
        unet_kwargs: dict | None = None,
        scheduler_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.condition_channels = condition_channels
        norm_num_groups = math.gcd(base_channels, 32)
        unet_config = dict(
            in_channels=image_channels + condition_channels,
            out_channels=image_channels,
            layers_per_block=1,
            block_out_channels=(base_channels, base_channels * 2, base_channels * 4),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
            norm_num_groups=norm_num_groups,
            num_class_embeds=num_classes,
        )
        unet_config.update(unet_kwargs or {})
        self.unet = UNet2DModel(**unet_config)

        scheduler_config = dict(num_train_timesteps=timesteps)
        scheduler_config.update(scheduler_kwargs or {})
        self.scheduler = DDPMScheduler(**scheduler_config)

    def _model_input(self, images: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if self.condition_channels == 0:
            return images
        return torch.cat([images, condition], dim=1)

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the DDPM training loss for a batch of lens images."""
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            self.scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device
        )
        noised_images = self.scheduler.add_noise(images, noise, timesteps)
        model_input = self._model_input(noised_images, condition)
        predicted_noise = self.unet(model_input, timesteps, class_labels=labels).sample
        return functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        count: int,
        image_size: int = 64,
        labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``count`` generated single-channel lens images."""
        was_training = self.unet.training
        self.unet.eval()
        device = next(self.unet.parameters()).device
        images = torch.randn(count, self.image_channels, image_size, image_size, device=device)
        if labels is not None:
            labels = labels.to(device)
        if condition is not None:
            condition = condition.to(device)
        for timestep in self.scheduler.timesteps:
            model_input = self._model_input(images, condition)
            predicted_noise = self.unet(model_input, timestep, class_labels=labels).sample
            images = self.scheduler.step(predicted_noise, timestep, images).prev_sample
        self.unet.train(was_training)
        return images
