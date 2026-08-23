# Paper: https://arxiv.org/abs/2006.11239
# Reference: https://github.com/ML4SCI/DeepLense/blob/main/Difflense_Aleksandr_Duplinskii/Unconditional_diffusion/model_grav.py
"""Denoising diffusion for lens images (2D) and spectra (1D)."""

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch import nn
from torch.nn import functional


class DDPM(nn.Module):
    """A DDPM baseline wrapping a diffusers U-Net and scheduler, optionally conditional.

    Composes a diffusers U-Net (``UNet2DModel`` for lens images,
    ``UNet1DModel`` for spectra) and ``DDPMScheduler`` the way a diffusers
    pipeline does: build the components with diffusers' own constructors and
    pass them in. This baseline adds no architecture beyond what diffusers
    already provides. Whether a sample is an image or a spectrum follows
    ``unet``'s own type, not a separate flag.

    Samples passed to :meth:`forward` must have shape ``(batch,
    image_channels, height, width)`` for a 2D U-Net or ``(batch,
    image_channels, length)`` for a 1D U-Net, with values normalized to
    approximately ``[-1, 1]``.

    Class conditioning (labels passed to :meth:`forward`/:meth:`sample`) is
    only supported for a 2D U-Net built with ``num_class_embeds`` set, since
    ``UNet1DModel`` has no class-embedding path. Set ``condition_channels``
    to condition on a sample of the same spatial size (e.g. an upsampled
    low-resolution image for super-resolution, or a corrupted spectrum for
    denoising) — matched by ``unet``'s ``in_channels`` — concatenated
    channel-wise and passed as ``condition`` to :meth:`forward` and
    :meth:`sample`. Leave both ``None``/``0`` for unconditional generation.
    """

    def __init__(
        self,
        unet: nn.Module,
        scheduler: DDPMScheduler,
        image_channels: int = 1,
        condition_channels: int = 0,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.image_channels = image_channels
        self.condition_channels = condition_channels

    def _model_input(self, images: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if self.condition_channels == 0:
            return images
        return torch.cat([images, condition], dim=1)

    def _predict_noise(
        self, model_input: torch.Tensor, timesteps: torch.Tensor | int, labels: torch.Tensor | None
    ) -> torch.Tensor:
        if isinstance(self.unet, UNet2DModel):
            return self.unet(model_input, timesteps, class_labels=labels).sample
        return self.unet(model_input, timesteps).sample

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the DDPM training loss for a batch of lens images or spectra."""
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            self.scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device
        )
        noised_images = self.scheduler.add_noise(images, noise, timesteps)
        model_input = self._model_input(noised_images, condition)
        predicted_noise = self._predict_noise(model_input, timesteps, labels)
        return functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        count: int,
        sample_size: int = 64,
        labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``count`` generated lens images (2D U-Net) or spectra (1D U-Net).

        ``sample_size`` is the image side length for a 2D U-Net or the
        spectrum length for a 1D U-Net.
        """
        was_training = self.unet.training
        self.unet.eval()
        device = next(self.unet.parameters()).device
        shape = (
            (count, self.image_channels, sample_size, sample_size)
            if isinstance(self.unet, UNet2DModel)
            else (count, self.image_channels, sample_size)
        )
        images = torch.randn(*shape, device=device)
        if labels is not None:
            labels = labels.to(device)
        if condition is not None:
            condition = condition.to(device)
        for timestep in self.scheduler.timesteps:
            model_input = self._model_input(images, condition)
            predicted_noise = self._predict_noise(model_input, timestep, labels)
            images = self.scheduler.step(predicted_noise, timestep, images).prev_sample
        self.unet.train(was_training)
        return images
