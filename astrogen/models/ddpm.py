# Paper: https://arxiv.org/abs/2006.11239
# Reference: https://github.com/ML4SCI/DeepLense/blob/main/Difflense_Aleksandr_Duplinskii/Unconditional_diffusion/model_grav.py
"""Denoising diffusion for lens images (2D) and spectra (1D)."""

import math

import torch
from diffusers import DDPMScheduler, UNet1DModel, UNet2DModel
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

    Samples passed to ``forward`` must have shape ``(batch,
    image_channels, height, width)`` for a 2D U-Net or ``(batch,
    image_channels, length)`` for a 1D U-Net, with values normalized to
    approximately ``[-1, 1]``. The training target follows ``scheduler``'s
    configured prediction type: noise, clean sample, or velocity.

    Class conditioning (labels passed to ``forward``/``sample``) is
    only supported for a 2D U-Net built with ``num_class_embeds`` set, since
    ``UNet1DModel`` has no class-embedding path; passing labels otherwise
    raises rather than silently ignoring them. Set ``condition_channels``
    to condition on a sample of the same spatial size (e.g. an upsampled
    low-resolution image for super-resolution, or a corrupted spectrum for
    denoising), matched by ``unet``'s ``in_channels`` and concatenated
    channel-wise, then passed as ``condition`` to ``forward`` and
    ``sample``. Leave both ``None``/``0`` for unconditional generation.

    Continuous physical-parameter conditioning (e.g. lens mass, orientation,
    redshift) reuses the same ``labels`` argument: build the U-Net with
    ``class_embed_type="identity"`` and pass a ``parameter_encoder`` (e.g.
    :class:`astrogen.layers.ParameterEncoder`) whose output dimension matches
    the U-Net's ``time_embed_dim`` (``block_out_channels[0] * 4`` by default).
    ``labels`` is then the raw parameter tensor (shape ``(batch,
    num_parameters)``); it is passed through ``parameter_encoder`` before
    reaching the U-Net's class-embedding path.
    """

    def __init__(
        self,
        unet: nn.Module,
        scheduler: DDPMScheduler,
        image_channels: int = 1,
        condition_channels: int = 0,
        parameter_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.image_channels = image_channels
        self.condition_channels = condition_channels
        self.parameter_encoder = parameter_encoder

    def _model_input(self, images: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if self.condition_channels == 0:
            return images
        return torch.cat([images, condition], dim=1)

    def _predict(
        self, model_input: torch.Tensor, timesteps: torch.Tensor | int, labels: torch.Tensor | None
    ) -> torch.Tensor:
        # Mirrors UNet2DModel.forward's own class_embedding-vs-class_labels check
        # (unet_2d.py) rather than a separate flag: class_embedding only exists,
        # and is non-None, on a UNet2DModel built with num_class_embeds set.
        if getattr(self.unet, "class_embedding", None) is not None:
            if self.parameter_encoder is not None and labels is not None:
                labels = self.parameter_encoder(labels)
            return self.unet(model_input, timesteps, class_labels=labels).sample
        if labels is not None:
            raise ValueError(
                "labels were provided but unet has no class-embedding path; "
                "build it with num_class_embeds set to use class conditioning"
            )
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
        prediction = self._predict(model_input, timesteps, labels)
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "sample":
            alphas_cumprod = self.scheduler.alphas_cumprod.to(
                device=images.device, dtype=images.dtype
            )
            alpha = alphas_cumprod[timesteps].reshape(
                images.shape[0], *((1,) * (images.ndim - 1))
            )
            weights = alpha / (1 - alpha)
            return (weights * functional.mse_loss(prediction, images, reduction="none")).mean()
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(images, noise, timesteps)
        else:
            raise ValueError(
                f"unsupported scheduler prediction_type: {self.scheduler.config.prediction_type!r}"
            )
        return functional.mse_loss(prediction, target)

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
            prediction = self._predict(model_input, timestep, labels)
            images = self.scheduler.step(prediction, timestep, images).prev_sample
        self.unet.train(was_training)
        return images


def build_conditional_ddpm(
    image_channels: int,
    base_channels: int,
    timesteps: int,
    unet_kwargs: dict | None,
    scheduler_kwargs: dict | None,
    model_cls: type[UNet1DModel] | type[UNet2DModel] = UNet2DModel,
) -> DDPM:
    """Build a channel-conditioned ``DDPM`` sharing family C's interface.

    Used by ``DenoisingDDPM``. Pass
    ``model_cls=UNet1DModel`` for spectra; follows ``DDPM``'s convention
    of keying behavior off the U-Net's own type rather than a separate
    dimensionality flag.
    """
    if model_cls not in (UNet1DModel, UNet2DModel):
        raise ValueError(f"model_cls must be UNet1DModel or UNet2DModel, got {model_cls!r}")
    is_1d = model_cls is UNet1DModel
    block = "DownBlock1D" if is_1d else "DownBlock2D"
    up_block = "UpBlock1D" if is_1d else "UpBlock2D"
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
    return DDPM(
        model_cls(**unet_config),
        DDPMScheduler(**scheduler_config),
        image_channels=image_channels,
        condition_channels=image_channels,
    )
