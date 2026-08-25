# Reference: https://github.com/ML4SCI/DeepLense/blob/main/Super_Resolution_Atal_Gupta/models/cgdpm.ipynb
"""Conditional Gaussian diffusion model from the DeepLense super-resolution reference."""

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional


class SinusoidalPositionEmbedding(nn.Module):
    """Embed scalar diffusion timesteps with sinusoidal features."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dimension = self.dimension // 2
        scale = math.log(10_000) / (half_dimension - 1)
        frequencies = torch.exp(
            torch.arange(half_dimension, device=timesteps.device) * -scale
        )
        embeddings = timesteps[:, None] * frequencies[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class Upsample(nn.Module):
    """Double spatial resolution with a transposed convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convolution = nn.ConvTranspose2d(channels, channels, 4, 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.convolution(inputs)


class Downsample(nn.Module):
    """Halve spatial resolution with a strided convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(channels, channels, 3, 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.convolution(inputs)


class ConvolutionBlock(nn.Module):
    """Convolution, group normalization, and Mish activation."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConditionedResidualBlock(nn.Module):
    """Reference residual block with resized image conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dimension: int,
        condition_channels: int,
        groups: int = 8,
    ) -> None:
        super().__init__()
        conditioned_channels = in_channels + condition_channels
        self.time_projection = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_embedding_dimension, out_channels),
        )
        self.block1 = ConvolutionBlock(conditioned_channels, out_channels, groups)
        self.block2 = ConvolutionBlock(out_channels, out_channels, groups)
        self.residual = (
            nn.Conv2d(conditioned_channels, out_channels, 1)
            if conditioned_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, inputs: torch.Tensor, time_embedding: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        condition = functional.interpolate(
            condition, size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )
        conditioned_inputs = torch.cat((inputs, condition), dim=1)
        hidden_states = self.block1(conditioned_inputs)
        hidden_states = hidden_states + self.time_projection(time_embedding)[:, :, None, None]
        hidden_states = self.block2(hidden_states)
        return hidden_states + self.residual(conditioned_inputs)


class LinearAttention(nn.Module):
    """Linear spatial attention used at every reference U-Net resolution."""

    def __init__(self, channels: int, heads: int = 4, head_channels: int = 32) -> None:
        super().__init__()
        self.heads = heads
        self.head_channels = head_channels
        hidden_channels = heads * head_channels
        self.to_queries_keys_values = nn.Conv2d(channels, hidden_channels * 3, 1, bias=False)
        self.to_output = nn.Conv2d(hidden_channels, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = inputs.shape
        queries_keys_values = self.to_queries_keys_values(inputs).reshape(
            batch, 3, self.heads, self.head_channels, height * width
        )
        queries, keys, values = queries_keys_values.unbind(dim=1)
        keys = keys.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", keys, values)
        outputs = torch.einsum("bhde,bhdn->bhen", context, queries)
        outputs = outputs.reshape(batch, self.heads * self.head_channels, height, width)
        return self.to_output(outputs)


class RezeroResidual(nn.Module):
    """Residual branch with a learned, zero-initialized scale."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.module(inputs) * self.scale


class CGDPMUNet(nn.Module):
    """Conditioned U-Net used by the DeepLense CGDPM reference."""

    def __init__(
        self,
        image_channels: int = 1,
        condition_channels: int = 1,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        groups: int = 8,
    ) -> None:
        super().__init__()
        dimensions = [image_channels, *(base_channels * multiplier for multiplier in channel_multipliers)]
        in_out = list(zip(dimensions[:-1], dimensions[1:]))
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, base_channels * 4),
            nn.Mish(),
            nn.Linear(base_channels * 4, base_channels),
        )

        self.down_blocks = nn.ModuleList()
        for index, (in_channels, out_channels) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ConditionedResidualBlock(
                            in_channels,
                            out_channels,
                            base_channels,
                            condition_channels,
                            groups,
                        ),
                        ConditionedResidualBlock(
                            out_channels,
                            out_channels,
                            base_channels,
                            condition_channels,
                            groups,
                        ),
                        RezeroResidual(LinearAttention(out_channels)),
                        nn.Identity() if is_last else Downsample(out_channels),
                    ]
                )
            )

        middle_channels = dimensions[-1]
        self.middle_block1 = ConditionedResidualBlock(
            middle_channels,
            middle_channels,
            base_channels,
            condition_channels,
            groups,
        )
        self.middle_attention = RezeroResidual(LinearAttention(middle_channels))
        self.middle_block2 = ConditionedResidualBlock(
            middle_channels,
            middle_channels,
            base_channels,
            condition_channels,
            groups,
        )

        self.up_blocks = nn.ModuleList()
        for in_channels, out_channels in reversed(in_out[1:]):
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ConditionedResidualBlock(
                            out_channels * 2,
                            in_channels,
                            base_channels,
                            condition_channels,
                            groups,
                        ),
                        ConditionedResidualBlock(
                            in_channels,
                            in_channels,
                            base_channels,
                            condition_channels,
                            groups,
                        ),
                        RezeroResidual(LinearAttention(in_channels)),
                        Upsample(in_channels),
                    ]
                )
            )

        self.final = nn.Sequential(
            ConvolutionBlock(base_channels, base_channels, groups),
            nn.Conv2d(base_channels, image_channels, 1),
        )

    def forward(
        self, images: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)
        skip_connections = []
        for residual1, residual2, attention, downsample in self.down_blocks:
            images = residual1(images, time_embedding, condition)
            images = residual2(images, time_embedding, condition)
            images = attention(images)
            skip_connections.append(images)
            images = downsample(images)

        images = self.middle_block1(images, time_embedding, condition)
        images = self.middle_attention(images)
        images = self.middle_block2(images, time_embedding, condition)

        for residual1, residual2, attention, upsample in self.up_blocks:
            images = torch.cat((images, skip_connections.pop()), dim=1)
            images = residual1(images, time_embedding, condition)
            images = residual2(images, time_embedding, condition)
            images = attention(images)
            images = upsample(images)
        return self.final(images)


class CGDPM(nn.Module):
    """Reference conditional diffusion process and U-Net.

    Training images and conditions are expected in ``[-1, 1]``, the model
    space diffusers uses throughout (``VaeImageProcessor.normalize``,
    ``DDPMScheduler(clip_sample_range=1.0)``); :meth:`sample` returns that
    same space, leaving conversion to a display range to the caller. The
    model predicts Gaussian noise with an L1 objective.

    Two deliberate departures from the DeepLense source, which works in
    ``[0, 1]``: that source ends sampling with ``clamp(-1, 1)``, a shift to
    ``[0, 1]``, and a batch-wide min-max rescale, none of which is done here;
    and ``clip_sample``/``clip_sample_range`` are added, taking their names,
    defaults, and semantics from ``DDPMScheduler`` — clamping predicted
    ``x_0`` each reverse step. With ``clip_sample=False`` the reverse step is
    the source's equation unchanged.
    """

    def __init__(
        self,
        image_channels: int = 1,
        condition_channels: int = 1,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        groups: int = 8,
        timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError(f"timesteps must be at least 2, got {timesteps}")
        self.timesteps = timesteps
        self.image_channels = image_channels
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        if prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError(f"unsupported prediction_type: {prediction_type}")
        self.prediction_type = prediction_type
        self.unet = CGDPMUNet(
            image_channels,
            condition_channels,
            base_channels,
            channel_multipliers,
            groups,
        )
        schedule_timesteps = torch.linspace(0, 1, timesteps)
        offset = 0.008
        betas = (1 - torch.cos((schedule_timesteps + offset) / (1 + offset) * math.pi)) / 2
        betas = betas * (beta_end - beta_start) + beta_start
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Return the reference L1 noise-prediction loss."""
        timesteps = torch.randint(1, self.timesteps, (images.shape[0],), device=images.device)
        noise = torch.randn_like(images)
        alpha = self.alphas_cumprod[timesteps].reshape(
            images.shape[0], *((1,) * (images.ndim - 1))
        )
        noised_images = alpha.sqrt() * images + (1 - alpha).sqrt() * noise
        prediction = self.unet(noised_images, timesteps, condition)
        target = (
            noise
            if self.prediction_type == "epsilon"
            else alpha.sqrt() * noise - (1 - alpha).sqrt() * images
        )
        return functional.l1_loss(prediction, target)

    @torch.no_grad()
    def sample(self, count: int, sample_size: int, condition: torch.Tensor) -> torch.Tensor:
        """Generate conditioned images with the reference reverse process."""
        was_training = self.unet.training
        self.unet.eval()
        device = self.betas.device
        condition = condition.to(device)
        images = torch.randn(
            count, self.image_channels, sample_size, sample_size, device=device
        )
        for index in reversed(range(1, self.timesteps)):
            timesteps = torch.full((count,), index, device=device, dtype=torch.long)
            prediction = self.unet(images, timesteps, condition)
            alpha = self.alphas[timesteps].reshape(count, 1, 1, 1)
            alpha_cumprod = self.alphas_cumprod[timesteps].reshape(count, 1, 1, 1)
            predicted_noise = (
                prediction
                if self.prediction_type == "epsilon"
                else alpha_cumprod.sqrt() * prediction
                + (1 - alpha_cumprod).sqrt() * images
            )
            if self.clip_sample:
                # DDPMScheduler.step clamps predicted x_0 (scheduling_ddpm.py:527);
                # re-deriving the noise from the clamped x_0 applies it while
                # leaving the source's reverse equation below untouched.
                predicted_original = (
                    images - (1 - alpha_cumprod).sqrt() * predicted_noise
                ) / alpha_cumprod.sqrt()
                predicted_original = predicted_original.clamp(
                    -self.clip_sample_range, self.clip_sample_range
                )
                predicted_noise = (
                    images - alpha_cumprod.sqrt() * predicted_original
                ) / (1 - alpha_cumprod).sqrt()
            beta = self.betas[timesteps].reshape(count, 1, 1, 1)
            noise = torch.randn_like(images) if index > 1 else torch.zeros_like(images)
            images = (
                (images - (1 - alpha) / (1 - alpha_cumprod).sqrt() * predicted_noise)
                / alpha.sqrt()
                + beta.sqrt() * noise
            )
        self.unet.train(was_training)
        return images
