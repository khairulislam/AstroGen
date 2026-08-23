"""Embedding layers for generative models."""

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Map a batch of scalar diffusion times to sinusoidal embeddings."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 2:
            raise ValueError("dimension must be at least 2")
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Return an embedding with shape ``(batch, dimension)``."""
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape (batch,)")

        half_dimension = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000) * torch.arange(half_dimension, device=timesteps.device)
            / max(half_dimension - 1, 1)
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dimension % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding
