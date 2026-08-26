"""Image/signal-quality metrics shared by the denoising and super-resolution
evaluation examples."""

import math

import torch


def psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    """Peak signal-to-noise ratio, in dB, for tensors normalized to ``[-1, 1]``
    (``data_range=2``, the default)."""
    mse = torch.mean((prediction - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(data_range**2 / mse)
