"""Image/signal-quality metrics shared by the denoising and super-resolution
evaluation examples."""

import math

import torch
from skimage.metrics import structural_similarity


def psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    """Peak signal-to-noise ratio, in dB, for tensors normalized to ``[-1, 1]``
    (``data_range=2``, the default)."""
    mse = torch.mean((prediction - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(data_range**2 / mse)


def ssim(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    """Structural similarity, averaged over the batch, for tensors shaped
    ``(batch, channels, height, width)`` and normalized to ``[-1, 1]``
    (``data_range=2``, the default). Paired with PSNR by the reference
    super-resolution repos (e.g. ``Super_Resolution_Pranath_Reddy``'s results
    table) and by ``examples/cgdpm_super_resolution.ipynb``."""
    prediction, target = prediction.detach().cpu().numpy(), target.detach().cpu().numpy()
    scores = [
        structural_similarity(p, t, data_range=data_range, channel_axis=0)
        for p, t in zip(prediction, target)
    ]
    return sum(scores) / len(scores)
