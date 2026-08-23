"""Task-specific meta-architectures."""

from .denoising import DenoisingDDPM
from .super_resolution import SuperResolutionDDPM

__all__ = [
    "DenoisingDDPM",
    "SuperResolutionDDPM",
]
