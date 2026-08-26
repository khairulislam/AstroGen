"""Tiny classifiers used only as evaluation scorers (never as generative
models), following PLAN.md's Evaluation template: train a small classifier
on real data, then score generated/restored samples with it."""

import torch
from torch import nn


class TinyImageClassifier(nn.Module):
    """Scores class-conditional DDPM samples (A1) by predicted class."""

    def __init__(self, image_channels: int = 1, num_classes: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(base_channels * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images).flatten(1))


class TinySpectralClassifier(nn.Module):
    """Scores denoised spectra (D1) by predicted class, standing in for D3
    (spectral classification), which PLAN.md marks eval-only rather than a
    core model."""

    def __init__(self, num_channels: int = 1, num_classes: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, base_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(base_channels * 2, num_classes)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(spectra).flatten(1))
