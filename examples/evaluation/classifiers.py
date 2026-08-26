"""Tiny classifiers used only as evaluation scorers (never as generative
models), following PLAN.md's Evaluation template: train a small classifier
on real data, then score generated/restored samples with it."""

import torch
from torch import nn
from torch.nn import functional


class _ResidualBlock(nn.Module):
    """Strided residual block: halves spatial size, doubles channel capacity."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = functional.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return functional.relu(y + self.skip(x))


class TinyImageClassifier(nn.Module):
    """Scores class-conditional DDPM samples (A1) by predicted class.

    A small residual stack, rather than a plain shallow CNN: the localized,
    low-contrast substructure signal in real lensing images (unlike this
    example's earlier synthetic stand-in) needs the extra depth and batch
    normalization to be separable at all.
    """

    def __init__(self, image_channels: int = 1, num_classes: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        stage_channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 4)
        blocks, in_channels = [], base_channels
        for out_channels in stage_channels:
            blocks.append(_ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(images))
        return self.classifier(features.mean(dim=(-2, -1)))


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
