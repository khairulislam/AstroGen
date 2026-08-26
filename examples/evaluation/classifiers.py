"""Classifiers used only as evaluation scorers (never as generative models),
following PLAN.md's Evaluation template: train a classifier on real data,
then score generated/restored samples with it."""

import torch
from torch import nn
from torch.nn import functional


class _BasicBlock(nn.Module):
    """ResNet-18's basic block: two 3x3 convs plus a projected skip on stride/width change."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = functional.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return functional.relu(y + self.skip(x))


class ImageClassifier(nn.Module):
    """Scores class-conditional DDPM samples (A1) by predicted class.

    ResNet-18's own body (4 stages of 2 basic blocks, channels doubling
    64-128-256-512), matching the depth of DeepLense's own reference
    evaluation classifier (``DeepLense_Diffusion_Rishi/scripts/run_resnet.py``,
    ``torchvision.models.resnet18``). The stem differs: that reference's 7x7
    stride-2 conv plus stride-2 maxpool is calibrated for 224px ImageNet
    input, and at this dataset's much smaller resolution discards the same
    low-contrast substructure detail the classifier needs to separate,
    before a single residual block sees it (measured: 61% best accuracy,
    repeatedly collapsing back to chance). A plain stride-1 stem — the usual
    fix for ResNets on small images (as in CIFAR-ResNet) — reaches 82%.
    """

    def __init__(self, image_channels: int = 1, num_classes: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        stage_channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        blocks, in_channels = [], base_channels
        for out_channels in stage_channels:
            blocks += [_BasicBlock(in_channels, out_channels, stride=2), _BasicBlock(out_channels, out_channels, stride=1)]
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_channels, num_classes))

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
