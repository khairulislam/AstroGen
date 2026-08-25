from unittest.mock import patch

import torch
from torch import nn

from astrogen.tasks import SuperResolutionDDPM


class _RecordingUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device_anchor = nn.Parameter(torch.empty(0))
        self.images: torch.Tensor | None = None
        self.condition: torch.Tensor | None = None

    def forward(
        self, images: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        self.images = images
        self.condition = condition
        return torch.zeros_like(images)


def test_super_resolution_ddpm_trains_and_samples() -> None:
    model = SuperResolutionDDPM(
        base_channels=8,
        channel_multipliers=(1, 2),
        timesteps=3,
    )
    low_resolution = torch.rand(2, 1, 8, 8) * 2 - 1
    high_resolution = torch.rand(2, 1, 16, 16) * 2 - 1

    loss = model(low_resolution, high_resolution)
    loss.backward()
    samples = model.sample(low_resolution, image_size=16)

    assert loss.isfinite()
    assert samples.shape == high_resolution.shape
    assert torch.isfinite(samples).all()


def test_super_resolution_ddpm_preserves_reference_condition_range() -> None:
    model = SuperResolutionDDPM(
        base_channels=8,
        channel_multipliers=(1, 2),
        timesteps=2,
    )
    unet = _RecordingUNet()
    model.ddpm.unet = unet
    low_resolution = torch.full((1, 1, 4, 4), -0.5)
    high_resolution = torch.full((1, 1, 8, 8), 0.5)

    model(low_resolution, high_resolution)

    assert unet.images is not None
    assert unet.condition is not None
    assert torch.allclose(unet.images.mean(), torch.tensor(0.5), atol=0.2)
    assert torch.allclose(unet.condition, torch.full_like(unet.condition, -0.5))

    with (
        patch(
            "torch.randn",
            return_value=torch.linspace(-1, 1, 64).reshape(1, 1, 8, 8),
        ),
        patch("torch.randn_like", side_effect=torch.zeros_like),
    ):
        sample = model.ddpm.sample(1, 8, torch.full((1, 1, 8, 8), -0.5))

    assert torch.isfinite(sample).all()
    assert torch.equal(unet.condition, torch.full_like(unet.condition, -0.5))
