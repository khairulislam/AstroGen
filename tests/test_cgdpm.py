from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional

from astrogen.models import CGDPM


class _RecordingZeroPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self, images: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((images.clone(), timesteps.clone(), condition.clone()))
        return torch.zeros_like(images)


def test_cgdpm_forward_matches_reference_equation() -> None:
    diffusion = CGDPM(base_channels=8, channel_multipliers=(1,), timesteps=4)
    predictor = _RecordingZeroPredictor()
    diffusion.unet = predictor
    images = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]])
    condition = torch.full_like(images, 0.3)
    noise = torch.tensor([[[[-1.0, -0.5], [0.5, 1.0]]]])
    timesteps = torch.tensor([2])

    with (
        patch("torch.randint", return_value=timesteps),
        patch("torch.randn_like", return_value=noise),
    ):
        loss = diffusion(images, condition)

    alpha_cumprod = diffusion.alphas_cumprod[timesteps].reshape(1, 1, 1, 1)
    expected_images = (
        alpha_cumprod.sqrt() * images + (1 - alpha_cumprod).sqrt() * noise
    )
    actual_images, actual_timesteps, actual_condition = predictor.calls[0]
    assert torch.allclose(actual_images, expected_images)
    assert torch.equal(actual_timesteps, timesteps)
    assert torch.equal(actual_condition, condition)
    assert torch.equal(loss, functional.l1_loss(torch.zeros_like(noise), noise))


def test_cgdpm_sampling_matches_reference_equation() -> None:
    # clip_sample=False is the DeepLense source's reverse equation unchanged.
    diffusion = CGDPM(
        base_channels=8, channel_multipliers=(1,), timesteps=4, clip_sample=False
    )
    predictor = _RecordingZeroPredictor()
    diffusion.unet = predictor
    condition = torch.full((1, 1, 2, 2), 0.3)
    initial = torch.tensor([[[[-0.8, -0.2], [0.4, 0.9]]]])
    injected_noise = [
        torch.full_like(initial, 0.1),
        torch.full_like(initial, -0.2),
    ]

    with (
        patch("torch.randn", return_value=initial),
        patch("torch.randn_like", side_effect=injected_noise),
    ):
        sample = diffusion.sample(1, sample_size=2, condition=condition)

    expected = initial
    for index, noise in zip((3, 2, 1), (*injected_noise, torch.zeros_like(initial))):
        expected = (
            expected / diffusion.alphas[index].sqrt()
            + diffusion.betas[index].sqrt() * noise
        )

    assert [call[1].item() for call in predictor.calls] == [3, 2, 1]
    assert all(torch.equal(call[2], condition) for call in predictor.calls)
    assert torch.allclose(sample, expected)


def test_cgdpm_v_prediction_target() -> None:
    diffusion = CGDPM(
        base_channels=8,
        channel_multipliers=(1,),
        timesteps=4,
        prediction_type="v_prediction",
    )
    predictor = _RecordingZeroPredictor()
    diffusion.unet = predictor
    images = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]])
    condition = torch.zeros_like(images)
    noise = torch.tensor([[[[-1.0, -0.5], [0.5, 1.0]]]])
    timesteps = torch.tensor([2])

    with (
        patch("torch.randint", return_value=timesteps),
        patch("torch.randn_like", return_value=noise),
    ):
        loss = diffusion(images, condition)

    alpha_cumprod = diffusion.alphas_cumprod[timesteps].reshape(1, 1, 1, 1)
    velocity = alpha_cumprod.sqrt() * noise - (1 - alpha_cumprod).sqrt() * images
    assert torch.equal(loss, functional.l1_loss(torch.zeros_like(velocity), velocity))


def test_cgdpm_clip_sample_bounds_predicted_original() -> None:
    """clip_sample clamps predicted x_0, matching DDPMScheduler's default."""
    clipped = CGDPM(base_channels=8, channel_multipliers=(1,), timesteps=4)
    assert clipped.clip_sample and clipped.clip_sample_range == 1.0
    unclipped = CGDPM(
        base_channels=8, channel_multipliers=(1,), timesteps=4, clip_sample=False
    )
    condition = torch.zeros(1, 1, 2, 2)
    initial = torch.full((1, 1, 2, 2), 3.0)
    samples = []
    for diffusion in (clipped, unclipped):
        diffusion.unet = _RecordingZeroPredictor()
        with (
            patch("torch.randn", return_value=initial),
            patch("torch.randn_like", side_effect=lambda x: torch.zeros_like(x)),
        ):
            samples.append(diffusion.sample(1, sample_size=2, condition=condition))
    # A zero-noise predictor makes predicted x_0 = x_t / sqrt(alpha_cumprod),
    # far outside [-1, 1] here, so clipping must pull the sample in.
    assert samples[0].abs().max() < samples[1].abs().max()
