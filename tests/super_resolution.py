import torch

from astrogen.tasks import SuperResolutionDDPM


def test_super_resolution_ddpm_trains_and_samples() -> None:
    model = SuperResolutionDDPM(base_channels=8, timesteps=3)
    low_resolution = torch.randn(2, 1, 8, 8).clamp(-1, 1)
    high_resolution = torch.randn(2, 1, 16, 16).clamp(-1, 1)

    loss = model(low_resolution, high_resolution)
    loss.backward()
    samples = model.sample(low_resolution, image_size=16)

    assert loss.isfinite()
    assert samples.shape == high_resolution.shape
    assert torch.isfinite(samples).all()
