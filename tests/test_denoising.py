import torch
from diffusers import UNet1DModel

from astrogen.tasks import DenoisingDDPM


def test_denoising_ddpm_trains_and_samples() -> None:
    model = DenoisingDDPM(base_channels=8, timesteps=3)
    clean = torch.randn(2, 1, 16, 16).clamp(-1, 1)
    corrupted = clean + 0.1 * torch.randn_like(clean)

    loss = model(corrupted, clean)
    loss.backward()
    samples = model.sample(corrupted)

    assert loss.isfinite()
    assert samples.shape == clean.shape
    assert torch.isfinite(samples).all()


def test_spectral_denoising_ddpm_trains_and_samples() -> None:
    model = DenoisingDDPM(base_channels=8, timesteps=3, model_cls=UNet1DModel)
    clean = torch.randn(2, 1, 64).clamp(-1, 1)
    corrupted = clean + 0.1 * torch.randn_like(clean)

    loss = model(corrupted, clean)
    loss.backward()
    samples = model.sample(corrupted)

    assert loss.isfinite()
    assert samples.shape == clean.shape
    assert torch.isfinite(samples).all()
