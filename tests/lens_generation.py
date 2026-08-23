import torch

from astrogen.models import DeepLenseVAE, DDPM


def test_lens_ddpm_trains_and_samples() -> None:
    model = DDPM(base_channels=8, timesteps=3)
    images = torch.randn(2, 1, 16, 16).clamp(-1, 1)

    loss = model(images)
    loss.backward()
    samples = model.sample(2, image_size=16)

    assert loss.isfinite()
    assert samples.shape == images.shape
    assert torch.isfinite(samples).all()


def test_deeplense_vae_trains_and_samples() -> None:
    model = DeepLenseVAE(latent_dimension=8, base_channels=2)
    images = torch.rand(2, 1, 64, 64)

    loss = model.loss(images)
    loss.backward()
    samples = model.sample(2)

    assert loss.isfinite()
    assert samples.shape == images.shape
    assert torch.isfinite(samples).all()
    assert samples.min() >= 0
    assert samples.max() <= 1
