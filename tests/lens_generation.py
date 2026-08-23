import torch

from astrogen.models import GaussianDiffusion
from astrogen.tasks import LensImageDDPM


def test_noise_images_follows_the_schedule() -> None:
    diffusion = GaussianDiffusion(timesteps=4)
    images = torch.ones(2, 1, 8, 8)
    timesteps = torch.tensor([0, 3])

    noised_images, noise = diffusion.noise_images(images, timesteps, torch.zeros_like(images))

    assert torch.equal(noise, torch.zeros_like(images))
    expected = diffusion.alpha_bars[timesteps].sqrt().view(2, 1, 1, 1)
    assert torch.allclose(noised_images, expected.expand_as(images))


def test_lens_ddpm_trains_and_samples() -> None:
    model = LensImageDDPM(base_channels=8, timesteps=3)
    images = torch.randn(2, 1, 16, 16).clamp(-1, 1)

    loss = model(images)
    loss.backward()
    samples = model.sample(2, image_size=16)

    assert loss.isfinite()
    assert samples.shape == images.shape
    assert torch.isfinite(samples).all()
