import torch
from diffusers import DDPMScheduler, UNet1DModel, UNet2DModel

from astrogen.layers import ParameterEncoder
from astrogen.models import DDPM, DeepLenseVAE


def test_lens_ddpm_trains_and_samples() -> None:
    unet = UNet2DModel(
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(8, 16, 32),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )
    model = DDPM(unet, DDPMScheduler(num_train_timesteps=3))
    images = torch.randn(2, 1, 16, 16).clamp(-1, 1)

    loss = model(images)
    loss.backward()
    samples = model.sample(2, sample_size=16)

    assert loss.isfinite()
    assert samples.shape == images.shape
    assert torch.isfinite(samples).all()


def test_lens_ddpm_class_conditional_trains_and_samples() -> None:
    unet = UNet2DModel(
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(8, 16, 32),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
        num_class_embeds=3,
    )
    model = DDPM(unet, DDPMScheduler(num_train_timesteps=3))
    images = torch.randn(2, 1, 16, 16).clamp(-1, 1)
    labels = torch.tensor([0, 2])

    loss = model(images, labels=labels)
    loss.backward()
    samples = model.sample(2, sample_size=16, labels=labels)

    assert loss.isfinite()
    assert samples.shape == images.shape
    assert torch.isfinite(samples).all()


def test_lens_ddpm_variable_conditional_trains_and_samples() -> None:
    base_channels = 8
    unet = UNet2DModel(
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(base_channels, 16, 32),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
        class_embed_type="identity",
    )
    parameter_encoder = ParameterEncoder(num_parameters=3, embedding_dim=base_channels * 4)
    model = DDPM(unet, DDPMScheduler(num_train_timesteps=3), parameter_encoder=parameter_encoder)
    images = torch.randn(2, 1, 16, 16).clamp(-1, 1)
    parameters = torch.randn(2, 3)  # e.g. mass, orientation, redshift

    loss = model(images, labels=parameters)
    loss.backward()
    samples = model.sample(2, sample_size=16, labels=parameters)

    assert loss.isfinite()
    assert samples.shape == images.shape
    assert torch.isfinite(samples).all()


def test_spectra_ddpm_trains_and_samples() -> None:
    unet = UNet1DModel(
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(8, 16, 32),
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D"),
        norm_num_groups=8,
    )
    model = DDPM(unet, DDPMScheduler(num_train_timesteps=3))
    spectra = torch.randn(2, 1, 64).clamp(-1, 1)

    loss = model(spectra)
    loss.backward()
    samples = model.sample(2, sample_size=64)

    assert loss.isfinite()
    assert samples.shape == spectra.shape
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
