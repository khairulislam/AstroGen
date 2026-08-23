# AstroGen

AstroGen is a lightweight library of generative-model cores for astronomy data.

## Table of contents

- [Usage](#usage)
- [DDPM](#ddpm)
- [DeepLense VAE](#deeplense-vae)
- [Super-Resolution DDPM](#super-resolution-ddpm)
- [Denoising DDPM](#denoising-ddpm)
- [Resources](#resources)
- [Citations](#citations)

## Usage

AstroGen uses ordinary PyTorch modules. Import an implementation directly from
its documented package, provide the tensor contract described by that model,
and use it in a standard PyTorch training or inference loop.

```python
import torch
from diffusers import DDPMScheduler, UNet2DModel
from astrogen.models import DDPM

unet = UNet2DModel(
    in_channels=1, out_channels=1, layers_per_block=1,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    norm_num_groups=32,
)
model = DDPM(unet, DDPMScheduler(num_train_timesteps=1_000))
lens_images = torch.rand(8, 1, 64, 64) * 2 - 1
loss = model(lens_images)
samples = model.sample(count=4, sample_size=64)
```

## DDPM

`DDPM` composes a diffusers U-Net and `DDPMScheduler` the way a diffusers
pipeline does: you build the components with diffusers' own constructors
and pass them in — `DDPM` adds only the training-loss and sampling-loop
logic, no architecture beyond what diffusers already provides. Models whose
paper customizes the denoiser or noise process keep their own implementation
instead.

Whether a sample is a lens image (2D) or a spectrum (1D) follows the `unet`
you pass in, not a separate flag: give it a `UNet2DModel` for images with
shape `(batch, 1, height, width)`, or a `UNet1DModel` for spectra with shape
`(batch, 1, length)`. Both expect values normalized to approximately
`[-1, 1]`. Class conditioning (labels passed to `forward`/`sample`) is only
supported for a `UNet2DModel` built with `num_class_embeds` set, since
`UNet1DModel` has no class-embedding path.

```python
import torch
from diffusers import DDPMScheduler, UNet2DModel
from astrogen.models import DDPM

unet = UNet2DModel(
    in_channels=1, out_channels=1, layers_per_block=1,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    norm_num_groups=32,
)
model = DDPM(unet, DDPMScheduler(num_train_timesteps=1_000))
lens_images = torch.rand(8, 1, 64, 64) * 2 - 1
loss = model(lens_images)
samples = model.sample(count=4, sample_size=64)
```

Set `num_class_embeds` on the U-Net to condition on discrete labels (e.g.
`axion`/`CDM`/`no_sub` substructure type), passed as a `(batch,)` long
tensor of class indices:

```python
unet = UNet2DModel(..., num_class_embeds=3)
model = DDPM(unet, DDPMScheduler(num_train_timesteps=1_000))
labels = torch.randint(3, (8,))
loss = model(lens_images, labels=labels)
samples = model.sample(count=4, sample_size=64, labels=torch.tensor([0, 1, 2, 2]))
```

Set `condition_channels` on `DDPM` (and size the U-Net's `in_channels`
accordingly) to condition on an image of the same spatial size (e.g. a
low-resolution image for super-resolution — see
[Super-Resolution DDPM](#super-resolution-ddpm) below), concatenated
channel-wise and passed as `condition`:

```python
unet = UNet2DModel(in_channels=2, out_channels=1, ...)  # image + condition channels
model = DDPM(unet, DDPMScheduler(num_train_timesteps=1_000), condition_channels=1)
condition = torch.rand(8, 1, 64, 64) * 2 - 1
loss = model(lens_images, condition=condition)
samples = model.sample(count=4, sample_size=64, condition=condition[:4])
```

## DeepLense VAE

`DeepLenseVAE` is a compact convolutional variational autoencoder based on
DeepLense's lens-image VAE. Unlike the DDPM, it learns a Gaussian latent space
and generates images in a single decoder pass.

It accepts single-channel images with shape `(batch, 1, 64, 64)` and values in
`[0, 1]`.

```python
import torch
from astrogen.models import DeepLenseVAE

model = DeepLenseVAE(latent_dimension=128)
lens_images = torch.rand(8, 1, 64, 64)
loss = model.loss(lens_images)
samples = model.sample(count=4)
```

## Super-Resolution DDPM

`SuperResolutionDDPM` upscales a low-resolution lens image with a
channel-conditioned `DDPM`, following the SR3 recipe: the low-resolution
image is bilinearly upsampled to the target resolution and concatenated,
channel-wise, with the noisy high-resolution image at every denoising step.
It reuses `DDPM`'s `condition_channels` path rather than a dedicated
architecture, since this baseline needs none beyond that.

Low- and high-resolution images must have the same channel count and values
normalized to approximately `[-1, 1]`.

```python
import torch
from astrogen.tasks import SuperResolutionDDPM

model = SuperResolutionDDPM()
low_resolution = torch.rand(8, 1, 32, 32) * 2 - 1
high_resolution = torch.rand(8, 1, 64, 64) * 2 - 1
loss = model(low_resolution, high_resolution)
samples = model.sample(low_resolution, image_size=64)
```

## Denoising DDPM

`DenoisingDDPM` restores a corrupted image or spectrum with a
channel-conditioned `DDPM`, reusing the same conditioning path as
[Super-Resolution DDPM](#super-resolution-ddpm): the corrupted sample is
concatenated, channel-wise, with the noisy clean sample at every denoising
step. Unlike super-resolution, corrupted and clean samples share the same
spatial size, so no resizing is applied. Set `dimensionality="1d"` for
spectral denoising — the direct 1D counterpart of image denoising, following
[spectrai](https://github.com/conor-horgan/spectrai)'s `spectral_denoising`
task — instead of the default `dimensionality="2d"` for lens images.

Corrupted and clean samples must have the same channel count and values
normalized to approximately `[-1, 1]`. Corruption (e.g. Gaussian noise plus
blur, following family C's super-resolution recipe) is applied by the caller
before training.

```python
import torch
from astrogen.tasks import DenoisingDDPM

model = DenoisingDDPM()
clean = torch.rand(8, 1, 64, 64) * 2 - 1
corrupted = clean + 0.1 * torch.randn_like(clean)
loss = model(corrupted, clean)
samples = model.sample(corrupted)
```

For spectra:

```python
model = DenoisingDDPM(dimensionality="1d")
clean_spectrum = torch.rand(8, 1, 500) * 2 - 1
corrupted_spectrum = clean_spectrum + 0.1 * torch.randn_like(clean_spectrum)
loss = model(corrupted_spectrum, clean_spectrum)
samples = model.sample(corrupted_spectrum)
```

## Resources

* [DeepLense](https://github.com/ML4SCI/DeepLense/tree/main): Explores cutting-edge Machine Learning techniques for the study of Strong Gravitational Lensing and Dark Matter Sub-structure, using both simulated and real lensing images.
* [STAR](https://github.com/GuoCheng12/STAR/tree/main): The STAR (Super-Resolution for Astronomical Star Fields) dataset is a large-scale benchmark for developing field-level super-resolution models in astronomy.
* [Spectrai](https://github.com/conor-horgan/spectrai): an open-source deep learning framework designed to facilitate the training of neural networks on spectral data and enable comparison between different methods.
* [Hubble meets Webb](https://github.com/vkinakh/Hubble-meets-Webb): A study on the image-to-image translation problem for the prediction of future satellite Webb images from the available Hubble images.

## Citations

Please cite the corresponding generative method and source authors linked in the scripts when using these implementations.

```bibtex
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2006.11239},
  year={2020}
}
```

```bibtex
@article{kingma2013autoencoding,
  title={Auto-Encoding Variational Bayes},
  author={Kingma, Diederik P. and Welling, Max},
  journal={arXiv preprint arXiv:1312.6114},
  year={2013}
}
```

```bibtex
@article{saharia2022image,
  title={Image Super-Resolution via Iterative Refinement},
  author={Saharia, Chitwan and Ho, Jonathan and Chan, William and Salimans, Tim and Fleet, David J. and Norouzi, Mohammad},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022}
}
```
