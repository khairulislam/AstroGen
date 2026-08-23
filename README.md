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
from astrogen.models import DDPM

model = DDPM()
lens_images = torch.rand(8, 1, 64, 64) * 2 - 1
loss = model(lens_images)
samples = model.sample(count=4, image_size=64)
```

## DDPM

`DDPM` is a compact 2D denoising diffusion model for gravitational-lens
images, unconditional or class-conditional. It implements the
[DDPM method](https://arxiv.org/abs/2006.11239) using
[diffusers](https://github.com/huggingface/diffusers)' `UNet2DModel` and
`DDPMScheduler` directly, since this baseline needs no architecture beyond
what diffusers already provides. Models whose paper customizes the denoiser
or noise process keep their own implementation instead.

It accepts normalized, single-channel images with shape
`(batch, 1, height, width)` and values approximately in `[-1, 1]`.

```python
import torch
from astrogen.models import DDPM

model = DDPM()
lens_images = torch.rand(8, 1, 64, 64) * 2 - 1
loss = model(lens_images)
samples = model.sample(count=4, image_size=64)
```

Set `num_classes` to condition on discrete labels (e.g. `axion`/`CDM`/`no_sub`
substructure type), passed as a `(batch,)` long tensor of class indices:

```python
model = DDPM(num_classes=3)
labels = torch.randint(3, (8,))
loss = model(lens_images, labels=labels)
samples = model.sample(count=4, image_size=64, labels=torch.tensor([0, 1, 2, 2]))
```

Set `condition_channels` to condition on an image of the same spatial size
(e.g. a low-resolution image for super-resolution — see
[Super-Resolution DDPM](#super-resolution-ddpm) below), concatenated
channel-wise and passed as `condition`:

```python
model = DDPM(condition_channels=1)
condition = torch.rand(8, 1, 64, 64) * 2 - 1
loss = model(lens_images, condition=condition)
samples = model.sample(count=4, image_size=64, condition=condition[:4])
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

`DenoisingDDPM` restores a corrupted lens image with a channel-conditioned
`DDPM`, reusing the same conditioning path as
[Super-Resolution DDPM](#super-resolution-ddpm): the corrupted image is
concatenated, channel-wise, with the noisy clean image at every denoising
step. Unlike super-resolution, corrupted and clean images share the same
spatial size, so no resizing is applied.

Corrupted and clean images must have the same channel count and values
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
