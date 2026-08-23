# AstroGen

AstroGen is a lightweight library of generative-model cores for astronomy data.

## Table of contents

- [Usage](#usage)
- [DDPM](#ddpm)
- [DeepLense VAE](#deeplense-vae)
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

`DDPM` is a compact unconditional 2D denoising diffusion model for
gravitational-lens images. It implements the [DDPM method](https://arxiv.org/abs/2006.11239)
using [diffusers](https://github.com/huggingface/diffusers)' `UNet2DModel` and
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
