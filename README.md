# AstroGen

AstroGen is a lightweight library of generative-model cores for astronomy data.

## Table of contents

- [Usage](#usage)
- [DDPM](#ddpm)
- [DeepLense VAE](#deeplense-vae)
- [CGDPM](#cgdpm)
- [Super-Resolution DDPM](#super-resolution-ddpm)
- [Denoising DDPM](#denoising-ddpm)
- [Resources](#resources)
- [Citations](#citations)

## Usage

AstroGen uses ordinary PyTorch modules. Import an implementation directly from
its documented package, provide the tensor contract described by that model,
and use it in a standard PyTorch training or inference loop. Each section
below gives that contract and a minimal example.

## DDPM

[DDPM](https://arxiv.org/abs/2006.11239) learns to reverse a fixed noising
process, turning noise back into data one step at a time. `DDPM` implements
only that training and sampling logic, reusing a standard denoiser and
schedule from diffusers rather than a custom architecture.

Dimensionality follows the `unet` rather than a separate flag: a
`UNet2DModel` for images shaped `(batch, 1, height, width)`, or a
`UNet1DModel` for spectra shaped `(batch, 1, length)`. Both expect values
normalized to approximately `[-1, 1]`.

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

Discrete labels (e.g. `axion`/`CDM`/`no_sub` substructure type) enter through
the U-Net's class embedding, so they need a `UNet2DModel` built with
`num_class_embeds`; `UNet1DModel` has no such path and raises instead of
silently ignoring the labels:

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

`DeepLenseVAE` is a compact convolutional [variational autoencoder](https://arxiv.org/abs/1312.6114)
based on DeepLense's lens-image VAE. Unlike the DDPM, it learns a Gaussian
latent space and generates images in a single decoder pass.

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

## CGDPM

`CGDPM` is the conditional Gaussian diffusion model of the
[DeepLense super-resolution reference](https://github.com/ML4SCI/DeepLense/blob/main/Super_Resolution_Atal_Gupta/models/cgdpm.ipynb).
It customizes both the denoiser and the noise process, so it is implemented
directly rather than as a diffusers subclass. Conditioning is not a single
input-layer concatenation: the condition is resampled to each resolution and
concatenated into every residual block, so the signal reaches all scales of
the U-Net. Linear attention — attention in time linear in pixel count, from
[Efficient Attention](https://arxiv.org/abs/1812.01243) — runs at every
scale, gated by a zero-initialized
[ReZero](https://arxiv.org/abs/2003.04887) residual so training starts from
the plain convolutional path. The betas follow a cosine-shaped schedule
([improved DDPM](https://arxiv.org/abs/2102.09672)), which destroys
information more slowly than a linear one, and the noise-prediction loss is
L1 rather than L2.

Images and condition are single tensors in `[-1, 1]`, and `sample` returns
that same space. Because each block resamples it, the condition need not
match the sample's spatial size. Two departures from the source, which works
in `[0, 1]`, are documented on the class: no terminal shift-and-rescale of
samples, and an optional `clip_sample` of predicted `x_0` per reverse step,
named after `DDPMScheduler` (see
[docs/diffusers-conventions.md](docs/diffusers-conventions.md)). With
`clip_sample=False` the reverse step is the source's equation unchanged.

For super-resolution, prefer [Super-Resolution DDPM](#super-resolution-ddpm)
below, which wraps `CGDPM` and handles the low-resolution conditioning.

```python
import torch
from astrogen.models import CGDPM

model = CGDPM(image_channels=1, condition_channels=1)
lens_images = torch.rand(8, 1, 64, 64) * 2 - 1
condition = torch.rand(8, 1, 64, 64) * 2 - 1
loss = model(lens_images, condition)
samples = model.sample(count=4, sample_size=64, condition=condition[:4])
```

## Super-Resolution DDPM

`SuperResolutionDDPM` conditions the reverse process on a low-resolution
image, the [SR3](https://arxiv.org/abs/2104.07636) formulation of
super-resolution as conditional diffusion. It wraps [CGDPM](#cgdpm),
bilinearly upsampling the low-resolution image to the target size and passing
it as that model's condition.

Low- and high-resolution images must share a channel count and be normalized
to `[-1, 1]`. Wide-dynamic-range astronomical data needs a stretch such as
`asinh` before that mapping, not a bare min-max rescale; see
[examples/train_super_resolution_star.ipynb](examples/train_super_resolution_star.ipynb).

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

`DenoisingDDPM` restores a corrupted sample with the same channel-conditioned
[DDPM](https://arxiv.org/abs/2006.11239) path: the corrupted sample is
concatenated, channel-wise, with the noisy clean sample at every denoising
step. Corruption is a degradation rather than a resolution change, so the two
share a spatial size and no resizing applies.

Corrupted and clean samples must share a channel count and be normalized to
approximately `[-1, 1]`; the caller applies the corruption (e.g. Gaussian
noise plus blur) before training. As with `DDPM`, dimensionality follows the
U-Net: pass `model_cls=UNet1DModel` for spectral denoising, the 1D
counterpart following [spectrai](https://github.com/conor-horgan/spectrai)'s
`spectral_denoising` task.

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
from diffusers import UNet1DModel

model = DenoisingDDPM(model_cls=UNet1DModel)
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
@inproceedings{nichol2021improved,
  title={Improved Denoising Diffusion Probabilistic Models},
  author={Nichol, Alexander Quinn and Dhariwal, Prafulla},
  booktitle={International Conference on Machine Learning},
  year={2021}
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
