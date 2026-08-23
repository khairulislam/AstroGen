# AstroGen

AstroGen is a lightweight library of generative-model cores for astronomy data.

## Table of contents

- [Usage](#usage)
- [Lens Image DDPM](#lens-image-ddpm)
- [Resources](#resources)
- [Citations](#citations)

## Usage

AstroGen uses ordinary PyTorch modules. Import an implementation directly from
its documented package, provide the tensor contract described by that model,
and use it in a standard PyTorch training or inference loop.

## Lens Image DDPM

`LensImageDDPM` is a compact unconditional 2D denoising diffusion model for
gravitational-lens images. It implements the [DDPM method](https://arxiv.org/abs/2006.11239)
with a time-conditioned U-Net denoiser.

It accepts normalized, single-channel images with shape
`(batch, 1, height, width)` and values approximately in `[-1, 1]`.

```python
from astrogen.tasks import LensImageDDPM

model = LensImageDDPM()
loss = model(lens_images)
samples = model.sample(count=4, image_size=64)
```

## Resources

* [DeepLense](https://github.com/ML4SCI/DeepLense/tree/main): Explores cutting-edge Machine Learning techniques for the study of Strong Gravitational Lensing and Dark Matter Sub-structure, using both simulated and real lensing images.
* [STAR](https://github.com/GuoCheng12/STAR/tree/main): The STAR (Super-Resolution for Astronomical Star Fields) dataset is a large-scale benchmark for developing field-level super-resolution models in astronomy.
* [Spectrai](https://github.com/conor-horgan/spectrai): an open-source deep learning framework designed to facilitate the training of neural networks on spectral data and enable comparison between different methods.
* [Hubble meets Webb](https://github.com/vkinakh/Hubble-meets-Webb): A study on the image-to-image translation problem for the prediction of future satellite Webb images from the available Hubble images.

## Citations

Please cite the DDPM method when using this implementation.

```bibtex
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2006.11239},
  year={2020}
}
```
