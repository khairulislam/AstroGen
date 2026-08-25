"""Minimal end-to-end example: train and sample DDPM on a small tensor dataset.

Uses random tensors standing in for lens images (64x64, values in [-1, 1])
rather than a real dataset, since data loading stays outside the core
library. Run from the repository root with:

    python -m examples.train_lens_ddpm
"""

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader, TensorDataset

from astrogen.models import DDPM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    dataset = TensorDataset(torch.rand(64, 1, 32, 32) * 2 - 1)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    unet = UNet2DModel(
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=32,
    )
    model = DDPM(unet, DDPMScheduler(num_train_timesteps=1_000)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(3):
        for (images,) in loader:
            optimizer.zero_grad()
            loss = model(images.to(DEVICE))
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}: loss {loss.item():.4f}")

    samples = model.sample(count=4, sample_size=32)
    print(f"sampled {samples.shape} images in [{samples.min():.2f}, {samples.max():.2f}]")


if __name__ == "__main__":
    main()
