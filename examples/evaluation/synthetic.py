"""Synthetic image generator shared by the denoising (B1) and
super-resolution (C1) evaluation examples: smooth Gaussian blobs, so a
restoration model has actual structure to recover (plain random noise has
none)."""

import torch


def make_blob_images(count: int, size: int, blobs_per_image: int = 3) -> torch.Tensor:
    coordinate = torch.linspace(-1, 1, size)
    grid_y, grid_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    images = []
    for _ in range(count):
        image = torch.zeros(size, size)
        for _ in range(blobs_per_image):
            center_x, center_y = torch.rand(2) * 1.6 - 0.8
            sigma = 0.15 + 0.15 * torch.rand(1).item()
            amplitude = torch.rand(1).item() * 2 - 1
            image += amplitude * torch.exp(
                -((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / (2 * sigma**2)
            )
        images.append(image)
    return torch.stack(images).unsqueeze(1).clamp(-1, 1)
