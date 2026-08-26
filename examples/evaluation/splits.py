"""Train/validation split shared by the training-notebook examples."""

import torch


def train_val_split(
    *tensors: torch.Tensor, val_fraction: float = 0.2, seed: int = 0
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Split each of ``tensors`` along dim 0 into ``(train, val)`` tuples.

    All tensors share one random permutation, so paired tensors (e.g.
    images and labels) stay aligned.
    """
    count = tensors[0].shape[0]
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    val_count = max(1, int(count * val_fraction))
    val_index, train_index = permutation[:val_count], permutation[val_count:]
    train = tuple(tensor[train_index] for tensor in tensors)
    val = tuple(tensor[val_index] for tensor in tensors)
    return train, val
