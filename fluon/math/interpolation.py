import torch


def bilinear(space: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    ...


def nearest(space: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return space[*index.int().clamp(max=torch.tensor(space.shape[:-1]) - 1)]
