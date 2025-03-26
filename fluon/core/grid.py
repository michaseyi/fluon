from typing import List, Literal
import torch
from fluon.math.interpolation import bilinear, nearest
import torch.nn.functional as F


class Grid:
    def __init__(self,
                 center: List[float],
                 extent: List[float],
                 resolution: List[int],
                 quantity_dim: int = 1,
                 dtype: torch.dtype = torch.float32,
                 interpolation_mode: Literal["bilinear"] | Literal["nearest"] = "nearest"
                 ):

        if not (len(center) == len(extent) == len(resolution)):
            raise ValueError(f"Mismatch in dimensions: center({len(center)}), extent({len(extent)}), resolution({len(resolution)})."
                             "All must have the same length.")

        self.space_dim = len(center)
        self.center = torch.tensor(center, dtype=dtype)
        self.extent = torch.tensor(extent, dtype=dtype)
        self.resolution = torch.tensor(resolution, dtype=torch.int32)
        self.dtype = dtype
        self.grid = torch.zeros([*resolution, quantity_dim], dtype=dtype)
        self.interpolation_mode = interpolation_mode

        if interpolation_mode == 'bilinear':
            self.interpolator = bilinear
        else:
            self.interpolator = nearest

    def sample(self, point: List[float]) -> torch.Tensor:

        if len(point) != self.space_dim:
            raise ValueError(
                f"Mismatch in dimensions: point({len(point)}), self.space_dim({self.space_dim})."
                "All must have the same number of dimensions")

        # Todo: Use a transformation matrix to compute index and cache the matrix
        query = torch.tensor(point, dtype=self.dtype)
        half_extent = self.extent / 2
        lower_bound = self.center - half_extent
        index = (query - lower_bound) / (self.extent / self.resolution)

        # Todo: Implement wrapping and allow defined constants to be returned for queries outstide the domain
        if (index < torch.zeros_like(index)).any() or (index > self.resolution.to(self.dtype)).any():
            raise IndexError(
                f"Index out of bounds: index({index})."
                "Index must be within the resolution bounds.")

        return self.interpolator(self.grid, index)


grid = Grid(center=[0, 0], extent=[2, 2],
            resolution=[100, 100], quantity_dim=1, interpolation_mode='nearest')


# print(grid.sample([0, 0, 0]))


# grid = torch.ones((1, 2, 100, 100))
# input = torch.tensor([-1, 0], dtype=torch.float).view(1, 1, 1, 2)

# output = F.grid_sample(grid, input, mode='bilinear',
#                        padding_mode='zeros', align_corners=True)


# print(output.shape, output)
