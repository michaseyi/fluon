import torch


def is_enclosed(space: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return (0 < index).all() and (index < torch.tensor(space.shape[:-1])).all()


def index_with_zero_boundary(space: torch.Tensor, index: torch.Tensor):
    return space[*index.int()] if is_enclosed(space, index) else torch.zeros(space.shape[-1:], dtype=space.dtype)


def bilinear_1d(space: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    p_x = index
    if ~is_enclosed(space, p_x):
        return torch.zeros(space.shape[-1:], dtype=space.dtype)

    p_x0 = (p_x.round() - 0.5).clamp(0, space.shape[0])
    p_x1 = (p_x.round() + 0.5).clamp(0, space.shape[0])

    q_x0 = index_with_zero_boundary(space,  p_x0)
    q_x1 = index_with_zero_boundary(space, p_x1)

    w_x0x1: torch.Tensor = 1 - ((p_x - p_x0) / (p_x1 - p_x0))

    q_x = (w_x0x1 * q_x0) + ((1 - w_x0x1) * q_x1)

    return q_x


def bilinear_2d(space: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    p_xy = index

    if ~is_enclosed(space, p_xy):
        return torch.zeros(space.shape[-1:], dtype=space.dtype)

    p_x0y0 = (p_xy.round() - 0.5).clamp(torch.tensor(0),
                                        torch.tensor(space.shape[0:2]))
    p_x1y1 = (p_xy.round() + 0.5).clamp(torch.tensor(0),
                                        torch.tensor(space.shape[0:2]))

    p_x1y0 = torch.tensor([p_x1y1[0], p_x0y0[1]])
    p_x0y1 = torch.tensor([p_x0y0[0], p_x1y1[1]])

    q_x0y0 = index_with_zero_boundary(space, p_x0y0)
    q_x1_y0 = index_with_zero_boundary(space, p_x1y0)

    q_x0y1 = index_with_zero_boundary(space, p_x0y1)
    q_x1y1 = index_with_zero_boundary(space, p_x1y1)

    w_x0x1 = 1 - ((p_xy[0] - p_x0y0[0]) /
                  (p_x1y1[0] - p_x0y0[0]))

    q_xy0 = (w_x0x1 * q_x0y0) + ((1 - w_x0x1) * q_x1_y0)
    q_xy1 = (w_x0x1 * q_x0y1) + ((1 - w_x0x1) * q_x1y1)

    w_y0y1 = 1 - ((p_xy[1] - p_x0y0[1]) /
                  (p_x1y1[1] - p_x0y0[1]))

    q_xy = (w_y0y1 * q_xy0) + ((1 - w_y0y1) * q_xy1)

    return q_xy




def bilinear_3d(space: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    p_xyz = index
    if ~is_enclosed(space, p_xyz):
        return torch.zeros(space.shape[-1:], dtype=space.dtype)

    p_x0y0z0 = (p_xyz.round() - 0.5).clamp(torch.tensor(0),
                                           torch.tensor(space.shape[:3]))
    # p_x1y0z0


    # p_x0y0z1
    # p_x1y0z1

    # p_x0y1z0
    # p_x1y1z0

    # p_x0y1z1
    p_x1y1z1 = (p_xyz.round() + 0.5).clamp(torch.tensor(0),
                                           torch.tensor(space.shape[:3]))
    

    ...


def bilinear(space: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    dim = len(space.shape) - 1

    if dim == 1:
        return bilinear_1d(space, index)
    elif dim == 2:
        return bilinear_2d(space, index)
    elif dim == 3:
        return bilinear_3d(space, index)

    else:
        raise ValueError(
            f"Bilinear interpolation is not supported for {dim}-dimensional indices.")



def nearest(space: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return space[*index.int().clamp(max=torch.tensor(space.shape[:-1]) - 1)]

grid = torch.tensor([
    [1, 3, 9, 2, 0],
    [3, 7, 8, 5, 11],
    [1, 4, 12, 1, 13],
    [2, 7, 10, 7, 8],
    [9, 3, 9, 1, 4]
], dtype=torch.float).unsqueeze(-1)


heatmap = bilinear_2d(grid,
                      torch.tensor([4.875, 5.875], dtype=torch.float))


print(heatmap)
