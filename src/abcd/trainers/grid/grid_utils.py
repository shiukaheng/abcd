from typing import Dict, List, Tuple

import torch

from abcd.core.GaussianModel import GaussianModel
from abcd.geometry.bounding_box import BoundingBox
from abcd.geometry.grid import Grid, GridIndex


def bounding_box_mask(model: GaussianModel, bounding_box: BoundingBox) -> torch.Tensor:
    minimum = bounding_box.min.to(model.positions)
    maximum = bounding_box.max.to(model.positions)
    return torch.all((model.positions >= minimum) & (model.positions < maximum), dim=1)


def cut(model: GaussianModel, bounding_box: BoundingBox, invert=False) -> GaussianModel:
    """
    Cuts out a part of a Gaussian model using a bounding box.
    """
    valid = bounding_box_mask(model, bounding_box)
    if invert:
        valid = ~valid
    return model[valid]


def split_model(
    model: GaussianModel, grid: Grid, min_gaussians: int
) -> Dict[GridIndex, Tuple[GaussianModel, BoundingBox]]:
    """
    Splits a Gaussian model into a grid of cells.
    """
    if grid.grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if len(model) == 0:
        raise ValueError("Cannot split an empty Gaussian model")

    origin = grid.grid_origin.to(model.positions)
    coordinates = torch.floor((model.positions - origin) / grid.grid_size).to(
        torch.int64
    )
    indices = sorted(
        {
            GridIndex(*(int(value) for value in coordinate))
            for coordinate in coordinates.detach().cpu().tolist()
        },
        key=lambda index: (index.x, index.y, index.z),
    )

    cell_models = {}
    sparse_cells = 0
    for cell_index in indices:
        cell_coordinate = torch.tensor(
            cell_index, device=coordinates.device, dtype=coordinates.dtype
        )
        mask = torch.all(coordinates == cell_coordinate, dim=1)
        cell_min = cell_coordinate.to(model.positions.dtype) * grid.grid_size + origin
        cell_bounding_box = BoundingBox(cell_min, cell_min + grid.grid_size)
        cell_model = model[mask]
        if len(cell_model) < min_gaussians:
            sparse_cells += 1
        cell_models[cell_index] = (cell_model, cell_bounding_box)

    if sparse_cells:
        print(
            f"Retaining {sparse_cells} occupied cells below "
            f"min_gaussians={min_gaussians} to preserve all Gaussians"
        )
    return cell_models


def merge_model(
    models: List[Tuple[GaussianModel, BoundingBox]], device, clean=True
) -> GaussianModel:
    """
    Merges a list of Gaussian models into a single model. Assumes the models have the same SH degree, SH channels, background color, and scales range.
    """
    if len(models) == 0:
        raise ValueError("No models to merge")
    # Move all models to the same device
    models = [(model.to(device), bounding_box) for model, bounding_box in models]
    if clean:  # Remove Gaussians outside the bounding box
        models = [
            (cut(model, bounding_box), bounding_box) for model, bounding_box in models
        ]
    if len(models) == 0:
        return GaussianModel(
            positions=torch.empty((0, 3)),  # num_gaussians, xyz
            sh_coefficients=torch.empty(
                (
                    0,
                    models[0][0].sh_coefficients.shape[1],
                    models[0][0].sh_coefficients.shape[2],
                )
            ),  # num_gaussians, sh_channels, sh_degree
            rotations=torch.empty((0, 4)),  # num_gaussians, quaternion
            scales=torch.empty((0, 3)),  # num_gaussians, xyz
            opacities=torch.empty((0, 1)),  # num_gaussians, 1
            sh_degree=models[0][0].sh_degree,  # sh_degree
            background_color=models[0][0].background_color,  # rgb
            scales_range=models[0][0].scales_range,  # xyz
        )
    elif len(models) == 1:
        return models[0][0]
    else:
        return GaussianModel.concatenate([model for model, _ in models])
