from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm
from gs.core.GaussianModel import GaussianModel
from gs.geometry.bounding_box import BoundingBox
from gs.geometry.grid import Grid, GridIndex

def cut(model: GaussianModel, bounding_box: BoundingBox, invert=False) -> GaussianModel:
    """
    Cuts out a part of a Gaussian model using a bounding box.
    """
    valid = torch.all((model.positions >= bounding_box.min.to(model.positions.device)) & (model.positions <= bounding_box.max.to(model.positions.device)), dim=1)
    if invert:
        valid = ~valid
    return model[valid]

def split_model(model: GaussianModel, grid: Grid, min_gaussians: int) -> Dict[GridIndex, Tuple[GaussianModel, BoundingBox]]:
    """
    Splits a Gaussian model into a grid of cells.
    """
    model_bounding_box = model.calculate_bounding_box()
    cells = grid.calculate_bounding_box_cells(model_bounding_box) # Get the cells that are inside the bounding box
    cell_models = {}
    for cell_bounding_box, cell_index in tqdm(cells, desc="Splitting model into cells"):
        cell_model = cut(model, cell_bounding_box)
        if len(cell_model) < min_gaussians:
            continue
        cell_models[cell_index]=(cell_model, cell_bounding_box)
    return cell_models

def merge_model(models: List[Tuple[GaussianModel, BoundingBox]], device, clean=True) -> GaussianModel:
    """
    Merges a list of Gaussian models into a single model. Assumes the models have the same SH degree, SH channels, background color, and scales range.
    """
    if len(models) == 0:
        raise ValueError("No models to merge")
    # Move all models to the same device
    models = [(model.to(device), bounding_box) for model, bounding_box in models]
    if clean: # Remove Gaussians outside the bounding box
        models = [(cut(model, bounding_box), bounding_box) for model, bounding_box in models]
    if len(models) == 0:
        return GaussianModel(
            positions=torch.empty((0, 3)), # num_gaussians, xyz
            sh_coefficients=torch.empty((0, models[0][0].sh_coefficients.shape[1], models[0][0].sh_coefficients.shape[2])), # num_gaussians, sh_channels, sh_degree
            rotations=torch.empty((0, 4)), # num_gaussians, quaternion
            scales=torch.empty((0, 3)), # num_gaussians, xyz
            opacities=torch.empty((0, 1)), # num_gaussians, 1
            sh_degree=models[0][0].sh_degree, # sh_degree
            background_color=models[0][0].background_color, # rgb
            scales_range=models[0][0].scales_range, # xyz
        )
    elif len(models) == 1:
        return models[0][0]
    else:
        models[0][0].concatenate([model for model, _ in models[1:]])
        return models[0][0]
    