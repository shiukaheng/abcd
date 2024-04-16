from typing import Tuple, List
from gs.core.BaseCamera import BaseCamera
from gs.core.GaussianModel import GaussianModel
import torch

def calculate_bounding_box(model: GaussianModel) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError

def get_grid_bounding_boxes(bounding_box: Tuple[torch.Tensor, torch.Tensor], grid_size: torch.Tensor, grid_origin: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    raise NotImplementedError

def divide_model(model: GaussianModel, cameras: BaseCamera, grid_size: torch.Tensor, grid_origin: torch.Tensor) -> List[GaussianModel, List[BaseCamera]]:
    model_bounding_box = calculate_bounding_box(model)
    bounding_boxes = get_grid_bounding_boxes(model_bounding_box, grid_size, grid_origin)
    pass