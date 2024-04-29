from dataclasses import dataclass
from typing import Literal, Union
import torch
from gs.core.GaussianModel import GaussianModel
from gs.geometry.grid import Grid
from gs.trainers.basic.config import BasicTrainConfig

@dataclass
class AutoGridConfig:
    target_num_cells: int = 8
    center: bool = True

    def apply(self, model: GaussianModel) -> Grid:
        model_bounding_box = model.calculate_bounding_box()
        target_cell_volume = model_bounding_box.volume / self.target_num_cells
        grid_size = target_cell_volume ** (1/3)
        if self.center:
            grid_origin = model_bounding_box.center.to("cpu")
        else:
            grid_origin = torch.zeros(3)
        return Grid(grid_size, grid_origin)

@dataclass
class GridTrainConfig(BasicTrainConfig):
    sync_interval: int = 1000
    grid_config: Union[Grid, AutoGridConfig] = AutoGridConfig()
    max_memory: Union[int, None] = None
    max_gaussians: Union[int, None] = None
    extra_cell_compensation: Literal["last", "uniform", "disabled"] = "uniform"
    min_gaussians: int = 50