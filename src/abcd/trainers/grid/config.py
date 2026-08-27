from dataclasses import dataclass, field
from typing import Literal, Union

import torch

from abcd.core.GaussianModel import GaussianModel
from abcd.geometry.grid import Grid
from abcd.trainers.basic.config import BasicTrainConfig


@dataclass
class AutoGridConfig:
    target_num_cells: int = 8
    center: bool = True

    def apply(self, model: GaussianModel) -> Grid:
        model_bounding_box = model.calculate_bounding_box()
        target_cell_volume = model_bounding_box.volume / self.target_num_cells
        grid_size = target_cell_volume ** (1 / 3)
        if self.center:
            grid_origin = model_bounding_box.center.to("cpu")
        else:
            grid_origin = torch.zeros(3)
        return Grid(grid_size, grid_origin)


@dataclass
class GridTrainConfig(BasicTrainConfig):
    sync_interval: int = 1000
    grid_config: Union[Grid, AutoGridConfig] = field(default_factory=AutoGridConfig)
    extra_cell_compensation: Literal["last", "uniform", "disabled"] = "last"
    min_gaussians: int = 50
    precomposite_enabled: bool = True
    precomposite_storage: Literal["gpu", "cpu"] = "gpu"
    cache_storage: Literal["disk", "ram"] = "disk"
    cache_dir: str | None = None
    cache_fingerprint: str = "abcd-v1"
    resume: bool = False
