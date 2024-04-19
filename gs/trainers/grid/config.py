from dataclasses import dataclass
from typing import Literal, Union
import torch
from gs.geometry.grid import Grid
from gs.trainers.basic.config import BasicTrainConfig


@dataclass
class GridTrainConfig(BasicTrainConfig):
    sync_interval: int = 1000
    grid: Grid = Grid()
    max_memory: Union[int, None] = None
    max_gaussians: Union[int, None] = None
    extra_cell_compensation: Literal["last", "uniform", "disabled"] = "uniform"