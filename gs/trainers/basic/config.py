from dataclasses import dataclass
from typing import Literal, Union

from gs.core.View import KnownView


@dataclass
class BasicTrainConfig:
    iterations: int = 5000 
    randomize: bool = True 
    positions_lr_init: float = 0.00016
    positions_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    rotations_lr: float = 0.001
    scales_lr: float = 0.005
    opacities_lr: float = 0.05
    sh_coefficients_lr: float = 0.0025
    model_train_device: Union[str, None] = "cuda"
    model_store_device: Union[str, None] = "cpu"
    camera_train_device: Union[str, None] = "cuda"
    camera_store_device: Union[str, None] = "cpu"
    scene_scale: float = None
    up_sh_interval: int = 1000
    densify_interval: int = 100
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002
    opacity_reset_interval: int = 3000
    opacity_threshold: float = 0.005
    screen_size_threshold: float = 20
    world_size_threshold_multiplier: float = 0.1
    reset_to_opacity: float = 0.01
    max_memory: Union[int, None] = None
    max_gaussians: Union[int, None] = None
    starting_iter: int = 0
    ending_iter: Union[int, None] = None
    preview_camera: Union[KnownView, Literal["all"], None] = None
    transparency_loss_weight: float = 0.0
    split_n_samples: int = 2
    split_shrink_factor: float = 0.8