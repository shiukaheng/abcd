import random
from typing import List
import torch
from tqdm import tqdm
from gs.core.BaseCamera import BaseCamera
from gs.core.GaussianModel import GaussianModel
from gs.helpers.loss import mix_l1_ssim_loss
from gs.helpers.scene import estimate_scene_scale
from gs.trainers.basic.helpers import densify, get_expon_lr_func, prune, prune_opacity_only, reset_opacities
from gs.visualization.TrainingViewer import TrainingViewer

def train(
        model: GaussianModel, 
        cameras: List[BaseCamera], 
        grid_origin: torch.Tensor,
        grid_size: torch.Tensor,
        iterations: int = 5000, 
        randomize: bool = True, 
        positions_lr_init: float = 0.00016,
        positions_lr_final: float = 0.0000016,
        position_lr_delay_mult: float = 0.01,
        position_lr_max_steps: int = 30000,
        rotations_lr: float = 0.001,
        scales_lr: float = 0.005,
        opacities_lr: float = 0.05,
        sh_coefficients_lr: float = 0.0025,
        device: str = "cuda",
        scene_scale: float = None,
        up_sh_interval: int = 1000,
        densify_interval: int = 100,
        densify_from_iter: int = 500,
        densify_until_iter: int = 15000,
        densify_grad_threshold: float = 0.0002,
        opacity_reset_interval: int = 3000,
        opacity_threshold: float = 0.005,
        screen_size_threshold: float = 20,
        world_size_threshold_multiplier: float = 0.1,
        reset_to_opacity: float = 0.01,
    ):
    pass