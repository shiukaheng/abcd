from dataclasses import dataclass
from typing import Literal, Union

from gs.core.View import KnownView


@dataclass
class BasicTrainConfig:

    # Iterations to train model for
    iterations: int = 5000 

    # Whether to randomize image order during training
    randomize: bool = True 

    # Learning rate for positions (uses training schedule)
    positions_lr_init: float = 0.00016
    positions_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000

    # Other learning rates
    rotations_lr: float = 0.001
    scales_lr: float = 0.005
    opacities_lr: float = 0.05
    sh_coefficients_lr: float = 0.0025

    # Device to train model on
    model_train_device: Union[str, None] = "cuda"
    model_store_device: Union[str, None] = "cpu"
    camera_train_device: Union[str, None] = "cuda"
    camera_store_device: Union[str, None] = "cpu"

    # Scene scale, affects learning rate. Automatically estimated if None.
    scene_scale: float = None

    # How often to up sample SH coefficients (starting from 0)
    up_sh_interval: int = 1000

    # When and how often to densify the model
    densify_interval: int = 100
    densify_from_iter: int = 500
    densify_until_iter: int = 15000

    # How big does the gradient (proxy for error) have to be to densify
    densify_grad_threshold: float = 0.0002

    # How often to reset opacity (only happens during densification)
    opacity_reset_interval: int = 3000

    # What opacity to reach for pruning
    opacity_threshold: float = 0.005

    # What screen radius to reach for pruning
    screen_size_threshold: float = 20

    # Factor controlling how large a Gaussian has to be in world space to be pruned
    world_size_threshold_multiplier: float = 0.1

    # During opacity reset, what opacity to set to
    reset_to_opacity: float = 0.01

    # Maximum memory to use before stopping densification
    max_memory: Union[int, None] = None 

    # Maximum number of gaussians to use before stopping densification
    max_gaussians: Union[int, None] = None

    # Starting and ending iteration (for resuming training)
    starting_iter: int = 0
    ending_iter: Union[int, None] = None

    # Selected camera for debugging training process 
    # (None for no preview, "all" for whatever is being trained, or specific camera)
    preview_camera: Union[KnownView, Literal["all"], None] = None

    # Additional loss for transparency of rendered image, 
    # encourages model to not rely on the black background
    transparency_loss_weight: float = 0.0

    # How many Gaussians to split a existing Gaussian into during densification
    split_n_samples: int = 2

    # How much to shrink the Gaussians during densification, 
    # on top of the factor from the number of samples
    split_shrink_factor: float = 0.8

    # Additional loss for Gaussians with opacities that are not 0 or 1. 
    # Encourages Gaussians to be either fully opaque or fully transparent.
    opacity_uncertainty_penalty: float = 0.0