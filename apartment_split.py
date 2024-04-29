# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from gs.geometry.grid import Grid
import torch
from gs.io.colmap import load
from gs.core.GaussianModel import GaussianModel
from gs.trainers.grid.GridGaussianModel import GridGaussianModel
from gs.trainers.grid.train import train
from gs.trainers.grid.config import GridTrainConfig

# Load COLMAP dataset
cameras, pointcloud = load('./datasets/apartment')

# Create model
input_model = GaussianModel.from_point_cloud(pointcloud, constant_scale=0.5, scales_range=(0.01, 1.0))

output_model = train(
    input_model, 
    cameras, 
    GridTrainConfig(
        iterations=20000,
        densify_until_iter=15000,
        grid_config=Grid(100, 
        grid_origin=torch.Tensor([0,0,55])), 
        # sync_interval=1750,
        sync_interval=240,
        densify_interval=100,
        scales_lr=0.01,
        rotations_lr=0.01,
        positions_lr_init=0.0004,
        positions_lr_final=0.00002,
        extra_cell_compensation="last",
        preview_camera=cameras[11],
        min_gaussians=100,
    ))

# Save model
output_model.save_ply("./apartment_split.ply")