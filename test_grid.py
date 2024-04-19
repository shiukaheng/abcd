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
input_model = GaussianModel.from_point_cloud(pointcloud, constant_scale=0.02, scales_range=(0.01, 0.05))

train(
    input_model, 
    cameras, 
    GridTrainConfig(
        grid=Grid(100, 
        grid_origin=torch.Tensor([0,0,55])), 
        sync_interval=500,
        densify_interval=50,
        scales_lr=0.01,
        extra_cell_compensation="last",
        preview_camera=cameras[11],
        min_gaussians=100,
    ))