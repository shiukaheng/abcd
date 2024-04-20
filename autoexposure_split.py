
###

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
cameras, pointcloud = load('./datasets/autoexposure')

# Create model
input_model = GaussianModel.from_point_cloud(
    pointcloud, 
    constant_scale=0.05, 
    scales_range=(0.0001, 0.5)
)

output_model = train(
    input_model, 
    cameras, 
    GridTrainConfig(
        grid=Grid(15, 
        grid_origin=torch.Tensor([0,0,0])), 
        sync_interval=250,
        extra_cell_compensation="last",
        min_gaussians=100,
        preview_camera=cameras[10],

        iterations=20000,
        densify_until_iter=15000,
        densify_interval=100,
        scales_lr=0.01,
        densify_from_iter=200,
        # sh_mlp_lr=0.005,
        transparency_loss_weight=0.1,
        split_n_samples=4,
        split_shrink_factor=0.7,
        positions_lr_init=0.0006,
        positions_lr_final=0.000016,
    ))

# Save model
output_model.save_ply("./autoexposure_split.ply")