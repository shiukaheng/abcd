# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from gs.geometry.grid import Grid
import torch
from gs.io.colmap import load
from gs.core.GaussianModel import GaussianModel
from gs.trainers.basic.config import BasicTrainConfig
from gs.trainers.basic.train import train

# Load COLMAP dataset
cameras, pointcloud = load('./datasets/minidame')

# Create model
input_model = GaussianModel.from_point_cloud(
    pointcloud, 
    # constant_scale=0.05, 
    # scales_range=(0.0001, 0.5)
)

output_model = train(
    input_model, 
    cameras, 
    BasicTrainConfig(
        iterations=20000,
        densify_until_iter=15000,
        densify_interval=100,
        # scales_lr=0.01,
        densify_from_iter=200,
        sh_mlp_lr=0.03,
        transparency_loss_weight=1,
        split_n_samples=4,
        split_shrink_factor=0.7,
        # positions_lr_init=0.0006,
        # positions_lr_final=0.000016,
        preview_camera="all",
        sh_mlp=True,
    ))

# Save model
output_model.save_ply("./autoexposure.ply")