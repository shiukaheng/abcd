# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from gs.geometry.grid import Grid
import torch
from gs.io.colmap import load
from gs.core.GaussianModel import GaussianModel
from gs.trainers.basic.config import BasicTrainConfig
from gs.trainers.basic.train import train

# Load COLMAP dataset
cameras, pointcloud = load('./datasets/apartment')

# Create model
input_model = GaussianModel.from_point_cloud(pointcloud, constant_scale=0.5, scales_range=(0.01, 1.0))

output_model = train(
    input_model, 
    cameras, 
    BasicTrainConfig(
        iterations=20000,
        densify_until_iter=15000,
        densify_interval=100,
        scales_lr=0.01,
        preview_camera=cameras[11],
        densify_from_iter=200,
    ))

# Save model
output_model.save_ply("./apartment_simple.ply")