from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.trainers.basic.train import train

# Load COLMAP dataset
cameras, pointcloud = load('./datasets/stairs/') # Replace with your dataset path

# Initialize Gaussian model

model = GaussianModel.from_point_cloud(pointcloud, constant_scale=0.01).cuda()

# Train the model
train(model, cameras, iterations=30000, scales_lr=0.01, positions_lr_init=0.00005, positions_lr_final=0.0000005)