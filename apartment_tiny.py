from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.trainers.basic import train

# Load COLMAP dataset
cameras, pointcloud = load('./datasets/apartment/') # Replace with your dataset path

# Initialize Gaussian model

model = GaussianModel.from_point_cloud(pointcloud, constant_scale=0.05, min_scale=0.01, max_scale=0.1).cuda()

# Train the model
train(model, cameras, iterations=30000, scales_lr=0.05, positions_lr_init=0.00005, positions_lr_final=0.0000005)