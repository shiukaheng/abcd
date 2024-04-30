import torch
from gs.core.GaussianModel import GaussianModel
from gs.geometry.grid import Grid
from gs.io.ply import read_ply
from gs.io.colmap import load
from gs.trainers.basic.train import train
from gs.trainers.grid.config import GridTrainConfig

if __name__ == "__main__":
    # Load regular COLMAP dataset
    cameras, sparse = load("./datasets/corridor")
    # Load additional dense reconstruction
    dense = read_ply("./datasets/corridor/dense_downsample_2.ply")
    
    input_model = GaussianModel.from_point_cloud(dense, constant_scale=0.03, scales_range=(0.001, 0.5))

    output_model = train(
    input_model, 
    cameras, 
    GridTrainConfig(
        iterations=20000,
        densify_until_iter=15000,
        grid_config=Grid(30, 
        grid_origin=torch.Tensor([0,0,-5])), 
        sync_interval=250,
        densify_interval=2000,
        scales_lr=0.02,
        rotations_lr=0.02,
        extra_cell_compensation="last",
        # preview_camera=cameras[471],
        preview_camera="all",
        min_gaussians=1000,
        densify_from_iter=10000,
    ))

    # Save model
    output_model.save_ply("./dense_gs_reconstruction.ply")