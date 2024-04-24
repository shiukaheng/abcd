import torch
from gs.core.GaussianModel import GaussianModel
from gs.geometry.grid import Grid
from gs.io.ply import read_ply
from gs.io.colmap import load
from gs.trainers.grid.train import train
from gs.trainers.grid.config import GridTrainConfig

if __name__ == "__main__":
    cameras, sparse = load("./datasets/mip_nerf_360/kitchen/")
    
    input_model = GaussianModel.from_point_cloud(sparse).cuda()

    output_model = train(
    input_model, 
    cameras, 
    GridTrainConfig(
        iterations=8000,
        densify_until_iter=6000,
        grid=Grid(15, 
        grid_origin=torch.Tensor([0,0,0])), 
        sync_interval=750,
        densify_interval=100,
        # scales_lr=0.02,
        # rotations_lr=0.02,
        extra_cell_compensation="last",
        preview_camera=cameras[266],
        min_gaussians=500,
    ))

    # Save model
    output_model.save_ply("./presentation_lm_bench.ply")