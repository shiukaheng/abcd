import torch
from gs.core.GaussianModel import GaussianModel
from gs.geometry.grid import Grid
from gs.io.ply import read_ply
from gs.io.colmap import load
from gs.trainers.grid.grid_utils import split_model
from gs.trainers.basic.train import train
from gs.trainers.grid.config import AutoGridConfig, GridTrainConfig
from gs.visualization.Viewer import Viewer

if __name__ == "__main__":
    cameras, sparse = load("./datasets/mip_nerf_360/treehill/")
    input_model = GaussianModel.from_point_cloud(sparse).cuda()

    output_model = train(
    input_model, 
    cameras, 
    GridTrainConfig(
        iterations=10000,
        densify_until_iter=8000,
        grid_config=Grid(40, 
        grid_origin=torch.Tensor([0,0,0])), 
        sync_interval=250,
        densify_interval=100,
        extra_cell_compensation="last",
        preview_camera=cameras[105],
        min_gaussians=500,
    ))

    # Save model
    output_model.save_ply("./samples/treehill_basic_bench.ply")