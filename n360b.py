from typing import List
import torch
from gs.core.GaussianModel import GaussianModel
from gs.geometry.grid import Grid
from gs.io.colmap import load
from gs.trainers.grid.train import train as grid_train
from gs.trainers.basic.train import train as basic_train
from gs.trainers.grid.config import GridTrainConfig
import os

if __name__ == "__main__":

    # Discover all datasets within ./datasets/mip_nerf_360/
    datasets: List[str] = []
    for root, dirs, files in os.walk("./datasets/mip_nerf_360/"):
        # If there is a "./images" directory and "./sparse" directory, then it is a dataset
        if "images" in dirs and "sparse" in dirs:
            datasets.append(root)

    for i, dataset in enumerate(datasets):

        dataset_name = dataset.split("/")[-1]
        grid_save_file = f"./samples/{dataset_name}_grid_bench.ply"
        basic_save_file = f"./samples/{dataset_name}_basic_bench.ply"

        print(f"Running dataset {i+1}/{len(datasets)}: {dataset_name}")

        training_config = GridTrainConfig(
            iterations=10000,
            densify_until_iter=8000,
            sync_interval=250,
            densify_interval=100,
            extra_cell_compensation="last",
            min_gaussians=50,
        )

        cameras, sparse = load(dataset)

        # Train grid model
        if not os.path.exists(grid_save_file):
            input_model = GaussianModel.from_point_cloud(sparse).cuda() # Constant scale of 0.01 seems to work well
            output_model = grid_train(
                input_model, 
                cameras, 
                training_config
            )
            output_model.save_ply(grid_save_file)
            del output_model

        # Train basic model
        if not os.path.exists(basic_save_file):
            input_model = GaussianModel.from_point_cloud(sparse).cuda()
            output_model = basic_train(
                input_model, 
                cameras, 
                training_config
            )
            # Save model
            output_model.save_ply(basic_save_file)
            del output_model

    print("Done!")