# This is an example of training a vanilla 3D Gaussian Splatting model.

import os
from typing import Union

from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.trainers.basic.config import BasicTrainConfig
from gs.trainers.basic.train import train as basic_train

def get_save_path(dataset_path: str, save_path: str) -> str:
    # Helper function to get save path
    if save_path is None:
        dataset_name = os.path.basename(dataset_path)
        save_folder = os.path.join("../samples")
        save_filename = f"{dataset_name}_3dgs.ply"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, save_filename)
    return save_path

def train_3dgs(dataset_path: str, save_path: Union[str, None]=None):
    """
    Train a 3D Gaussian Splatting model on a dataset.
    """

    # Import dataset
    cameras, sparse = load(dataset_path)

    # Create configuration for training
    config = BasicTrainConfig(
        preview_camera=cameras[0],
    )

    # Create initial Gaussian model
    input_model = GaussianModel.from_point_cloud(sparse)

    # Train model
    output_model = basic_train(input_model, cameras, config)

    # Save model
    save_path = get_save_path(dataset_path, save_path)

    output_model.save_ply(save_path)

    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    dataset_path = "./datasets/mip_nerf_360/kitchen"
    train_3dgs(dataset_path)