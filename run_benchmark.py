import gc
import os
from typing import Literal

import torch
import tyro

from gs.core.GaussianModel import GaussianModel
from gs.geometry.grid import Grid
from gs.io.colmap import load
from gs.profiling import MemoryMonitor, TrainingContext
from gs.trainers.basic.config import BasicTrainConfig
from gs.trainers.basic.train import train as basic_train
from gs.trainers.grid.config import GridTrainConfig
from gs.trainers.grid.train import train as grid_train


def run_benchmark(
    dataset: str,
    output: str,
    method: Literal["vanilla", "grid_naive", "grid_gpu", "grid_cpu"],
    iterations: int = 5000,
    grid_size: int = 5,
    sync_interval: int = 250,
    min_gaussians: int = 50,
    images_subdir: str = "images_4",
):
    os.makedirs(output, exist_ok=True)

    model_path = os.path.join(output, "model.ply")
    training_csv_path = os.path.join(output, "training.csv")
    memory_csv_path = os.path.join(output, "memory.csv")

    print(f"\n{'=' * 60}")
    print(f"Running: {dataset} / {method}")
    print(f"{'=' * 60}\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    cameras, sparse = load(dataset, images_subdir)

    pid = os.getpid()

    with MemoryMonitor(pid, memory_csv_path, interval_ms=100):
        with TrainingContext(training_csv_path):
            if method == "vanilla":
                config = BasicTrainConfig(
                    iterations=iterations,
                    preview_camera=None,
                )
                input_model = GaussianModel.from_point_cloud(sparse)
                output_model = basic_train(input_model, cameras, config)
            elif method == "grid_naive":
                config = GridTrainConfig(
                    iterations=iterations,
                    grid_config=Grid(grid_size=grid_size),
                    min_gaussians=min_gaussians,
                    preview_camera=None,
                    sync_interval=sync_interval,
                    extra_cell_compensation="disabled",
                    precomposite_enabled=False,
                )
                input_model = GaussianModel.from_point_cloud(sparse)
                output_model = grid_train(input_model, cameras, config)
            elif method == "grid_gpu":
                config = GridTrainConfig(
                    iterations=iterations,
                    grid_config=Grid(grid_size=grid_size),
                    min_gaussians=min_gaussians,
                    preview_camera=None,
                    sync_interval=sync_interval,
                    extra_cell_compensation="last",
                    precomposite_enabled=True,
                    precomposite_storage="gpu",
                )
                input_model = GaussianModel.from_point_cloud(sparse)
                output_model = grid_train(input_model, cameras, config)
            elif method == "grid_cpu":
                config = GridTrainConfig(
                    iterations=iterations,
                    grid_config=Grid(grid_size=grid_size),
                    min_gaussians=min_gaussians,
                    preview_camera=None,
                    sync_interval=sync_interval,
                    extra_cell_compensation="last",
                    precomposite_enabled=True,
                    precomposite_storage="cpu",
                )
                input_model = GaussianModel.from_point_cloud(sparse)
                output_model = grid_train(input_model, cameras, config)

    output_model.save_ply(model_path)
    print(f"Model saved to {model_path}")
    print(f"Training log: {training_csv_path}")
    print(f"Memory log: {memory_csv_path}")


if __name__ == "__main__":
    tyro.cli(run_benchmark)
