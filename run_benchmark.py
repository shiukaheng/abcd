import gc
import os
from typing import Literal

import torch
import tyro

from gs.core.GaussianModel import GaussianModel
from gs.geometry.grid import Grid
from gs.io.colmap import load
from gs.profiling import Logger
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
    densify_interval: int = 100,
    densify_from_iter: int = 500,
    densify_until_iter: int = 15000,
    densify_grad_threshold: float = 0.0002,
    opacity_threshold: float = 0.005,
    split_n_samples: int = 2,
    split_shrink_factor: float = 0.8,
):
    os.makedirs(output, exist_ok=True)

    model_path = os.path.join(output, "model.ply")
    log_path = os.path.join(output, "benchmark.jsonl")

    print(f"\n{'=' * 60}")
    print(f"Running: {dataset} / {method}")
    print(f"{'=' * 60}\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    cameras, sparse = load(dataset, images_subdir)

    pid = os.getpid()

    with Logger(pid, log_path, interval_ms=100):
        if method == "vanilla":
            config = BasicTrainConfig(
                iterations=iterations,
                preview_camera=None,
                densify_interval=densify_interval,
                densify_from_iter=densify_from_iter,
                densify_until_iter=densify_until_iter,
                densify_grad_threshold=densify_grad_threshold,
                opacity_threshold=opacity_threshold,
                split_n_samples=split_n_samples,
                split_shrink_factor=split_shrink_factor,
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
                densify_interval=densify_interval,
                densify_from_iter=densify_from_iter,
                densify_until_iter=densify_until_iter,
                densify_grad_threshold=densify_grad_threshold,
                opacity_threshold=opacity_threshold,
                split_n_samples=split_n_samples,
                split_shrink_factor=split_shrink_factor,
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
                densify_interval=densify_interval,
                densify_from_iter=densify_from_iter,
                densify_until_iter=densify_until_iter,
                densify_grad_threshold=densify_grad_threshold,
                opacity_threshold=opacity_threshold,
                split_n_samples=split_n_samples,
                split_shrink_factor=split_shrink_factor,
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
                densify_interval=densify_interval,
                densify_from_iter=densify_from_iter,
                densify_until_iter=densify_until_iter,
                densify_grad_threshold=densify_grad_threshold,
                opacity_threshold=opacity_threshold,
                split_n_samples=split_n_samples,
                split_shrink_factor=split_shrink_factor,
            )
            input_model = GaussianModel.from_point_cloud(sparse)
            output_model = grid_train(input_model, cameras, config)

    output_model.save_ply(model_path)
    print(f"Model saved to {model_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    tyro.cli(run_benchmark)
