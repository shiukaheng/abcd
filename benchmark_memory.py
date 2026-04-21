import gc
import os
import time
import multiprocessing as mp

import torch

from gs.core.GaussianModel import GaussianModel
from gs.geometry.grid import Grid
from gs.io.colmap import load
from gs.profiling import MemoryMonitor, TrainingContext
from gs.trainers.basic.config import BasicTrainConfig
from gs.trainers.basic.train import train as basic_train
from gs.trainers.grid.config import GridTrainConfig
from gs.trainers.grid.train import train as grid_train
from gs.visualization import Viewer


DATASETS = ["bonsai", "kitchen", "garden", "room", "stump"]

METHODS = [
    ("vanilla", {}),
    (
        "grid_disabled",
        {"extra_cell_compensation": "disabled", "precomposite_enabled": False},
    ),
    (
        "grid_last_gpu",
        {
            "extra_cell_compensation": "last",
            "precomposite_enabled": True,
            "precomposite_storage": "gpu",
        },
    ),
    (
        "grid_last_cpu",
        {
            "extra_cell_compensation": "last",
            "precomposite_enabled": True,
            "precomposite_storage": "cpu",
        },
    ),
]

GRID_SIZE = 5
ITERATIONS = 5000
SYNC_INTERVAL = 250
MIN_GAUSSIANS = 50
IMAGES_SUBDIR = "images_4"

BENCHMARK_DIR = "./benchmark_results"


def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def reset_viewer():
    Viewer.shared_viser["viser"] = None
    Viewer.shared_viser["viewer"] = None


def run_benchmark(dataset: str, method: str, config_overrides: dict):
    dataset_path = f"./datasets/{dataset}"
    output_dir = os.path.join(BENCHMARK_DIR, f"{dataset}_{method}")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.ply")
    training_csv_path = os.path.join(output_dir, "training.csv")
    memory_csv_path = os.path.join(output_dir, "memory.csv")

    print(f"\n{'=' * 60}")
    print(f"Running: {dataset} / {method}")
    print(f"{'=' * 60}\n")

    reset_viewer()
    cleanup_memory()
    cameras, sparse = load(dataset_path, IMAGES_SUBDIR)

    pid = os.getpid()

    with MemoryMonitor(pid, memory_csv_path, interval_ms=100):
        with TrainingContext(training_csv_path):
            if method == "vanilla":
                config = BasicTrainConfig(
                    iterations=ITERATIONS,
                    preview_camera=None,
                )
                input_model = GaussianModel.from_point_cloud(sparse)
                output_model = basic_train(input_model, cameras, config)
            else:
                config = GridTrainConfig(
                    iterations=ITERATIONS,
                    grid_config=Grid(grid_size=GRID_SIZE),
                    min_gaussians=MIN_GAUSSIANS,
                    preview_camera=None,
                    sync_interval=SYNC_INTERVAL,
                    **config_overrides,
                )
                input_model = GaussianModel.from_point_cloud(sparse)
                output_model = grid_train(input_model, cameras, config)

    output_model.save_ply(model_path)
    print(f"Model saved to {model_path}")
    print(f"Training log: {training_csv_path}")
    print(f"Memory log: {memory_csv_path}")

    del output_model
    del input_model
    del cameras
    del sparse
    reset_viewer()
    cleanup_memory()


def main():
    mp.set_start_method("spawn", force=True)

    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    total = len(DATASETS) * len(METHODS)
    current = 0

    start_time = time.time()

    for dataset in DATASETS:
        for method, config_overrides in METHODS:
            current += 1
            print(f"\n[{current}/{total}] Starting {dataset} / {method}")
            run_benchmark(dataset, method, config_overrides)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Benchmark complete!")
    print(f"Total time: {elapsed / 3600:.2f} hours")
    print(f"Results saved to {BENCHMARK_DIR}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
