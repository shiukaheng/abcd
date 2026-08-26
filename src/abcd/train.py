import gc
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import tyro

from abcd.core.GaussianModel import GaussianModel
from abcd.core.View import KnownView
from abcd.geometry.grid import Grid
from abcd.io.colmap import load
from abcd.io.split import split_train_test_cameras
from abcd.profiling import Logger
from abcd.trainers.basic.config import BasicTrainConfig
from abcd.trainers.basic.train import train as basic_train
from abcd.trainers.grid.config import GridTrainConfig
from abcd.trainers.grid.train import train as grid_train

Method = Literal["3dgs", "abcd", "abcd-no-compositing"]


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    dataset: Path,
    output: Path,
    method: Method = "abcd",
    iterations: int = 5_000,
    partition_size: float = 5.0,
    sync_interval: int = 250,
    images_subdir: str = "images_4",
    holdout_every: int = 8,
    seed: int = 0,
    densify_interval: int = 100,
    densify_from_iter: int = 500,
    densify_until_iter: int = 15_000,
    densify_grad_threshold: float = 0.0002,
    opacity_threshold: float = 0.005,
    split_n_samples: int = 2,
    split_shrink_factor: float = 0.8,
    preview: str | None = None,
) -> None:
    """Train the 3DGS baseline, ABCD, or the compositing ablation."""

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA-capable GPU")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if partition_size <= 0:
        raise ValueError("partition_size must be positive")
    if sync_interval <= 0:
        raise ValueError("sync_interval must be positive")

    output.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    cameras, sparse = load(str(dataset), images_subdir)
    train_cameras, test_cameras = split_train_test_cameras(cameras, holdout_every)
    if not train_cameras:
        raise ValueError("The camera split produced no training cameras")
    training_views = cast(list[KnownView], train_cameras)

    if preview is None:
        preview_camera = None
    elif preview == "all":
        preview_camera = "all"
    elif preview.isdigit():
        preview_camera = train_cameras[int(preview)]
    else:
        raise ValueError("preview must be omitted, 'all', or a camera index")

    run_config = {
        "dataset": str(dataset.resolve()),
        "method": method,
        "iterations": iterations,
        "partition_size": partition_size,
        "sync_interval": sync_interval,
        "images_subdir": images_subdir,
        "holdout_every": holdout_every,
        "seed": seed,
        "densify_interval": densify_interval,
        "densify_from_iter": densify_from_iter,
        "densify_until_iter": densify_until_iter,
        "densify_grad_threshold": densify_grad_threshold,
        "opacity_threshold": opacity_threshold,
        "split_n_samples": split_n_samples,
        "split_shrink_factor": split_shrink_factor,
    }
    cache_fingerprint = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        **run_config,
        "git_revision": _git_revision(),
        "torch_version": torch.__version__,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "gpu": torch.cuda.get_device_name(),
        "training_camera_ids": [camera.id for camera in train_cameras],
        "test_camera_ids": [camera.id for camera in test_cameras],
    }
    (output / "run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    common = {
        "iterations": iterations,
        "preview_camera": preview_camera,
        "densify_interval": densify_interval,
        "densify_from_iter": densify_from_iter,
        "densify_until_iter": densify_until_iter,
        "densify_grad_threshold": densify_grad_threshold,
        "opacity_threshold": opacity_threshold,
        "split_n_samples": split_n_samples,
        "split_shrink_factor": split_shrink_factor,
    }
    input_model = GaussianModel.from_point_cloud(sparse)
    with Logger(os.getpid(), str(output / "training.jsonl"), interval_ms=100):
        if method == "3dgs":
            output_model = basic_train(
                input_model, training_views, BasicTrainConfig(**common)
            )
        else:
            composition_enabled = method == "abcd"
            config = GridTrainConfig(
                **common,
                grid_config=Grid(grid_size=partition_size),
                sync_interval=sync_interval,
                extra_cell_compensation=("last" if composition_enabled else "disabled"),
                precomposite_enabled=composition_enabled,
                precomposite_storage="cpu",
                cache_dir=str(output / "cache"),
                cache_fingerprint=cache_fingerprint,
                resume=(output / "cache" / "shards").is_dir(),
            )
            output_model = grid_train(input_model, training_views, config)

    output_model.save_ply(str(output / "model.ply"))


def main() -> None:
    tyro.cli(train)


if __name__ == "__main__":
    main()
