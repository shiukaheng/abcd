import gc
import json
from pathlib import Path

import torch
import tyro

from abcd.core.GaussianModel import GaussianModel
from abcd.geometry.grid import Grid, GridIndex
from abcd.trainers.grid.GridGaussianModel import GridGaussianModel


def _model(positions: torch.Tensor) -> GaussianModel:
    count = len(positions)
    return GaussianModel(
        positions=positions,
        sh_coefficients=torch.zeros((count, 16, 3), device=positions.device),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=positions.device).repeat(
            count, 1
        ),
        scales=torch.full((count, 3), -3.0, device=positions.device),
        opacities=torch.full((count, 1), 2.0, device=positions.device),
        sh_degree=3,
    ).to(positions.device)


def measure_memory(
    output: Path,
    partitions: tuple[int, ...] = (1, 2, 4, 8),
    gaussians_per_partition: int = 128,
) -> Path:
    """Record active-shard VRAM scaling while inactive shards remain on disk."""

    if not torch.cuda.is_available():
        raise RuntimeError("Memory measurement requires CUDA")
    if not partitions or min(partitions) <= 0:
        raise ValueError("partitions must contain positive counts")
    if gaussians_per_partition <= 0:
        raise ValueError("gaussians_per_partition must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for count in partitions:
        positions = torch.zeros((count * gaussians_per_partition, 3))
        positions[:, 0] = (
            torch.arange(count).repeat_interleave(gaussians_per_partition) + 0.25
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        grid_model = GridGaussianModel.from_gaussian_model(
            _model(positions),
            cameras=[],
            grid=Grid(grid_size=1.0),
            model_store_device="cpu",
            model_train_device="cuda",
            min_gaussians=0,
            cache_dir=str(output.parent / f"cache-{count}"),
            cache_fingerprint=f"memory-{count}-{gaussians_per_partition}",
        )
        grid_model.grid_set_active_cell_index(GridIndex(0, 0, 0))
        torch.cuda.synchronize()
        records.append(
            {
                "partitions": count,
                "gaussians_per_partition": gaussians_per_partition,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        del grid_model
        gc.collect()

    baseline = records[0]["peak_allocated_bytes"]
    max_spread = max(record["peak_allocated_bytes"] for record in records) - baseline
    payload = {
        "device": torch.cuda.get_device_name(),
        "records": records,
        "max_peak_spread_bytes": max_spread,
        "pass": max_spread < 1024 * 1024,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["pass"]:
        raise RuntimeError(f"Inactive-shard VRAM spread exceeded 1 MiB: {max_spread}")
    return output


def main() -> None:
    tyro.cli(measure_memory)


if __name__ == "__main__":
    main()
