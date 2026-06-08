import os
import re
from pathlib import Path

import pandas as pd
import tyro
from tqdm import tqdm

from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.eval import eval_views


def run_all_evaluations(
    benchmark_dir: str = "./benchmark_results",
    output_dir: str = "./evaluation",
    images_subdir: str = "images_4",
):
    print(f"\n{'=' * 60}")
    print(f"Running all evaluations")
    print(f"Benchmark dir: {benchmark_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'=' * 60}\n")

    pattern = re.compile(r"(.+)_(vanilla|grid_naive|grid_gpu|grid_cpu)")

    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        print(f"Benchmark directory not found: {benchmark_dir}")
        return

    model_dirs = []
    for folder in sorted(benchmark_path.iterdir()):
        if not folder.is_dir():
            continue
        match = pattern.match(folder.name)
        if not match:
            continue
        model_path = folder / "model.ply"
        if not model_path.exists():
            continue
        dataset, method = match.groups()
        model_dirs.append((dataset, method, model_path, folder))

    if not model_dirs:
        print("No benchmark models found")
        return

    print(f"Found {len(model_dirs)} models to evaluate\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset, method, model_path, folder in tqdm(model_dirs, desc="Evaluating"):
        dataset_path = f"./datasets/{dataset}"
        eval_output = output_path / f"{dataset}_{method}.csv"

        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}, skipping {dataset}_{method}")
            continue

        model = GaussianModel.from_ply(str(model_path))
        model.to("cuda")

        cameras, _ = load(dataset_path, images_subdir)

        results = eval_views(cameras, model)
        results.to_csv(eval_output, index=False)

        memory_path = folder / "memory.csv"
        if memory_path.exists():
            memory_df = pd.read_csv(memory_path)
            max_ram_mb = memory_df["ram_mb"].max()
            max_vram_mb = memory_df["vram_mb"].max()
            total_time_s = memory_df["timestamp_s"].max()
        else:
            max_ram_mb = 0
            max_vram_mb = 0
            total_time_s = 0

        all_results[f"{dataset}_{method}"] = {
            "psnr": results["psnr"].mean(),
            "ssim": results["ssim"].mean(),
            "lpips": results["lpips"].mean(),
            "max_ram_gb": max_ram_mb / 1024,
            "max_vram_gb": max_vram_mb / 1024,
            "total_time_s": total_time_s,
        }

    summary_path = output_path / "summary.csv"
    with open(summary_path, "w") as f:
        f.write("dataset,method,psnr,ssim,lpips,max_ram_gb,max_vram_gb,total_time_s\n")
        for key in sorted(all_results.keys()):
            dataset, method = key.rsplit("_", 1)
            r = all_results[key]
            f.write(f"{dataset},{method},{r['psnr']:.4f},{r['ssim']:.4f},{r['lpips']:.4f},{r['max_ram_gb']:.2f},{r['max_vram_gb']:.2f},{r['total_time_s']:.1f}\n")

    print(f"\n{'=' * 80}")
    print("Evaluation complete!")
    print(f"Results saved to {output_dir}")
    print("\nSummary:")
    print(f"{'Dataset':<10} {'Method':<10} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'RAM(GB)':>8} {'VRAM(GB)':>9} {'Time(s)':>8}")
    print("-" * 80)
    for key in sorted(all_results.keys()):
        dataset, method = key.rsplit("_", 1)
        r = all_results[key]
        print(f"{dataset:<10} {method:<10} {r['psnr']:>7.2f} {r['ssim']:>7.4f} {r['lpips']:>7.4f} {r['max_ram_gb']:>8.2f} {r['max_vram_gb']:>9.2f} {r['total_time_s']:>8.1f}")
    print("=" * 80)


if __name__ == "__main__":
    tyro.cli(run_all_evaluations)
