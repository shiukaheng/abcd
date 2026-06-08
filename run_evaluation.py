import os

import torch
import tyro

from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.eval import eval_views


def run_evaluation(
    model: str,
    dataset: str,
    output: str,
    images_subdir: str = "images_4",
):
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model}")
    print(f"Dataset: {dataset}")
    print(f"{'=' * 60}\n")

    loaded_model = GaussianModel.from_ply(model)
    loaded_model.to("cuda")

    cameras, _ = load(dataset, images_subdir)

    print(f"Loaded {len(cameras)} cameras")
    print(f"Model has {loaded_model.positions.shape[0]} Gaussians")

    print("\nEvaluating...")
    results = eval_views(cameras, loaded_model)

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    results.to_csv(output, index=False)
    print(f"Results saved to {output}")

    print("\n" + "=" * 60)
    print("Mean Metrics:")
    print(f"  PSNR:  {results['psnr'].mean():.4f}")
    print(f"  SSIM:  {results['ssim'].mean():.4f}")
    print(f"  LPIPS: {results['lpips'].mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    tyro.cli(run_evaluation)
