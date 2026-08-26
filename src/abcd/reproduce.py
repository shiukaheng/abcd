from pathlib import Path

import tyro

from abcd.evaluate import evaluate
from abcd.plot import plot
from abcd.summarize import summarize
from abcd.train import Method, train


def reproduce(
    dataset_root: Path,
    output: Path,
    scenes: tuple[str, ...] = ("garden", "kitchen"),
    methods: tuple[Method, ...] = ("3dgs", "abcd-no-compositing", "abcd"),
    iterations: int = 5_000,
    partition_size: float = 5.0,
    sync_interval: int = 250,
    images_subdir: str = "images_4",
    holdout_every: int = 8,
    seed: int = 0,
    force: bool = False,
) -> None:
    """Run all paper methods, evaluate held-out views, and generate results."""

    for scene in scenes:
        for method in methods:
            run_dir = output / scene / method
            model_path = run_dir / "model.ply"
            evaluation_path = run_dir / "evaluation.csv"
            if force or not model_path.is_file():
                train(
                    dataset=dataset_root / scene,
                    output=run_dir,
                    method=method,
                    iterations=iterations,
                    partition_size=partition_size,
                    sync_interval=sync_interval,
                    images_subdir=images_subdir,
                    holdout_every=holdout_every,
                    seed=seed,
                )
            if force or not evaluation_path.is_file():
                evaluate(
                    model=model_path,
                    dataset=dataset_root / scene,
                    output=evaluation_path,
                    images_subdir=images_subdir,
                    holdout_every=holdout_every,
                )
    results = summarize(output)
    plot(results)


def main() -> None:
    tyro.cli(reproduce)


if __name__ == "__main__":
    main()
