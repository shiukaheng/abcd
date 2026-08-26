import tomllib
from pathlib import Path

import tyro

from abcd.evaluate import evaluate
from abcd.plot import plot
from abcd.summarize import summarize
from abcd.train import train


def reproduce(
    dataset_root: Path,
    output: Path,
    config: Path = Path("configs/siggraph_2026.toml"),
    force: bool = False,
) -> None:
    """Run all paper methods, evaluate held-out views, and generate results."""

    settings = tomllib.loads(config.read_text(encoding="utf-8"))
    for scene in settings["scenes"]:
        for method in settings["methods"]:
            run_dir = output / scene / method
            model_path = run_dir / "model.ply"
            evaluation_path = run_dir / "evaluation.csv"
            if force or not model_path.is_file():
                train(
                    dataset=dataset_root / scene,
                    output=run_dir,
                    method=method,
                    iterations=settings["training"]["iterations"],
                    partition_size=settings["training"]["partition_size"],
                    sync_interval=settings["training"]["sync_interval"],
                    images_subdir=settings["dataset"]["images_subdir"],
                    holdout_every=settings["dataset"]["holdout_every"],
                    seed=settings["training"]["seed"],
                )
            if force or not evaluation_path.is_file():
                evaluate(
                    model=model_path,
                    dataset=dataset_root / scene,
                    output=evaluation_path,
                    images_subdir=settings["dataset"]["images_subdir"],
                    holdout_every=settings["dataset"]["holdout_every"],
                )
    results = summarize(output)
    plot(results)


def main() -> None:
    tyro.cli(reproduce)


if __name__ == "__main__":
    main()
