from pathlib import Path
from typing import cast

import tyro

from abcd.core.GaussianModel import GaussianModel
from abcd.core.View import KnownView
from abcd.eval import eval_views
from abcd.io.colmap import load
from abcd.io.split import split_train_test_cameras


def evaluate(
    model: Path,
    dataset: Path,
    output: Path,
    images_subdir: str = "images_4",
    holdout_every: int = 8,
    device: str = "cuda",
) -> None:
    """Evaluate a model on the deterministic held-out camera split."""

    loaded_model = GaussianModel.from_ply(str(model), device=device)
    cameras, _ = load(str(dataset), images_subdir)
    _, test_cameras = split_train_test_cameras(cameras, holdout_every)
    if not test_cameras:
        raise ValueError("The camera split produced no held-out cameras")
    results = eval_views(cast(list[KnownView], test_cameras), loaded_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)


def main() -> None:
    tyro.cli(evaluate)


if __name__ == "__main__":
    main()
