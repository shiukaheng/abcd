from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def split_train_test_cameras(
    cameras: Sequence[T], holdout_every: int = 8
) -> tuple[list[T], list[T]]:
    """Use every Nth camera as a deterministic held-out view."""

    if holdout_every < 2:
        raise ValueError("holdout_every must be at least 2")
    training = []
    testing = []
    for index, camera in enumerate(cameras):
        (testing if index % holdout_every == 0 else training).append(camera)
    return training, testing
