import pytest

from gs.io.split import split_train_test_cameras


def test_camera_split_is_deterministic_and_disjoint():
    cameras = list(range(10))
    training, testing = split_train_test_cameras(cameras, holdout_every=4)
    assert testing == [0, 4, 8]
    assert training == [1, 2, 3, 5, 6, 7, 9]
    assert set(training).isdisjoint(testing)
    assert sorted(training + testing) == cameras


def test_camera_split_rejects_invalid_interval():
    with pytest.raises(ValueError, match="at least 2"):
        split_train_test_cameras([1, 2], holdout_every=1)
