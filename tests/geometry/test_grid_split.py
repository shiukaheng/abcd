from collections import Counter

import pytest
import torch
from conftest import make_gaussian_model

from abcd.geometry.bounding_box import BoundingBox
from abcd.geometry.grid import Grid, GridIndex
from abcd.trainers.grid.grid_utils import cut, merge_model, split_model


def test_point_to_cell_uses_floor_for_negative_coordinates():
    grid = Grid(grid_size=1.0)
    assert grid.point_to_cell(torch.tensor([-0.1, 0.0, 0.0])) == GridIndex(-1, 0, 0)
    assert grid.point_to_cell(torch.tensor([0.0, 0.0, 0.0])) == GridIndex(0, 0, 0)
    assert grid.point_to_cell(torch.tensor([1.0, 0.0, 0.0])) == GridIndex(1, 0, 0)


def test_cut_owns_lower_boundary_and_excludes_upper_boundary():
    model = make_gaussian_model(torch.tensor([[0.0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]]))
    model._gradient_accumulator.copy_(torch.tensor([[1.0], [2.0], [3.0]]))
    model._gradient_accumulator_denominator.copy_(torch.tensor([[4.0], [5.0], [6.0]]))
    model.max_radii2D.copy_(torch.tensor([[7.0], [8.0], [9.0]]))
    selected = cut(
        model,
        BoundingBox(torch.tensor([0.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0])),
    )
    assert selected.positions[:, 0].tolist() == [0.0, 0.5]
    assert selected._gradient_accumulator[:, 0].tolist() == [1.0, 2.0]
    assert selected._gradient_accumulator_denominator[:, 0].tolist() == [4.0, 5.0]
    assert selected.max_radii2D[:, 0].tolist() == [7.0, 8.0]


def test_split_retains_sparse_cells_and_conserves_gaussians():
    positions = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1_000_000.0, 0.0, 0.0],
        ]
    )
    model = make_gaussian_model(positions)
    split = split_model(model, Grid(grid_size=1.0), min_gaussians=100)

    assert list(split) == [
        GridIndex(-1, 0, 0),
        GridIndex(0, 0, 0),
        GridIndex(1, 0, 0),
        GridIndex(1_000_000, 0, 0),
    ]
    merged = merge_model(list(split.values()), device="cpu", clean=True)
    expected = Counter(tuple(row) for row in positions.tolist())
    actual = Counter(tuple(row) for row in merged.positions.detach().tolist())
    assert actual == expected
    assert len(merged) == len(model)


def test_split_rejects_empty_models():
    model = make_gaussian_model(torch.empty((0, 3)))
    with pytest.raises(ValueError, match="empty"):
        split_model(model, Grid(grid_size=1.0), min_gaussians=0)
