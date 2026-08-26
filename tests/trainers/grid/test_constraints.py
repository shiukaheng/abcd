import torch
from conftest import make_gaussian_model

from abcd.geometry.grid import Grid, GridIndex
from abcd.trainers.grid.GridGaussianModel import GridGaussianModel


def test_active_positions_are_projected_into_half_open_cell(tmp_path):
    model = make_gaussian_model(torch.tensor([[0.25, 0.0, 0.0]]))
    grid_model = GridGaussianModel.from_gaussian_model(
        model,
        cameras=[],
        grid=Grid(grid_size=1.0),
        model_store_device="cpu",
        model_train_device="cpu",
        min_gaussians=0,
        cache_dir=str(tmp_path),
        cache_fingerprint="fixture",
    )
    grid_model.grid_set_active_cell_index(GridIndex(0, 0, 0))
    with torch.no_grad():
        grid_model.positions.copy_(torch.tensor([[-2.0, 2.0, 0.5]]))

    grid_model.constrain_positions()

    position = grid_model.positions[0]
    assert torch.all(position >= 0)
    assert torch.all(position < 1)
