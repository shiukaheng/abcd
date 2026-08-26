import torch
from conftest import make_gaussian_model

from gs.geometry.grid import Grid, GridIndex
from gs.trainers.grid.GridGaussianModel import GridGaussianModel


def test_only_active_shard_is_materialized(tmp_path):
    model = make_gaussian_model(torch.tensor([[0.25, 0.0, 0.0], [1.25, 0.0, 0.0]]))
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

    assert all(cell.model is None for cell in grid_model.grid_iter())

    grid_model.grid_set_active_cell_index(GridIndex(0, 0, 0))
    assert grid_model.grid_cells[GridIndex(0, 0, 0)].model is not None
    assert grid_model.grid_cells[GridIndex(1, 0, 0)].model is None

    grid_model.grid_set_active_cell_index(GridIndex(1, 0, 0))
    assert grid_model.grid_cells[GridIndex(0, 0, 0)].model is None
    assert grid_model.grid_cells[GridIndex(1, 0, 0)].model is not None


def test_resume_rebuilds_grid_without_materializing_shards(tmp_path):
    model = make_gaussian_model(torch.tensor([[0.25, 0.0, 0.0], [1.25, 0.0, 0.0]]))
    grid = Grid(grid_size=1.0)
    grid_model = GridGaussianModel.from_gaussian_model(
        model,
        cameras=[],
        grid=grid,
        model_store_device="cpu",
        model_train_device="cpu",
        min_gaussians=0,
        cache_dir=str(tmp_path),
        cache_fingerprint="fixture",
    )
    grid_model.grid_set_active_cell_index(GridIndex(0, 0, 0))
    grid_model.grid_active_cell.current_iter = 5
    grid_model.grid_set_active_cell_index(GridIndex(1, 0, 0))

    resumed = GridGaussianModel.from_gaussian_model(
        model,
        cameras=[],
        grid=grid,
        model_store_device="cpu",
        model_train_device="cpu",
        cache_dir=str(tmp_path),
        cache_fingerprint="fixture",
        resume=True,
    )

    assert all(cell.model is None for cell in resumed.grid_iter())
    assert resumed.grid_cells[GridIndex(0, 0, 0)].current_iter == 5
    resumed.grid_set_active_cell_index(GridIndex(0, 0, 0))
    assert resumed.grid_active_cell.model is not None
