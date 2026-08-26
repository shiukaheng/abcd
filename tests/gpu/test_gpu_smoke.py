import gc

import numpy as np
import pytest
import torch

from gs.core.GaussianModel import GaussianModel
from gs.core.View import ViewWithRes
from gs.geometry.grid import Grid, GridIndex
from gs.trainers.grid.GridGaussianModel import GridGaussianModel

pytestmark = pytest.mark.gpu


def make_model(positions: torch.Tensor) -> GaussianModel:
    count = positions.shape[0]
    model = GaussianModel(
        positions=positions,
        sh_coefficients=torch.zeros((count, 16, 3), device=positions.device),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=positions.device).repeat(
            count, 1
        ),
        scales=torch.full((count, 3), -3.0, device=positions.device),
        opacities=torch.full((count, 1), 2.0, device=positions.device),
        sh_degree=3,
    )
    return model.to(positions.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_rasterizer_forward_and_backward():
    generator = torch.Generator(device="cuda").manual_seed(0)
    positions = torch.rand((4096, 3), device="cuda", generator=generator)
    positions = positions * torch.tensor([20.0, 10.0, 16.0], device="cuda")
    positions = positions + torch.tensor([-10.0, -2.0, -8.0], device="cuda")
    model = make_model(positions)
    camera = ViewWithRes(
        R=np.array(
            [
                [0.9999338, -0.0106386, -0.0043777],
                [0.0089077, 0.9568137, -0.2905653],
                [0.0072798, 0.2905071, 0.9568451],
            ]
        ),
        t=np.array([-0.3283028, -1.9259563, 3.9580584]),
        fov_x=1.1868536,
        fov_y=0.8226821,
        image_height=840,
        image_width=1297,
    ).to("cuda")

    rgb, depth, alpha = model.forward(camera)
    assert rgb.shape == (3, 840, 1297)
    assert depth.shape == (1, 840, 1297)
    assert alpha.shape == (1, 840, 1297)
    assert alpha.sum() > 0
    rgb.sum().backward()
    assert model.positions.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_active_parameter_vram_is_independent_of_inactive_shards(tmp_path):
    allocated = []
    gaussians_per_cell = 128
    for cell_count in [1, 2, 4, 8]:
        positions = []
        for cell in range(cell_count):
            cell_positions = torch.zeros((gaussians_per_cell, 3))
            cell_positions[:, 0] = cell + 0.25
            positions.append(cell_positions)
        model = make_model(torch.cat(positions))
        grid_model = GridGaussianModel.from_gaussian_model(
            model,
            cameras=[],
            grid=Grid(grid_size=1.0),
            model_store_device="cpu",
            model_train_device="cuda",
            min_gaussians=0,
            cache_dir=str(tmp_path / str(cell_count)),
            cache_fingerprint=f"cells-{cell_count}",
        )
        grid_model.grid_set_active_cell_index(GridIndex(0, 0, 0))
        allocated.append(torch.cuda.memory_allocated())
        del grid_model, model
        gc.collect()
        torch.cuda.empty_cache()

    assert max(allocated) - min(allocated) < 1024 * 1024
