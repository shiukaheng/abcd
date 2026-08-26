import pytest
import torch
from conftest import make_gaussian_model

from gs.core.GaussianModel import GaussianModel


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_ply_round_trip_preserves_sh_degree_and_values(tmp_path, degree):
    model = make_gaussian_model(
        torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), sh_degree=degree
    )
    path = tmp_path / f"degree-{degree}.ply"

    model.save_ply(str(path))
    loaded = GaussianModel.from_ply(str(path), device="cpu")

    assert loaded.sh_degree == degree
    torch.testing.assert_close(loaded.positions, model.positions)
    torch.testing.assert_close(loaded.sh_coefficients, model.sh_coefficients)
    torch.testing.assert_close(loaded.rotations, model.rotations)
    torch.testing.assert_close(loaded.scales, model.scales)
    torch.testing.assert_close(loaded.opacities, model.opacities)
