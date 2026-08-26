import torch

from abcd.core.GaussianModel import GaussianModel


def make_gaussian_model(positions: torch.Tensor, sh_degree: int = 0) -> GaussianModel:
    positions = positions.to(dtype=torch.float32)
    count = positions.shape[0]
    coefficients = (sh_degree + 1) ** 2
    return GaussianModel(
        positions=positions,
        sh_coefficients=torch.arange(
            count * coefficients * 3, dtype=torch.float32
        ).reshape(count, coefficients, 3),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        scales=torch.zeros((count, 3)),
        opacities=torch.zeros((count, 1)),
        sh_degree=sh_degree,
    )
