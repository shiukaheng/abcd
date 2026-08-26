import torch


def fix_default_blended(
    value: torch.Tensor,
    alpha: torch.Tensor,
    default_value: float = 0.0,
    alpha_epsilon: float = 1e-10,
) -> torch.Tensor:
    """
    Adjusts the value based on alpha transparency.
    Transparent areas (alpha=0) will be set to a default value.
    """
    if default_value == 0.0:
        return torch.where(
            alpha > alpha_epsilon, value / alpha, torch.zeros_like(value)
        )
    else:
        return torch.where(
            alpha > alpha_epsilon,
            (value - default_value * (1 - alpha)) / alpha,
            torch.full_like(value, default_value),
        )
