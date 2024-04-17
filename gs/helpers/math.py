from typing import Callable, Tuple
import torch

"""
This module contains helper functions for mathematical operations.
"""

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def create_scaled_sigmoid(min: float, max: float) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """
    Creates scaled sigmoid / inverse sigmoid functions that maps the range [0, 1] to [min, max] and vice versa.
    """
    return lambda x: 1 / (1 + torch.exp(-x)) * (max - min) + min, lambda x: inverse_sigmoid((x - min) / (max - min))