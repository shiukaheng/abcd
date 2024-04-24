from typing import Callable, Tuple
import torch

"""
This module contains helper functions for mathematical operations.
"""

def inverse_sigmoid(x):
    return torch.log(x/(1-x)) # Note: If x=0, this will return -inf. If x=1, this will return nan.

def create_scaled_sigmoid(min: float, max: float, safety_factor=0.001) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """
    Creates scaled sigmoid / inverse sigmoid functions that maps the range [0, 1] to [min, max] and vice versa.
    """
    # return lambda x: 1 / (1 + torch.exp(-x)) * (max - min) + min, lambda x: inverse_sigmoid((x - min) / (max - min))
    # Use more optimized regular torch.sigmoid instead of custom sigmoid
    # return lambda x: torch.sigmoid(x) * (max - min) + min, lambda x: inverse_sigmoid((x - min) / (max - min))
    # Clamp using safety_factor after applying signoid such that output can never actually reach min or max so that inverse_sigmoid is always defined
    return (
        lambda x: torch.sigmoid(x) * (max - min) + min,
        lambda x: inverse_sigmoid(torch.clamp((x - min) / (max - min), safety_factor, 1 - safety_factor))
    )