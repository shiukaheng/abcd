from functools import reduce
from typing import List, Tuple

import torch

from abcd.compositing.gaussian_rendering_fix import fix_default_blended


def composite_images_rgbda(
    images: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Composites a list of (RGB, depth, alpha) images into a single image.

    Args:
        images (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): A list of tuples containing RGB, depth, and alpha images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the composite RGB, depth, and alpha images.
    """
    if not images:
        raise ValueError("At least one image is required for compositing")
    return reduce(_composite_images_rgbda, images)


def _composite_images_rgbda(
    back: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    front: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb_back, depth_back, alpha_back = back
    rgb_front, depth_front, alpha_front = front

    # Correct the color of the front RGB image before blending
    rgb_front_corrected = fix_default_blended(rgb_front, alpha_front)

    # Composite RGB using the corrected front RGB
    rgb = rgb_back * (1 - alpha_front) + rgb_front_corrected * alpha_front
    # Composite depth
    depth = depth_back * (1 - alpha_front) + depth_front * alpha_front
    # Composite alpha
    alpha = alpha_back + alpha_front * (1 - alpha_back)

    return rgb, depth, alpha
