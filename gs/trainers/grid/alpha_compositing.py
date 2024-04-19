import torch
from functools import reduce
from typing import List, Tuple

def _composite_images_rgbda(back: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], front: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Composites two (RGB, depth, alpha) images into a single image.
    rgb_back, depth_back, alpha_back = back
    rgb_front, depth_front, alpha_front = front
    # Composite RGB
    rgb = rgb_back * (1 - alpha_front) + rgb_front * alpha_front
    # Composite depth
    depth = depth_back * (1 - alpha_front) + depth_front * alpha_front
    # Composite alpha
    alpha = alpha_back + alpha_front * (1 - alpha_back)
    return rgb, depth, alpha

def composite_images_rgbda(images: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Composites a list of (RGB, depth, alpha) images into a single image. Assumes images are ordered from back to front.
    return reduce(_composite_images_rgbda, images)