import torch
from functools import reduce
from typing import List, Tuple

def correct_color_pre_blending(rgb: torch.Tensor, alpha: torch.Tensor):
    """
    Corrects the color of the RGB image to compensate for blending with a black background.
    This helps in reducing the dark edges due to the black background in areas of partial transparency.
    """
    # Avoid division by zero for fully transparent pixels by adding a small epsilon where alpha is zero.
    epsilon = 1e-10
    alpha_corrected = torch.where(alpha > 0, alpha, torch.full_like(alpha, epsilon))
    return rgb / alpha_corrected

def _composite_images_rgbda(back: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                            front: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb_back, depth_back, alpha_back = back
    rgb_front, depth_front, alpha_front = front
    
    # Correct the color of the front RGB image before blending
    rgb_front_corrected = correct_color_pre_blending(rgb_front, alpha_front)
    
    # Composite RGB using the corrected front RGB
    rgb = rgb_back * (1 - alpha_front) + rgb_front_corrected * alpha_front
    # Composite depth
    depth = depth_back * (1 - alpha_front) + depth_front * alpha_front
    # Composite alpha
    alpha = alpha_back + alpha_front * (1 - alpha_back)
    
    return rgb, depth, alpha

def composite_images_rgbda(images: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Composites a list of (RGB, depth, alpha) images into a single image. Assumes images are ordered from back to front.
    return reduce(_composite_images_rgbda, images)
