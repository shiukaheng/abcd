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
    # alpha_corrected = torch.where(alpha > 0, alpha, torch.full_like(alpha, epsilon))
    # return rgb / alpha_corrected
    return torch.where(alpha > epsilon, rgb / alpha, torch.zeros_like(rgb))

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

def fix_depth(depth: torch.Tensor, alpha: torch.Tensor, default_depth: float=0.0) -> torch.Tensor:
    """
    Adjusts the depth values based on alpha transparency.
    Transparent areas (alpha=0) will be set to a default depth value.
    """
    return torch.where(alpha > 0, depth, torch.full_like(depth, default_depth))

# def fix_rgb(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
#     """
#     Adjusts the RGB values for areas where the alpha is non-zero to compensate for blending with a black background.
#     This corrects colors to be more representative of their appearance without the influence of the background.
#     """
#     # Avoid division by zero for fully transparent pixels by adding a small epsilon where alpha is zero.
#     epsilon = 1e-10
#     alpha_corrected = torch.where(alpha > 0, alpha, torch.full_like(alpha, epsilon))
#     return rgb / alpha_corrected

# def fix_rgb(rgb: torch.Tensor, alpha: torch.Tensor, background_color: torch.Tensor) -> torch.Tensor:
#     """
#     Adjusts the RGB values for areas where the alpha is non-zero to compensate for blending with an arbitrary background color.
#     This corrects colors to be more representative of their appearance without the influence of the given background.
#     Parameters:
#     - rgb: Foreground RGB tensor.
#     - alpha: Alpha tensor corresponding to the RGB tensor.
#     - background_color: Tensor of the background color, same shape as one channel of rgb or broadcastable to that shape.
#     """
#     # Ensure background color is broadcastable to the shape of the RGB tensor
#     if background_color.ndim < rgb.ndim:
#         background_color = background_color.view(1, 1, -1).expand_as(rgb)

#     # Inverse of the alpha multiplication to remove background influence
#     epsilon = 1e-10  # Small constant to avoid division by zero
#     alpha_corrected = torch.where(alpha > 0, alpha, torch.full_like(alpha, epsilon))
    
#     # Adjust rgb values considering the background color
#     return (rgb - background_color * (1 - alpha)) / alpha_corrected
