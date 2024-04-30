from typing import List, Tuple
import numpy as np
import torch
from gs.core.View import View, ViewWithRes
from gs.geometry.raycast_sphere import sphere_ray_intersection

# Stub for SphereImagesMLP.

class SphereImagesMLP(torch.nn.Module):

    def __init__(self, radii=[100,150,200], origin=np.array([0, 0, 0]), width=256, hidden_depth=6):
        super().__init__()  # Initialize the base class
        self.radius = radii
        self.origin = torch.tensor(origin, dtype=torch.float32)  # Ensure origin is a torch tensor
        self.width = width
        # Use a 3D MLP to map from unit spherical coordinates + radii to RGBD
        layers = [
            torch.nn.Linear(4, width),
            torch.nn.ReLU()
        ]
        layers.extend([
            torch.nn.Sequential(
                torch.nn.Linear(width, width),
                torch.nn.ReLU()
            ) for _ in range(hidden_depth)
        ])
        layers.extend([
            torch.nn.Linear(width, width // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(width // 2, 4)
        ])
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        # Subtract origin from rays x, y, z to get rays in sphere's local coordinate system
        local_rays = rays[:, :3] - self.origin
        spherical_positions = sphere_ray_intersection(local_rays, self.radius)
        # Mask for rays that did not intersect the sphere
        no_intersection_mask = torch.isnan(spherical_positions).any(dim=1)

        # Prepare default value for no intersection
        output_values = torch.zeros((rays.shape[0], 4), dtype=torch.float32, device=rays.device) # Create tensor of zeros of shape (num_rays, 4)
        
        # Compute outputs for intersecting rays, skip rays that did not intersect
        intersecting_rays = spherical_positions[~no_intersection_mask]
        if intersecting_rays.shape[0] > 0:
            output_values[~no_intersection_mask] = self.mlp(intersecting_rays)

        return output_values # Return tensor of shape (num_rays, 4)
    
    def forward_fixed_origin(self, ray_origin: torch.Tensor, ray_directions: torch.Tensor):
        raise NotImplementedError("This function is not implemented")

class MultiSphereImage(torch.nn.Module):
    def __init__(self, sphere_radii: List[float] = [torch.inf]):
        sphere_radii = sorted(sphere_radii, reverse=True) # Sort radii in descending order. The smallest spheres always render last
        self.spheres = [SphereImagesMLP(radius) for radius in sphere_radii]
    def forward(self, view: ViewWithRes) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: # (rgb, depth, alpha)
        # Calculate tensor of sampling points from view's projection matrix and resolution -> (num_layers, theta, phi)
        origin, directions = view.get_rays() # Origin: (3,), directions: (image_height, image_width, 3)
        # Flatten directions to (num_rays, 3)
        directions = directions.view(-1, 3)
        # Calculate RGBD for each sphere: (h, w, c: (r,g,b,a))
        output_buffer = torch.zeros((view.image_height, view.image_width, 4), dtype=torch.float32, device=directions.device)
        # Iteratively alpha blend the spheres
        for sphere in self.spheres: # No sorting, assumes smallest sphere always on top
            # Calculate RGBA for the sphere
            sphere_output = sphere(directions) # Output in shape (num_rays, 4)
            # Calculate alpha blending
            output_buffer = sphere_output[:, :3] * sphere_output[:, 3:] + output_buffer * (1 - sphere_output[:, 3:])
        # Reshape output buffer to (h, w, c)
        return output_buffer.view(view.image_height, view.image_width, 4)