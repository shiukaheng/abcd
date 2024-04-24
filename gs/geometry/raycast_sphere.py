from typing import Tuple
import torch

# N rays, M spheres

def sphere_ray_intersection_fixed_origin(
        origin: torch.Tensor, # (3,) tensor
        directions: torch.Tensor, # (N, 3) tensor
        sphere_origin: torch.Tensor, # (3,) tensor
        sphere_radii: torch.Tensor # (M, 3) tensor - Allow for simultaneous intersection with multiple spheres
    ) -> torch.Tensor: # Returns (N, M, 3) tensor
    """
    Given a multiple rays and a multi-layered sphere, compute the intersection points of the rays with the spheres.
    Outputs (N, M, 3), for N rays on M sphere in (x, y, z) coordinates.
    """

    assert origin.shape == (3,)
    assert directions.shape[1] == 3
    assert sphere_origin.shape == (3,)
    assert sphere_radii.shape[1] == 3

    # Transform rays to the context of a unit sphere centered at the origin
    p, d = transform_rays_to_unit_sphere_fixed_origin(origin, directions, sphere_origin, sphere_radii) # unit_pos: (M, 3,) tensor, unit_dir: (N, 3) tensor

    # Compute intersections with the unit sphere
    unit_intersections = unit_sphere_ray_intersection_fixed_origin(p, d) # (N, M, 3) tensor

    assert unit_intersections.shape[2] == 3
    
    return unit_intersections

def transform_rays_to_unit_sphere_fixed_origin(
        origin: torch.Tensor, # (3,) tensor
        directions: torch.Tensor, # (N, 3) tensor
        sphere_origin: torch.Tensor, # (3,) tensor
        sphere_radii: torch.Tensor # (M, 3) tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: # Returns (M, 3,) tensor, (N, 3) tensor
    """
    Given a set of rays and a set of spheres, 
    transform rays to the context of a unit sphere centered at the origin.
    """
    raise NotImplementedError("This function is not implemented")

def unit_sphere_ray_intersection_fixed_origin(
        origin: torch.Tensor, # (M, 3,) tensor. 
        # We have M spheres, and 1 ray origin. If we scale the rays to use a unit sphere,
        # we instead have M ray origins, each corresponding to a sphere.
        directions: torch.Tensor # (N, 3) tensor
    ) -> torch.Tensor: # Returns (N, M, 3) tensor
    """
    Given a set of rays and a unit sphere centered at the origin,
    compute the intersection points of the rays with the unit sphere.
    """
    raise NotImplementedError("This function is not implemented")
    # At this point.. Maybe just use regular unit_sphere_ray_intersection without fixed origin.