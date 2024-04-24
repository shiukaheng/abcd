import torch

def unit_sphere_ray_intersection(rays):
    # Assuming rays is of shape [N, 6] where each row is [x, y, z, dx, dy, dz]
    p = rays[:, :3]  # Start points of the rays
    d = rays[:, 3:]  # Direction vectors of the rays

    # Normalize the direction vectors
    d_norm = torch.norm(d, dim=1, keepdim=True)
    d = d / d_norm

    # Calculate components for the quadratic formula
    p_dot_d = torch.sum(p * d, dim=1)
    p_dot_p = torch.sum(p * p, dim=1)
    
    # Discriminant for intersection calculation
    discriminant = p_dot_d**2 - (p_dot_p - 1)

    # Initialize the result tensor for intersection points
    intersections = torch.full_like(p, float('nan'))

    # Check where the discriminant is non-negative (i.e., there are real roots)
    real_roots = discriminant >= 0

    # Calculate the smallest positive t for those rays with real roots
    sqrt_disc = torch.sqrt(discriminant[real_roots])
    t1 = -p_dot_d[real_roots] - sqrt_disc
    t2 = -p_dot_d[real_roots] + sqrt_disc

    # Choose the smallest positive t
    t = torch.where(t1 > 0, t1, t2)
    t = torch.where((t2 > 0) & (t2 < t), t2, t)

    # Calculate intersection points for valid t values
    intersections[real_roots] = p[real_roots] + t.unsqueeze(1) * d[real_roots]

    return intersections

def unit_sphere_ray_intersection_fixed_origin(origin, directions):
    # origin: Tensor of shape [3] - the starting point of all rays
    # directions: Tensor of shape [N, 3] - the direction vectors of the rays
    
    # Normalize the direction vectors
    d_norm = torch.norm(directions, dim=1, keepdim=True)
    directions_normalized = directions / d_norm

    # Calculate components for the quadratic formula using broadcasting
    p_dot_d = torch.sum(origin * directions_normalized, dim=1)
    p_dot_p = torch.sum(origin * origin)
    
    # Discriminant for intersection calculation
    discriminant = p_dot_d**2 - (p_dot_p - 1)

    # Initialize the result tensor for intersection points
    intersections = torch.full_like(directions, float('nan'))

    # Check where the discriminant is non-negative (i.e., there are real roots)
    real_roots = discriminant >= 0

    # Calculate the smallest positive t for those rays with real roots
    sqrt_disc = torch.sqrt(discriminant[real_roots])
    t1 = -p_dot_d[real_roots] - sqrt_disc
    t2 = -p_dot_d[real_roots] + sqrt_disc

    # Choose the smallest positive t
    t = torch.where(t1 > 0, t1, t2)
    t = torch.where((t2 > 0) & (t2 < t), t2, t)

    # Calculate intersection points for valid t values
    intersections[real_roots] = origin + t.unsqueeze(1) * directions_normalized[real_roots]

    return intersections

def transform_rays_to_unit_sphere(rays, sphere_origin, radius):
    # Translate rays to be relative to sphere origin
    p = (rays[:, :3] - sphere_origin) / radius
    d = rays[:, 3:]
    
    # Create new rays array
    transformed_rays = torch.cat([p, d], dim=1)
    return transformed_rays

def transform_rays_to_unit_sphere_fixed_origin(origin, directions, sphere_origin, radius):
    # Translate rays to be relative to sphere origin
    p = (origin - sphere_origin) / radius
    d = directions
    
    # Return new origin and directions
    return p, d

def transform_intersection_from_unit_sphere(intersections, sphere_origin, radius):
    # Scale intersections back to original sphere size and translate back
    return intersections * radius + sphere_origin

def sphere_ray_intersection(rays, sphere_origin, sphere_radius, unit_sphere_output=True):
    # Transform rays to the context of a unit sphere centered at the origin
    transformed_rays = transform_rays_to_unit_sphere(rays, sphere_origin, sphere_radius)
    
    # Compute intersections with the unit sphere
    unit_intersections = unit_sphere_ray_intersection(transformed_rays)
    
    if unit_sphere_output:
        return unit_intersections
    
    # Transform intersections back to the original sphere's scale and position
    intersections = transform_intersection_from_unit_sphere(unit_intersections, sphere_origin, sphere_radius)
    
    return intersections

def sphere_ray_intersection_fixed_origin(origin, directions, sphere_origin, sphere_radius, unit_sphere_output=True):
    # Transform rays to the context of a unit sphere centered at the origin
    p, d = transform_rays_to_unit_sphere_fixed_origin(origin, directions, sphere_origin, sphere_radius)
    
    # Compute intersections with the unit sphere
    unit_intersections = unit_sphere_ray_intersection_fixed_origin(p, d)
    
    if unit_sphere_output:
        return unit_intersections
    
    # Transform intersections back to the original sphere's scale and position
    intersections = transform_intersection_from_unit_sphere(unit_intersections, sphere_origin, sphere_radius)
    
    return intersections

if __name__ == "__main__":
    # Test the sphere_ray_intersection function
    rays = torch.tensor([[0.0, 0.0, -2.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, -2.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, -2.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, -2.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    
    sphere_origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    sphere_radius = 1.0
    
    intersections = sphere_ray_intersection(rays, sphere_origin, sphere_radius, unit_sphere_output=False)
    print(intersections)