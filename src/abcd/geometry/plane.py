from typing import NamedTuple

import torch

from abcd.geometry.bounding_box import BoundingBox


class Plane(NamedTuple):
    normal: torch.Tensor
    d: float

    def signed_distance(self, point: torch.Tensor) -> float:
        return torch.dot(self.normal, point) + self.d

    def intersects_box(self, box: "BoundingBox") -> bool:
        """
        Check if a bounding box intersects with the plane using the separating axis theorem.
        """
        # Get the positive and negative vertex with respect to the normal of the plane
        p_vertex = box.min.clone()
        n_vertex = box.max.clone()
        for i in range(3):  # Assuming 3D normals
            if self.normal[i] >= 0:
                p_vertex[i] = box.max[i]
                n_vertex[i] = box.min[i]

        # If the positive vertex is on or behind the plane, or the negative vertex is in front of the plane, they intersect
        return (
            self.signed_distance(p_vertex) >= 0 or self.signed_distance(n_vertex) <= 0
        )

    def __str__(self) -> str:
        return f"Plane(normal={self.normal}, d={self.d})"

    def shift_scalar(self, offset: float) -> "Plane":
        """
        Shifts the plane along its normal by a given offset.
        """
        return Plane(normal=self.normal, d=self.d + offset)
