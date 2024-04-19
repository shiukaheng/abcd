from typing import NamedTuple
import torch

class BoundingBox(NamedTuple):
    min: torch.Tensor
    max: torch.Tensor

    def get_opposing_vertices(self, normal: torch.Tensor) -> tuple:
        """
        Returns the most opposing vertices of the bounding box with respect to a plane's normal.
        """
        p_vertex = self.min.clone()
        n_vertex = self.max.clone()
        for i in range(3):  # Assuming 3D normals
            if normal[i] >= 0:
                p_vertex[i] = self.max[i]
                n_vertex[i] = self.min[i]
        return (p_vertex, n_vertex)

    def __contains__(self, point: torch.Tensor) -> bool:
        return torch.all(point >= self.min) and torch.all(point <= self.max)

    def __str__(self) -> str:
        return f"BoundingBox(min={self.min}, max={self.max})"

    def __repr__(self) -> str:
        return str(self)

    def __add__(self, offset: torch.Tensor) -> 'BoundingBox':
        return BoundingBox(min=self.min + offset, max=self.max + offset)

    def __sub__(self, offset: torch.Tensor) -> 'BoundingBox':
        return BoundingBox(min=self.min - offset, max=self.max - offset)
