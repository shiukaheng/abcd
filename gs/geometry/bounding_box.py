from typing import NamedTuple, Tuple
import torch

class BoundingBox(NamedTuple):
    min: torch.Tensor # torch.Tensor([x, y, z])
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
    
    def get_corners(self) -> Tuple[Tuple[float, float, float]]:
        """
        Returns the corners of the bounding box.
        """
        return (
            (self.min[0], self.min[1], self.min[2]),
            (self.min[0], self.min[1], self.max[2]),
            (self.min[0], self.max[1], self.min[2]),
            (self.min[0], self.max[1], self.max[2]),
            (self.max[0], self.min[1], self.min[2]),
            (self.max[0], self.min[1], self.max[2]),
            (self.max[0], self.max[1], self.min[2]),
            (self.max[0], self.max[1], self.max[2]),
        )
    
    def get_edges(self) -> Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Returns the edges of the bounding box.
        """
        corners = self.get_corners()
        return (
            (corners[0], corners[1]),
            (corners[0], corners[2]),
            (corners[0], corners[4]),
            (corners[1], corners[3]),
            (corners[1], corners[5]),
            (corners[2], corners[3]),
            (corners[2], corners[6]),
            (corners[3], corners[7]),
            (corners[4], corners[5]),
            (corners[4], corners[6]),
            (corners[5], corners[7]),
            (corners[6], corners[7]),
        )