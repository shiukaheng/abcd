from typing import TYPE_CHECKING, NamedTuple, Tuple

import torch

if TYPE_CHECKING:
    from abcd.geometry.grid import Grid


class BoundingBox(NamedTuple):
    min: torch.Tensor  # torch.Tensor([x, y, z])
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

    def __add__(self, offset: torch.Tensor) -> "BoundingBox":
        return BoundingBox(min=self.min + offset, max=self.max + offset)

    def __sub__(self, offset: torch.Tensor) -> "BoundingBox":
        return BoundingBox(min=self.min - offset, max=self.max - offset)

    def __mul__(self, scale: torch.Tensor) -> "BoundingBox":
        return BoundingBox(min=self.min * scale, max=self.max * scale)

    def __truediv__(self, scale: torch.Tensor) -> "BoundingBox":
        return BoundingBox(min=self.min / scale, max=self.max / scale)

    def __floordiv__(self, scale: torch.Tensor) -> "BoundingBox":
        return BoundingBox(min=self.min // scale, max=self.max // scale)

    def __mod__(self, scale: torch.Tensor) -> "BoundingBox":
        return BoundingBox(min=self.min % scale, max=self.max % scale)

    def __pow__(self, scale: torch.Tensor) -> "BoundingBox":
        return BoundingBox(min=self.min**scale, max=self.max**scale)

    def round_to_grid(self, grid: "Grid") -> "BoundingBox":
        """
        Rounds the bounding box to the grid.
        """
        shift = -grid.grid_origin
        scale = 1 / grid.grid_size

        transformed_bb = (self + shift) * scale
        min = torch.floor(transformed_bb.min)
        max = torch.ceil(transformed_bb.max)

        untransformed = BoundingBox(min / scale - shift, max / scale - shift)
        return untransformed

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

    def get_edges(
        self,
    ) -> Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
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

    @property
    def volume(self) -> float:
        """
        Returns the volume of the bounding box.
        """
        return torch.prod(self.max - self.min).item()

    @property
    def center(self) -> torch.Tensor:
        """
        Returns the center of the bounding box.
        """
        return (self.min + self.max) / 2

    def intersects(self, other: "BoundingBox") -> bool:
        """
        Returns whether the bounding box intersects with another bounding box.
        """
        return torch.all(self.min <= other.max) and torch.all(self.max >= other.min)
