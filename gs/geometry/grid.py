from typing import List, NamedTuple, Tuple

import torch

from gs.geometry.bounding_box import BoundingBox


class GridIndex(NamedTuple):
    """
    Represents a cell index in a grid.
    """

    x: int
    y: int
    z: int

    def to_string_id(self) -> str:
        """
        Converts the cell index to a string.
        """
        return f"{self.x}_{self.y}_{self.z}"


class Grid(NamedTuple):
    """
    Represents a grid.
    """

    grid_size: float = 1.0
    grid_origin: torch.Tensor = torch.tensor([0.0, 0.0, 0.0])

    def point_to_cell(self, point: torch.Tensor) -> GridIndex:
        """
        Finds the cell index of a point in the grid.
        """
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        coordinates = torch.floor(
            (point - self.grid_origin.to(point)) / self.grid_size
        ).to(torch.int64)
        return GridIndex(*(int(value) for value in coordinates.tolist()))

    def get_bounding_box(self, cell: GridIndex) -> BoundingBox:
        """
        Finds the bounding box of a cell.
        """
        cell_min = (
            torch.tensor([cell.x, cell.y, cell.z]) * self.grid_size + self.grid_origin
        )
        cell_max = cell_min + torch.tensor(
            [self.grid_size, self.grid_size, self.grid_size]
        )
        return BoundingBox(min=cell_min, max=cell_max)

    def get_cell_center(self, cell: GridIndex) -> torch.Tensor:
        """
        Finds the center of a cell.
        """
        return (
            torch.tensor([cell.x, cell.y, cell.z]) * self.grid_size
            + self.grid_origin
            + torch.tensor([self.grid_size / 2, self.grid_size / 2, self.grid_size / 2])
        )

    def calculate_adjacent_cells(self, cell: GridIndex) -> List[GridIndex]:
        """
        Finds the adjacent cells to a cell.
        """
        return [
            GridIndex(x=cell.x + dx, y=cell.y + dy, z=cell.z + dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if dx != 0 or dy != 0 or dz != 0
        ]

    def split_bounding_box(
        self, bounding_box: "BoundingBox"
    ) -> List[Tuple["BoundingBox", "GridIndex"]]:
        """
        Splits the bounding box into cells of the grid.
        """
        shift = -self.grid_origin.to(bounding_box.min.device)
        scale = 1 / self.grid_size

        transformed_bb = (bounding_box + shift) * scale
        min = torch.floor(transformed_bb.min)
        max = torch.ceil(transformed_bb.max)

        # Now, we can enumerate all bounding boxes in the grid
        cells = []
        for x in range(int(min[0]), int(max[0])):
            for y in range(int(min[1]), int(max[1])):
                for z in range(int(min[2]), int(max[2])):
                    cell_min = (
                        torch.tensor([x, y, z], device=bounding_box.min.device) / scale
                        - shift
                    )
                    cell_max = (
                        cell_min
                        + torch.tensor([1, 1, 1], device=bounding_box.min.device)
                        / scale
                    )
                    cells.append(
                        (BoundingBox(min=cell_min, max=cell_max), GridIndex(x, y, z))
                    )

        return cells
