from typing import List, Tuple
import torch
from tqdm import tqdm

from gs.geometry.bounding_box import BoundingBox
from typing import NamedTuple
    
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
    
    # Make sortable
    def __lt__(self, other):
        return self.to_string_id() < other.to_string_id()
    
    def __eq__(self, other):
        return self.to_string_id() == other.to_string_id()


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
        return GridIndex(
            x=int((point[0] - self.grid_origin[0]) / self.grid_size),
            y=int((point[1] - self.grid_origin[1]) / self.grid_size),
            z=int((point[2] - self.grid_origin[2]) / self.grid_size)
        )
    
    def get_bounding_box(self, cell: GridIndex) -> BoundingBox:
        """
        Finds the bounding box of a cell.
        """
        cell_min = torch.tensor([cell.x, cell.y, cell.z]) * self.grid_size + self.grid_origin
        cell_max = cell_min + torch.tensor([self.grid_size, self.grid_size, self.grid_size])
        return BoundingBox(min=cell_min, max=cell_max)
    
    def get_cell_center(self, cell: GridIndex) -> torch.Tensor:
        """
        Finds the center of a cell.
        """
        return torch.tensor([cell.x, cell.y, cell.z]) * self.grid_size + self.grid_origin + torch.tensor([self.grid_size / 2, self.grid_size / 2, self.grid_size / 2])
    
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
    
    def calculate_bounding_box_cells(self, bounding_box: BoundingBox) -> List[Tuple[BoundingBox, GridIndex]]:
        """
        Calculates the cells that are inside a bounding box.
        """
        min_grid = self.point_to_cell(bounding_box.min)
        max_grid = self.point_to_cell(bounding_box.max)
        
        total_x = max_grid.x - min_grid.x + 1
        total_y = max_grid.y - min_grid.y + 1
        total_z = max_grid.z - min_grid.z + 1
        total_cells = total_x * total_y * total_z
        
        cells = []
        
        for index in tqdm(range(total_cells), desc="Calculating cells"):
            x = index // (total_y * total_z) + min_grid.x
            y = (index % (total_y * total_z)) // total_z + min_grid.y
            z = (index % (total_y * total_z)) % total_z + min_grid.z
            
            cell_min = torch.tensor([x, y, z]) * self.grid_size + self.grid_origin
            cell_max = cell_min + torch.tensor([self.grid_size, self.grid_size, self.grid_size])
            
            cells.append((BoundingBox(min=cell_min, max=cell_max), GridIndex(x=x, y=y, z=z)))
        
        return cells