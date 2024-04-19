
from dataclasses import dataclass
from typing import Generic, Literal, Tuple, Dict, List, TypeVar, Union

import torch

from gs.core.View import View, ViewWithRes, KnownView
from gs.core.GaussianModel import GaussianModel
from gs.geometry.bounding_box import BoundingBox
from gs.geometry.grid import Grid, GridIndex
from gs.compositing.alpha_compositing import composite_images_rgbda
from gs.trainers.grid.forward_properties import forward_to_active_cell
from gs.trainers.grid.grid_utils import cut, merge_model, split_model

T = TypeVar('T')

class GridGaussianCell(Generic[T]): # T represents the type of the camera ID
    index: GridIndex # Unique identifier for the cell
    model: GaussianModel # Gaussian model for the cell
    bounding_box: BoundingBox # Bounding box of the cell
    prerenders: Dict[int, Dict[T, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] # View snapshots by iteration, camera id, obtaining (RGB, depth, alpha) images
    current_iter: int = 0 # Current iteration of the cell
    grid: Grid # Grid of the cell

    def __init__(self, grid: Grid, index: GridIndex, model: GaussianModel, bounding_box: BoundingBox):
        self.grid = grid
        self.index = index
        self.model = model
        self.bounding_box = bounding_box
        self.prerenders = {}
        self.center = torch.mean(torch.stack([bounding_box.min, bounding_box.max]), dim=0)

    def plane_distance(self, camera: View) -> float:
        """
        Returns the distance between the camera to a plane defined by the cell's center and the camera's look direction.
        """
        return torch.dot(self.center - camera.center.to("cpu"), camera.look_at.to("cpu"))
    
    def get_prerender(self, camera: KnownView[T], iteration: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the prerender of the cell for a camera at a specific iteration, or the latest iteration if not specified.
        """
        if iteration is None:
            iteration = self.current_iter
        prerender = self.prerenders.get(iteration, {}).get(camera.id, None)
        if prerender is None:
            raise ValueError(f"Prerender for camera {camera.id} at iteration {iteration} is not available.")
        return prerender
    
    def clean_model_edges(self):
        """
        Removes Gaussians outside the bounding box of the cell.
        """
        self.model = cut(self.model, self.bounding_box)

def split_to_grid_gaussian_cells(model: GaussianModel, grid: Grid, min_gaussians: int) -> Dict[GridIndex, GaussianModel]:
    """
    Splits a Gaussian model into a grid of cells.
    """
    split = split_model(model, grid, min_gaussians)
    return {index: cell_model for index, (cell_model, bounding_box) in split.items()}

def index_cameras_by_id(cameras: List[KnownView[T]]) -> Dict[T, KnownView]:
    """
    Indexes cameras by their ID.
    """
    return {camera.id: camera for camera in cameras}

CompensationType = Literal["last", "uniform", "disabled"]



# Pseudo-nn.Module that allows differentiable forward pass through a grid of Gaussian models, while only a single cell is active at a time
@forward_to_active_cell()
class GridGaussianModel(Generic[T]): # T represents the type of the camera ID

    cameras: Dict[T, KnownView] # Cameras indexed by their ID
    cells: Dict[GridIndex, GridGaussianCell[T]]
    _active_cell_index: Union[GridIndex, None]
    model_store_device: str
    model_train_device: str
    default_extra_cell_compensation: CompensationType
    camera_to_grid_visibility: Dict[T, List[GridIndex]] # Visibility of cameras to cells
    grid_to_camera_visibility: Dict[GridIndex, List[T]] # Visibility of cells to cameras

    def __init__(
            self,
            models: Dict[GridIndex, GaussianModel],
            cameras: List[KnownView],
            grid: Grid = Grid(),
            model_store_device: str = "cpu",
            model_train_device: str = "cuda",
            default_extra_cell_compensation: CompensationType = "uniform"
    ):
        self.cells: Dict[GridIndex, GridGaussianCell[T]] = {index: GridGaussianCell(grid, index, cell_model, grid.get_bounding_box(index)) for index, cell_model in models.items()}
        self.cameras = index_cameras_by_id(cameras)
        self._active_cell_index = None
        # object.__setattr__(self, "_active_cell_index", None)
        self.model_store_device = model_store_device
        self.model_train_device = model_train_device
        self.default_extra_cell_compensation = default_extra_cell_compensation

        # Calculate visibility 
        self.calculate_visibility()

    def calculate_visibility(self):
        """
        Calculates the visibility of cameras to cells and cells to cameras.
        """
        self.camera_to_grid_visibility = {}
        self.grid_to_camera_visibility = {}
        for camera in self.cameras.values():
            self.camera_to_grid_visibility[camera.id] = [] # Initialize the visibility of the camera to cells
            for cell in self.cells.values():
                if camera.frustum.intersects_bounding_box(cell.bounding_box):
                    self.camera_to_grid_visibility[camera.id].append(cell.index)
                    if cell.index not in self.grid_to_camera_visibility:
                        self.grid_to_camera_visibility[cell.index] = []
                    self.grid_to_camera_visibility[cell.index].append(camera.id)

    @staticmethod
    def from_gaussian_model(
        input_model: GaussianModel,
        cameras: List[KnownView],
        grid: Grid = Grid(),
        model_store_device: str = "cpu",
        model_train_device: str = "cuda",
        default_extra_cell_compensation: CompensationType = "uniform",
        min_gaussians: int = 50
    ):
        """
        Create a GridGaussianModel from a single Gaussian model.
        """
        cells = split_to_grid_gaussian_cells(input_model, grid, min_gaussians=min_gaussians)
        return GridGaussianModel(cells, cameras, grid, model_store_device, model_train_device, default_extra_cell_compensation)

    def grid_get(self, index: GridIndex) -> GaussianModel:
        """
        Returns the Gaussian model of the cell at the given ID.
        """
        return self.cells[index].model
    
    def grid_len(self) -> int:
        """
        Returns the number of cells in the grid.
        """
        return len(self.cells)
    
    def grid_iter(self):
        """
        Returns an iterator over the GaussianGridCells in the grid.
        """
        return iter(self.cells.values())
    
    def merge(self, clean=True) -> GaussianModel:
        """
        Merge the grid of Gaussian models into a single Gaussian model.
        """
        return merge_model([(cell.model, cell.bounding_box) for cell in self.cells.values()], self.model_store_device, clean)
    
    def set_active_cell_index(self, index: GridIndex):
        """
        Set the active cell in which we want to update the parameters.
        """
        # Move the active cell to the training device, and the rest to the storage device
        for i, cell in self.cells.items():
            if i == index:
                cell.model.to(self.model_train_device)
            else:
                cell.model.to(self.model_store_device)
        self._active_cell_index = index

    def set_active_cell(self, cell: GridGaussianCell[T]):
        """
        Set the active cell in which we want to update the parameters.
        """
        self.set_active_cell_index(cell.index)

    @property
    def active_cell(self) -> GridGaussianCell[T]:
        """
        Returns the active cell.
        """
        if self._active_cell_index is None:
            raise ValueError("No active cell is set.")
        return self.cells[self._active_cell_index]
    
    @property
    def active_cell_index(self) -> GridIndex:
        """
        Returns the index of the active cell.
        """
        return self._active_cell_index

    def get_visible_cells_from_camera(self, camera_id: T) -> List[GridGaussianCell[T]]:
        """
        Returns a list of cells that a camera should render based on its frustum.
        """
        # Basic implementation: Precompute frustum / cell intersections
        return [self.cells[cell_index] for cell_index in self.camera_to_grid_visibility[camera_id]]
        # TODO: Advanced implementation: Take into account occlusion and visibility, updating the list of visible cells dynamically
    
    def get_visible_cameras_from_cell(self, cell_index: GridIndex) -> List[KnownView[T]]:
        """
        Returns a list of cameras that should render a cell based on its frustum.
        """
        # Basic implementation: Precompute frustum / cell intersections
        return [self.cameras[cam_id] for cam_id in self.grid_to_camera_visibility[cell_index]]
        # TODO: Advanced implementation: Take into account occlusion and visibility, updating the list of visible cameras dynamically

    def calculate_newest_common_view_snapshot_iteration(self) -> int:
        """
        Returns the best uniform iteration to request for extra cell compensation.
        """
        newest_iterations = set()
        for cell in self.cells.values():
            keys = cell.prerenders.keys()
            if len(keys) == 0: # If a cell has no view snapshots, we know there is no common view snapshot
                return -1
            newest_iterations.add(max(keys))

        if len(newest_iterations) > 0: # If there are view snapshots in all cells
            # Return the lowest
            # ASSUMPTION: If a cell has a newer view snapshot, it will ALWAYS have a view snapshot for all previous iterations
            return min(newest_iterations)
        else: # If there are no view snapshots in any cell. This should not happen, since we already checked for cells with no view snapshots.
            return -1
        
    def prerender_active_cell(self, current_iter: int):
        """
        Prerender the active cell where and save it as the specified iteration in the active cell's view snapshots.
        """
        if self._active_cell_index is None:
            raise ValueError("No active cell is set.")
        # For each camera that can see the active cell, we render the cell and save the view snapshot, save to the active cell's view snapshots
        with torch.no_grad():
            cameras = self.get_visible_cameras_from_cell(self._active_cell_index)
            for camera in cameras:
                camera.to(self.model_train_device) # Move the camera to the training device
                rgb, depth, alpha = self.active_cell.model.forward(camera)
                camera.to(self.model_store_device) # Move the camera back to the storage device (if it was not already there
                rgb, depth, alpha = rgb.detach().cpu(), depth.detach().cpu(), alpha.detach().cpu()
                if current_iter not in self.active_cell.prerenders:
                    self.active_cell.prerenders[current_iter] = {}
                self.active_cell.prerenders[current_iter][camera.id] = (rgb, depth, alpha)

    def cull_active_cell_prerenders(self, older_than: int):
        """
        Cull the view snapshots of the active cell that are older than the specified iteration.
        """
        if self._active_cell_index is None:
            raise ValueError("No active cell is set.")
        for iteration in list(self.active_cell.prerenders.keys()):
            if iteration < older_than:
                del self.active_cell.prerenders[iteration]
        
    def forward(
            self, 
            camera: Union[ViewWithRes, KnownView], 
            active_sh_degree: Union[int, None] = None, 
            extra_cell_compensation: Union[CompensationType, None] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the grid of Gaussian models, rendering the active cell and compositing the other cells within the frustum if `extra_cell_compensation` is not None.
        
        Args:
            camera: The camera to render the scene from.
            extra_cell_compensation: 
            If "last", we composite the appearance of other cells with their latest view snapshot. 
            If "uniform", we composite the appearance of other cells with a uniform distribution of view snapshots. 
            If "disabled", we only render the active cell.
            If None, we use the default value of the GridGaussianModel.
        """

        # If camera is not a KnownView, then we directly render the active cell
        if not isinstance(camera, KnownView):
            return self.active_cell.model.forward(camera, active_sh_degree)
        
        if extra_cell_compensation is None:
            extra_cell_compensation = self.default_extra_cell_compensation

        # First, we render the current active cell
        active_rgb, active_depth, active_alpha = self.active_cell.model.forward(camera, active_sh_degree)
        if extra_cell_compensation == "disabled": # Early return if we do not composite other cells
            return active_rgb, active_depth, active_alpha
        active_plane_distance = self.active_cell.plane_distance(camera)

        # Next, we garner all the other cells within the frustum
        in_view_cells = self.get_visible_cells_from_camera(camera.id)

        # We calculate which iteration to request for extra cell compensation
        if extra_cell_compensation == "uniform":
            requested_iteration = self.calculate_newest_common_view_snapshot_iteration()
        elif extra_cell_compensation == "last":
            requested_iteration = None
        else:
            raise ValueError(f"Invalid value for extra_cell_compensation: {extra_cell_compensation}")

        # If there are no common view snapshots, we directly return the appearance of the active cell
        if requested_iteration == -1:
            return active_rgb, active_depth, active_alpha

        # Now, this is the actual case where we composite the appearance of other cells together!

        # We gather all the other layers to composite as ((RGB, depth, alpha), plane_distance) tuples
        prerendered_layers = [(cell.get_prerender(camera, requested_iteration), cell.plane_distance(camera)) for cell in in_view_cells if cell.current_iter != 0 and cell.index != self._active_cell_index]
        # Move all prerendered layers to the training device
        prerendered_layers = [((rgb.to(self.model_train_device), depth.to(self.model_train_device), alpha.to(self.model_train_device)), plane_distance) for ((rgb, depth, alpha), plane_distance) in prerendered_layers]
        # We add the active cell to the layers
        layers = prerendered_layers + [((active_rgb, active_depth, active_alpha), active_plane_distance)]
        # We sort the layers by plane distance
        layers.sort(key=lambda x: -x[1])
        # We remove the plane distance from the layers
        layers = [layer[0] for layer in layers]
        # We composite the layers
        composite = composite_images_rgbda(layers)
        return composite
    
    def __call__(self, camera: ViewWithRes, extra_cell_compensation: Union[CompensationType, None] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(camera, extra_cell_compensation)
    
    def to(self, device: str):
        # This behaviour honestly is not even needed in our use, since calling ".set_active_cell" will move the active cell to the device
        # We only move the active cell to the device
        self.active_cell.model.to(device)
        # We set the device for self.model_train_device such that the next active cell will be moved to the correct device
        self.model_train_device = device