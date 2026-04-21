from dataclasses import dataclass
from typing import Generic, Literal, Tuple, Dict, List, TypeVar, Union, Optional

import torch

from gs.core.View import View, ViewWithRes, KnownView
from gs.core.GaussianModel import GaussianModel
from gs.geometry.bounding_box import BoundingBox
from gs.geometry.grid import Grid, GridIndex
from gs.compositing.alpha_compositing import composite_images_rgbda
from gs.trainers.grid.forward_properties import forward_to_active_cell
from gs.trainers.grid.grid_utils import cut, merge_model, split_model

T = TypeVar("T")


class GridGaussianCell(Generic[T]):
    index: GridIndex
    model: GaussianModel
    bounding_box: BoundingBox
    prerenders: Dict[int, Dict[T, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
    current_iter: int = 0
    grid: Grid

    def __init__(
        self,
        grid: Grid,
        index: GridIndex,
        model: GaussianModel,
        bounding_box: BoundingBox,
    ):
        self.grid = grid
        self.index = index
        self.model = model
        self.bounding_box = bounding_box
        self.prerenders = {}
        self.center = torch.mean(
            torch.stack([bounding_box.min, bounding_box.max]), dim=0
        )

    def plane_distance(self, camera: View) -> float:
        return torch.dot(
            self.center.to("cpu") - camera.center.to("cpu"), camera.look_at.to("cpu")
        )

    def distance(self, camera: View) -> float:
        return torch.norm(self.center.to("cpu") - camera.center.to("cpu"))

    def get_prerender(
        self, camera: KnownView[T], iteration: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if iteration is None:
            iteration = self.current_iter
        prerender = self.prerenders.get(iteration, {}).get(camera.id, None)
        if prerender is None:
            raise ValueError(
                f"Prerender for camera {camera.id} at iteration {iteration} is not available."
            )
        rgb, depth, alpha = prerender
        rgb, depth, alpha = (
            rgb.to(torch.float32) / 255,
            depth.to(torch.float32),
            alpha.to(torch.float32) / 255,
        )
        return rgb, depth, alpha

    def clean_model_edges(self):
        self.model = cut(self.model, self.bounding_box)


def split_to_grid_gaussian_cells(
    model: GaussianModel, grid: Grid, min_gaussians: int
) -> Dict[GridIndex, GaussianModel]:
    split = split_model(model, grid, min_gaussians)
    return {index: cell_model for index, (cell_model, bounding_box) in split.items()}


def index_cameras_by_id(cameras: List[KnownView[T]]) -> Dict[T, KnownView]:
    return {camera.id: camera for camera in cameras}


CompensationType = Literal["last", "uniform", "disabled"]
PrecompositeStorage = Literal["gpu", "cpu"]


@forward_to_active_cell()
class GridGaussianModel(Generic[T]):
    """
    A grid of Gaussian models, where each cell is a Gaussian model.
    It pretends to be a single Gaussian model enough to be trained by the basic training function.
    The grid is split into cells, and only one cell is active at a time and loaded into memory, which makes it possible to train large models on limited memory.
    During training, the different cells are composited together to render the scene.
    The actual training logic can be found in `gs/trainers/grid/train.py`.

    Some methods / properties of `GaussianModel` that we implement can be found in `gs/trainers/grid/forward_properties.py` and are added by the class decorator.
    Other methods from `GaussianModel` are mirrored explicitly in this class, such as `forward` and `to`.

    All methods and properties relating to the grid training logic is specifically prefixed with `grid_` to avoid confusion with the methods of `GaussianModel`.
    """

    grid_cameras: Dict[T, KnownView]
    grid_cells: Dict[GridIndex, GridGaussianCell[T]]
    _active_cell_index: Union[GridIndex, None]
    grid_model_store_device: str
    grid_model_train_device: str
    grid_default_extra_cell_compensation: CompensationType
    camera_to_grid_visibility: Dict[T, List[GridIndex]]
    grid_to_camera_visibility: Dict[GridIndex, List[T]]

    _precomposited_bg: Dict[T, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    _precomposited_fg: Dict[T, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    _precomposite_storage_device: str
    _precomposite_enabled: bool

    def __init__(
        self,
        models: Dict[GridIndex, GaussianModel],
        cameras: List[KnownView],
        grid: Grid = Grid(),
        model_store_device: str = "cpu",
        model_train_device: str = "cuda",
        default_extra_cell_compensation: CompensationType = "uniform",
        precomposite_enabled: bool = True,
        precomposite_storage: PrecompositeStorage = "gpu",
    ):
        self.grid_cells: Dict[GridIndex, GridGaussianCell[T]] = {
            index: GridGaussianCell(
                grid, index, cell_model, grid.get_bounding_box(index)
            )
            for index, cell_model in models.items()
        }
        self.grid_cameras = index_cameras_by_id(cameras)
        self._active_cell_index = None
        self.grid_model_store_device = model_store_device
        self.grid_model_train_device = model_train_device
        self.grid_default_extra_cell_compensation = default_extra_cell_compensation

        self._precomposited_bg = {}
        self._precomposited_fg = {}
        self._precomposite_storage_device = precomposite_storage
        self._precomposite_enabled = precomposite_enabled

        self.grid_calculate_visibility()

    def grid_calculate_visibility(self):
        self.camera_to_grid_visibility = {}
        self.grid_to_camera_visibility = {}
        for camera in self.grid_cameras.values():
            self.camera_to_grid_visibility[camera.id] = []
            for cell in self.grid_cells.values():
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
        min_gaussians: int = 50,
        precomposite_enabled: bool = True,
        precomposite_storage: PrecompositeStorage = "gpu",
    ):
        cells = split_to_grid_gaussian_cells(
            input_model, grid, min_gaussians=min_gaussians
        )
        print(f"Split model into {len(cells)} cells")
        return GridGaussianModel(
            cells,
            cameras,
            grid,
            model_store_device,
            model_train_device,
            default_extra_cell_compensation,
            precomposite_enabled,
            precomposite_storage,
        )

    def grid_get(self, index: GridIndex) -> GaussianModel:
        return self.grid_cells[index].model

    def grid_len(self) -> int:
        return len(self.grid_cells)

    def grid_iter(self):
        return iter(self.grid_cells.values())

    def grid_merge(self, clean=True) -> GaussianModel:
        return merge_model(
            [(cell.model, cell.bounding_box) for cell in self.grid_cells.values()],
            self.grid_model_store_device,
            clean,
        )

    def grid_set_active_cell_index(self, index: GridIndex):
        for i, cell in self.grid_cells.items():
            if i == index:
                cell.model.to(self.grid_model_train_device)
            else:
                cell.model.to(self.grid_model_store_device)
        self._active_cell_index = index

    def grid_set_active_cell(self, cell: GridGaussianCell[T]):
        self.grid_set_active_cell_index(cell.index)

    @property
    def grid_active_cell(self) -> GridGaussianCell[T]:
        if self._active_cell_index is None:
            raise ValueError("No active cell is set.")
        return self.grid_cells[self._active_cell_index]

    @property
    def grid_active_cell_index(self) -> GridIndex:
        return self._active_cell_index

    def grid_get_visible_cells_from_camera(
        self, camera_id: T
    ) -> List[GridGaussianCell[T]]:
        return [
            self.grid_cells[cell_index]
            for cell_index in self.camera_to_grid_visibility[camera_id]
        ]

    def grid_get_visible_cameras_from_cell(
        self, cell_index: GridIndex
    ) -> List[KnownView[T]]:
        return [
            self.grid_cameras[cam_id]
            for cam_id in self.grid_to_camera_visibility[cell_index]
        ]

    def grid_calculate_newest_common_view_snapshot_iteration(self) -> int:
        newest_iterations = set()
        for cell in self.grid_cells.values():
            keys = cell.prerenders.keys()
            if len(keys) == 0:
                return -1
            newest_iterations.add(max(keys))

        if len(newest_iterations) > 0:
            return min(newest_iterations)
        else:
            return -1

    def grid_prerender_active_cell(self, current_iter: int):
        if self._active_cell_index is None:
            raise ValueError("No active cell is set.")
        with torch.no_grad():
            cameras = self.grid_get_visible_cameras_from_cell(self._active_cell_index)
            for camera in cameras:
                camera.to(self.grid_model_train_device)
                rgb, depth, alpha = self.grid_active_cell.model.forward(camera)
                camera.to(self.grid_model_store_device)
                rgb, depth, alpha = (
                    (torch.clamp(rgb, 0, 1) * 255).to(torch.uint8).cpu(),
                    depth.to(torch.float16).cpu(),
                    (alpha * 255).to(torch.uint8).cpu(),
                )
                if current_iter not in self.grid_active_cell.prerenders:
                    self.grid_active_cell.prerenders[current_iter] = {}
                self.grid_active_cell.prerenders[current_iter][camera.id] = (
                    rgb,
                    depth,
                    alpha,
                )

    def grid_cull_active_cell_prerenders(self, older_than: int):
        if self._active_cell_index is None:
            raise ValueError("No active cell is set.")
        for iteration in list(self.grid_active_cell.prerenders.keys()):
            if iteration < older_than:
                del self.grid_active_cell.prerenders[iteration]

    def grid_clear_precomposited_layers(self):
        self._precomposited_bg.clear()
        self._precomposited_fg.clear()

    def grid_precompose_visible_layers(
        self, extra_cell_compensation: CompensationType = None
    ):
        if self._active_cell_index is None:
            raise ValueError("No active cell is set.")

        if not self._precomposite_enabled:
            return

        if extra_cell_compensation is None:
            extra_cell_compensation = self.grid_default_extra_cell_compensation

        if extra_cell_compensation == "disabled":
            return

        self.grid_clear_precomposited_layers()

        if extra_cell_compensation == "uniform":
            requested_iteration = (
                self.grid_calculate_newest_common_view_snapshot_iteration()
            )
        elif extra_cell_compensation == "last":
            requested_iteration = None
        else:
            raise ValueError(
                f"Invalid value for extra_cell_compensation: {extra_cell_compensation}"
            )

        if requested_iteration == -1:
            return

        active_cell = self.grid_active_cell
        visible_cameras = self.grid_get_visible_cameras_from_cell(
            self._active_cell_index
        )

        storage_device = (
            self.grid_model_train_device
            if self._precomposite_storage_device == "gpu"
            else self.grid_model_store_device
        )

        with torch.no_grad():
            for camera in visible_cameras:
                in_view_cells = self.grid_get_visible_cells_from_camera(camera.id)
                active_distance = active_cell.distance(camera)

                bg_layers = []
                fg_layers = []

                for cell in in_view_cells:
                    if cell.current_iter == 0:
                        continue
                    if cell.index == self._active_cell_index:
                        continue

                    try:
                        prerender = cell.get_prerender(camera, requested_iteration)
                    except ValueError:
                        continue

                    cell_distance = cell.distance(camera)
                    prerender_gpu = (
                        prerender[0].to(storage_device),
                        prerender[1].to(storage_device),
                        prerender[2].to(storage_device),
                    )

                    if cell_distance < active_distance:
                        fg_layers.append((prerender_gpu, cell_distance))
                    else:
                        bg_layers.append((prerender_gpu, cell_distance))

                if len(bg_layers) > 0:
                    bg_layers.sort(key=lambda x: -x[1])
                    bg_only = [layer[0] for layer in bg_layers]
                    self._precomposited_bg[camera.id] = composite_images_rgbda(bg_only)
                else:
                    dummy = torch.zeros(
                        3,
                        int(camera.image_height),
                        int(camera.image_width),
                        device=storage_device,
                    )
                    dummy_depth = torch.zeros(
                        1,
                        int(camera.image_height),
                        int(camera.image_width),
                        device=storage_device,
                    )
                    dummy_alpha = torch.zeros(
                        1,
                        int(camera.image_height),
                        int(camera.image_width),
                        device=storage_device,
                    )
                    self._precomposited_bg[camera.id] = (
                        dummy,
                        dummy_depth,
                        dummy_alpha,
                    )

                if len(fg_layers) > 0:
                    fg_layers.sort(key=lambda x: -x[1])
                    fg_only = [layer[0] for layer in fg_layers]
                    self._precomposited_fg[camera.id] = composite_images_rgbda(fg_only)
                else:
                    dummy = torch.zeros(
                        3,
                        int(camera.image_height),
                        int(camera.image_width),
                        device=storage_device,
                    )
                    dummy_depth = torch.zeros(
                        1,
                        int(camera.image_height),
                        int(camera.image_width),
                        device=storage_device,
                    )
                    dummy_alpha = torch.zeros(
                        1,
                        int(camera.image_height),
                        int(camera.image_width),
                        device=storage_device,
                    )
                    self._precomposited_fg[camera.id] = (
                        dummy,
                        dummy_depth,
                        dummy_alpha,
                    )

    def forward(
        self,
        camera: Union[ViewWithRes, KnownView],
        active_sh_degree: Union[int, None] = None,
        extra_cell_compensation: Union[CompensationType, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(camera, KnownView):
            return self.grid_active_cell.model.forward(camera, active_sh_degree)

        if extra_cell_compensation is None:
            extra_cell_compensation = self.grid_default_extra_cell_compensation

        active_rgb, active_depth, active_alpha = self.grid_active_cell.model.forward(
            camera, active_sh_degree
        )
        if extra_cell_compensation == "disabled":
            return active_rgb, active_depth, active_alpha

        if (
            self._precomposite_enabled
            and camera.id in self._precomposited_bg
            and camera.id in self._precomposited_fg
        ):
            bg_rgb, bg_depth, bg_alpha = self._precomposited_bg[camera.id]
            fg_rgb, fg_depth, fg_alpha = self._precomposited_fg[camera.id]

            bg_rgb = bg_rgb.to(self.grid_model_train_device)
            bg_depth = bg_depth.to(self.grid_model_train_device)
            bg_alpha = bg_alpha.to(self.grid_model_train_device)
            fg_rgb = fg_rgb.to(self.grid_model_train_device)
            fg_depth = fg_depth.to(self.grid_model_train_device)
            fg_alpha = fg_alpha.to(self.grid_model_train_device)

            composite = composite_images_rgbda(
                [
                    (bg_rgb, bg_depth, bg_alpha),
                    (active_rgb, active_depth, active_alpha),
                    (fg_rgb, fg_depth, fg_alpha),
                ]
            )
            return composite

        active_plane_distance = self.grid_active_cell.distance(camera)

        in_view_cells = self.grid_get_visible_cells_from_camera(camera.id)

        if extra_cell_compensation == "uniform":
            requested_iteration = (
                self.grid_calculate_newest_common_view_snapshot_iteration()
            )
        elif extra_cell_compensation == "last":
            requested_iteration = None
        else:
            raise ValueError(
                f"Invalid value for extra_cell_compensation: {extra_cell_compensation}"
            )

        if requested_iteration == -1:
            return active_rgb, active_depth, active_alpha

        prerendered_layers = [
            (cell.get_prerender(camera, requested_iteration), cell.distance(camera))
            for cell in in_view_cells
            if cell.current_iter != 0 and cell.index != self._active_cell_index
        ]
        prerendered_layers = [
            (
                (
                    rgb.to(self.grid_model_train_device),
                    depth.to(self.grid_model_train_device),
                    alpha.to(self.grid_model_train_device),
                ),
                plane_distance,
            )
            for ((rgb, depth, alpha), plane_distance) in prerendered_layers
        ]
        layers_with_dist = prerendered_layers + [
            ((active_rgb, active_depth, active_alpha), active_plane_distance)
        ]
        layers_with_dist.sort(key=lambda x: -x[1])
        layers = [layer[0] for layer in layers_with_dist]
        composite = composite_images_rgbda(layers)
        return composite

    def __call__(
        self,
        camera: ViewWithRes,
        extra_cell_compensation: Union[CompensationType, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(camera, extra_cell_compensation)

    def to(self, device: str):
        self.grid_active_cell.model.to(device)
        self.grid_model_train_device = device
