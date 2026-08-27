import tempfile
from typing import Dict, Generic, List, Literal, Tuple, TypeVar, Union

import torch

from abcd.compositing.alpha_compositing import composite_images_rgbda
from abcd.core.GaussianModel import GaussianModel
from abcd.core.View import KnownView, View, ViewWithRes
from abcd.geometry.bounding_box import BoundingBox
from abcd.geometry.grid import Grid, GridIndex
from abcd.profiling import log_tensor_delete, log_tensor_set
from abcd.trainers.basic.state import BasicTrainState
from abcd.trainers.grid.forward_properties import forward_to_active_cell
from abcd.trainers.grid.grid_utils import bounding_box_mask, merge_model, split_model
from abcd.trainers.grid.storage import (
    CachedRender,
    DirectoryRenderCache,
    DirectoryShardStore,
    MemoryRenderCache,
    MemoryShardStore,
    ShardState,
)

T = TypeVar("T")


class GridGaussianCell(Generic[T]):
    index: GridIndex
    model: GaussianModel | None
    bounding_box: BoundingBox
    grid: Grid

    def __init__(
        self,
        grid: Grid,
        index: GridIndex,
        model: GaussianModel | None,
        bounding_box: BoundingBox,
        render_cache: DirectoryRenderCache,
        shard_store: DirectoryShardStore,
        gaussian_count: int | None = None,
        training_state: BasicTrainState | None = None,
    ):
        self.grid = grid
        self.index = index
        self.model = model
        self.gaussian_count = len(model) if model is not None else gaussian_count or 0
        self.bounding_box = bounding_box
        self.training_state = training_state or BasicTrainState()
        self.render_cache = render_cache
        self.shard_store = shard_store
        self.center = torch.mean(
            torch.stack([bounding_box.min, bounding_box.max]), dim=0
        )

    def plane_distance(self, camera: View) -> float:
        return float(
            torch.dot(
                self.center.to("cpu") - camera.center.to("cpu"),
                camera.look_at.to("cpu"),
            ).item()
        )

    def distance(self, camera: View) -> float:
        return torch.norm(self.center.to("cpu") - camera.center.to("cpu"))

    def get_prerender(
        self, camera: KnownView[T], iteration: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if iteration is None:
            iteration = self.current_iter
        try:
            prerender = self.render_cache.load(self.index, camera.id, iteration)
        except KeyError as error:
            raise ValueError(
                f"Prerender for camera {camera.id} at iteration {iteration} is not available."
            ) from error
        rgb, depth, alpha = prerender.rgb, prerender.depth, prerender.alpha
        rgb, depth, alpha = (
            rgb.to(torch.float32) / 255,
            depth.to(torch.float32),
            alpha.to(torch.float32) / 255,
        )
        return rgb, depth, alpha

    @property
    def current_iter(self) -> int:
        return self.training_state.next_iteration

    @current_iter.setter
    def current_iter(self, value: int) -> None:
        self.training_state.next_iteration = value

    def clean_model_edges(self):
        if self.model is None:
            raise ValueError(f"Cell {self.index} is not loaded")
        mask = bounding_box_mask(self.model, self.bounding_box)
        self.training_state.subset(mask)
        self.model = self.model[mask]
        self.gaussian_count = len(self.model)

    def load(self) -> None:
        if self.model is not None:
            return
        state = self.shard_store.load(self.index)
        self.model = state.model
        self.training_state = state.training
        self.gaussian_count = len(self.model)

    def store(self) -> None:
        if self.model is None:
            return
        self.gaussian_count = len(self.model)
        self.shard_store.store(self.index, ShardState(self.model, self.training_state))


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
        cache_dir: str | None = None,
        cache_fingerprint: str = "abcd-v1",
        resume: bool = False,
        cache_storage: Literal["disk", "ram"] = "disk",
    ):
        if cache_storage == "ram":
            if resume:
                raise ValueError("RAM cache cannot resume a previous process")
            self.grid_cache_dir = None
            self.render_cache = MemoryRenderCache()
            self.shard_store = MemoryShardStore()
        else:
            if cache_dir is None:
                cache_dir = tempfile.mkdtemp(prefix="abcd-cache-")
            self.grid_cache_dir = cache_dir
            self.render_cache = DirectoryRenderCache(cache_dir, cache_fingerprint)
            self.shard_store = DirectoryShardStore(cache_dir, cache_fingerprint)
        if resume:
            descriptions = self.shard_store.descriptions()
            if not descriptions:
                raise ValueError(
                    f"No resumable shards found in {self.shard_store.root}"
                )
            self.grid_cells = {
                index: GridGaussianCell(
                    grid,
                    index,
                    None,
                    grid.get_bounding_box(index),
                    self.render_cache,
                    self.shard_store,
                    gaussian_count=description["gaussian_count"],
                    training_state=BasicTrainState(
                        next_iteration=description["next_iteration"]
                    ),
                )
                for index, description in descriptions.items()
            }
        else:
            self.grid_cells = {
                index: GridGaussianCell(
                    grid,
                    index,
                    cell_model,
                    grid.get_bounding_box(index),
                    self.render_cache,
                    self.shard_store,
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

        if not resume:
            for cell in self.grid_cells.values():
                cell.store()
                cell.model = None

        self.grid_calculate_visibility()

    def grid_calculate_visibility(self):
        self.camera_to_grid_visibility = {}
        self.grid_to_camera_visibility = {
            cell.index: [] for cell in self.grid_cells.values()
        }
        for camera in self.grid_cameras.values():
            self.camera_to_grid_visibility[camera.id] = []
            for cell in self.grid_cells.values():
                if camera.frustum.intersects_bounding_box(cell.bounding_box):
                    self.camera_to_grid_visibility[camera.id].append(cell.index)
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
        cache_dir: str | None = None,
        cache_fingerprint: str = "abcd-v1",
        resume: bool = False,
        cache_storage: Literal["disk", "ram"] = "disk",
    ):
        cells = (
            {}
            if resume
            else split_to_grid_gaussian_cells(
                input_model, grid, min_gaussians=min_gaussians
            )
        )
        print(
            "Resuming grid from disk"
            if resume
            else f"Split model into {len(cells)} cells"
        )
        return GridGaussianModel(
            cells,
            cameras,
            grid,
            model_store_device,
            model_train_device,
            default_extra_cell_compensation,
            precomposite_enabled,
            precomposite_storage,
            cache_dir,
            cache_fingerprint,
            resume,
            cache_storage,
        )

    def grid_get(self, index: GridIndex) -> GaussianModel:
        cell = self.grid_cells[index]
        cell.load()
        return cell.model

    def grid_len(self) -> int:
        return len(self.grid_cells)

    def grid_iter(self):
        return iter(self.grid_cells.values())

    def grid_merge(self, clean=True) -> GaussianModel:
        if self._active_cell_index is not None:
            self.grid_active_cell.store()
        for cell in self.grid_cells.values():
            cell.load()
        return merge_model(
            [(cell.model, cell.bounding_box) for cell in self.grid_cells.values()],
            self.grid_model_store_device,
            clean,
        )

    def grid_set_active_cell_index(self, index: GridIndex):
        if (
            self._active_cell_index == index
            and self.grid_cells[index].model is not None
        ):
            return
        if self._active_cell_index is not None:
            old_cell = self.grid_active_cell
            old_cell.model.to(self.grid_model_store_device)
            old_cell.store()
            old_cell.model = None
        active_cell = self.grid_cells[index]
        active_cell.load()
        active_cell.model.to(self.grid_model_train_device)
        self._active_cell_index = index

        for cell in [active_cell]:
            prefix = f"cell_{cell.index}"
            for name in [
                "positions",
                "sh_coefficients_0",
                "sh_coefficients_rest",
                "rotations",
                "scales",
                "opacities",
            ]:
                tensor = getattr(cell.model, name)
                if isinstance(tensor, torch.Tensor):
                    log_tensor_set(f"{prefix}.{name}", tensor, role="parameter")

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
            keys = self.render_cache.iterations(cell.index)
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
                self.render_cache.store(
                    self._active_cell_index,
                    camera.id,
                    current_iter,
                    CachedRender(rgb, depth, alpha),
                )
                key = f"cell_{self._active_cell_index}.prerender.{current_iter}.cam_{camera.id}"
                log_tensor_set(
                    key,
                    rgb,
                    role="prerender",
                )

    def grid_cull_active_cell_prerenders(self, older_than: int):
        if self._active_cell_index is None:
            raise ValueError("No active cell is set.")
        self.render_cache.remove_older_than(self._active_cell_index, older_than)

    def grid_clear_precomposited_layers(self):
        for cam_id in list(self._precomposited_bg.keys()):
            log_tensor_delete(f"cam_{cam_id}.precomp_bg", reason="cleared")
        for cam_id in list(self._precomposited_fg.keys()):
            log_tensor_delete(f"cam_{cam_id}.precomp_fg", reason="cleared")
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
                active_distance = active_cell.plane_distance(camera)

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

                    cell_distance = cell.plane_distance(camera)
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

                log_tensor_set(
                    f"cam_{camera.id}.precomp_bg",
                    self._precomposited_bg[camera.id][0],
                    role="precomposite",
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

                log_tensor_set(
                    f"cam_{camera.id}.precomp_fg",
                    self._precomposited_fg[camera.id][0],
                    role="precomposite",
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

        active_plane_distance = self.grid_active_cell.plane_distance(camera)

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
            (
                cell.get_prerender(camera, requested_iteration),
                cell.plane_distance(camera),
            )
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

    def constrain_positions(self) -> None:
        """Keep active means inside the cell used for compositing and ownership."""

        cell = self.grid_active_cell
        minimum = cell.bounding_box.min.to(cell.model.positions)
        maximum = cell.bounding_box.max.to(cell.model.positions)
        inclusive_maximum = torch.nextafter(maximum, minimum)
        with torch.no_grad():
            cell.model.positions.clamp_(min=minimum, max=inclusive_maximum)

    def __call__(
        self,
        camera: ViewWithRes,
        extra_cell_compensation: Union[CompensationType, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(camera, extra_cell_compensation=extra_cell_compensation)

    def to(self, device: str):
        self.grid_active_cell.model.to(device)
        self.grid_model_train_device = device
