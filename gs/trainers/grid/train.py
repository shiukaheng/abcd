from typing import List

import torch
from gs.core.GaussianModel import GaussianModel
from gs.core.View import KnownView
from gs.geometry.grid import Grid
from gs.helpers.scene import estimate_scene_scale
from gs.profiling import log_tensor_delete, log_tensor_set
from gs.trainers.grid.config import AutoGridConfig, GridTrainConfig
from gs.trainers.grid.GridGaussianModel import GridGaussianModel
from gs.trainers.basic.train import train as basic_train
from gs.visualization.Viewer import Viewer


def train(
    model: GaussianModel, cameras: List[KnownView], c: GridTrainConfig, _viewer=None, aim_logger=None
):
    try:
        model.assert_validity()
    except AssertionError as e:
        print("Model is invalid")
        raise e

    if _viewer is None:
        viewer = Viewer(auto_start=False)
    else:
        viewer = _viewer
    viewer.set_model(model)

    if c.scene_scale is None:
        scene_scale = estimate_scene_scale(cameras).item()
    else:
        scene_scale = c.scene_scale

    # Split model into grid on its original device
    model.to(c.model_store_device)

    for name in [
        "positions",
        "sh_coefficients_0",
        "sh_coefficients_rest",
        "rotations",
        "scales",
        "opacities",
    ]:
        tensor = getattr(model, name)
        if isinstance(tensor, torch.Tensor):
            log_tensor_set(f"model.{name}", tensor, role="parameter")

    # Calculate grid if required

    if isinstance(c.grid_config, Grid):
        grid = c.grid_config
    elif isinstance(c.grid_config, AutoGridConfig):
        grid = c.grid_config.apply(model)

    grid_model = GridGaussianModel.from_gaussian_model(
        model,
        cameras,
        grid,
        c.model_store_device,
        c.model_train_device,
        min_gaussians=c.min_gaussians,
        default_extra_cell_compensation=c.extra_cell_compensation,
        precomposite_enabled=c.precomposite_enabled,
        precomposite_storage=c.precomposite_storage,
    )

    for name in [
        "positions",
        "sh_coefficients_0",
        "sh_coefficients_rest",
        "rotations",
        "scales",
        "opacities",
    ]:
        log_tensor_delete(f"model.{name}", reason="split_into_cells")

    for cell_idx, cell in enumerate(grid_model.grid_iter()):
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

    # We train each cell in the grid for sync_interval iterations
    global_iteration = 0
    while all(
        cell.current_iter < c.iterations for cell in grid_model.grid_iter()
    ):  # While there are cells that have not reached the target iteration
        cells = list(grid_model.grid_iter())
        if len(cells) == 0:
            raise ValueError(
                "Cell filtering criteria is too strict, no cells left to train"
            )
        filtered_cells = list(
            filter(lambda cell: cell.current_iter < c.iterations, cells)
        )
        if len(filtered_cells) == 0:
            print("All cells have reached the target iteration")
            break

        for (
            cell
        ) in filtered_cells:  # For each cell that has not reached the target iteration
            print(
                f"Training cell {cell.index} for {c.sync_interval} iterations, overall progress: {cell.current_iter}/{c.iterations}"
            )

            # Configure cell to train for sync_interval iterations, starting from its current iteration
            target_iteration = min(cell.current_iter + c.sync_interval, c.iterations)
            c_cell = GridTrainConfig(**c.__dict__)
            c_cell.starting_iter = cell.current_iter
            c_cell.ending_iter = target_iteration
            c_cell.scene_scale = scene_scale

            bounding_box_viz = viewer.add_cell_bounary(cell)

            # Train the cell
            grid_model.grid_set_active_cell_index(cell.index)
            if c.precomposite_enabled and c.extra_cell_compensation != "disabled":
                grid_model.grid_precompose_visible_layers(c.extra_cell_compensation)
            visible_cameras = grid_model.grid_get_visible_cameras_from_cell(
                cell.index
            )  # Get the cameras that can see the cell
            iterations_this_round = target_iteration - cell.current_iter
            basic_train(grid_model, visible_cameras, c_cell, viewer, global_iteration, aim_logger=aim_logger)
            global_iteration += iterations_this_round

            cell.clean_model_edges()
            bounding_box_viz.remove()

            # Pre-render the cell if required
            if c.extra_cell_compensation != "disabled":
                if c.extra_cell_compensation == "last":
                    # print(f"Culling cell {cell.index} for {c.sync_interval} iterations")
                    grid_model.grid_cull_active_cell_prerenders(target_iteration)
                # print(f"Prerendering cell {cell.index} for {c.sync_interval} iterations")
                grid_model.grid_prerender_active_cell(target_iteration)

            # Update the cell to notifiy that it has been trained
            cell.current_iter = target_iteration

            if aim_logger is not None:
                total = sum(cell.model.positions.size(0) for cell in grid_model.grid_iter())
                aim_logger.track(global_iteration, gaussians_total=total)

            # # Ask user to continue
            # input("Press Enter to continue...")

    # Merge model from grid
    merged = grid_model.grid_merge()

    for cell in grid_model.grid_iter():
        prefix = f"cell_{cell.index}"
        for name in [
            "positions",
            "sh_coefficients_0",
            "sh_coefficients_rest",
            "rotations",
            "scales",
            "opacities",
        ]:
            log_tensor_delete(f"{prefix}.{name}", reason="merged")

    for name in [
        "positions",
        "sh_coefficients_0",
        "sh_coefficients_rest",
        "rotations",
        "scales",
        "opacities",
    ]:
        tensor = getattr(merged, name)
        if isinstance(tensor, torch.Tensor):
            log_tensor_set(f"merged.{name}", tensor, role="parameter")

    print("Training complete!")
    print(f"Iterations per cell: {c.iterations}")
    print(f"Total cells trained: {len(grid_model.grid_cells)}")
    print(f"Total Gaussians: {len(merged)}")

    return merged
