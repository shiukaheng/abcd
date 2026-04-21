from typing import List

import torch
from gs.core.GaussianModel import GaussianModel
from gs.core.View import KnownView
from gs.geometry.grid import Grid
from gs.helpers.scene import estimate_scene_scale
from gs.trainers.grid.config import AutoGridConfig, GridTrainConfig
from gs.trainers.grid.GridGaussianModel import GridGaussianModel
from gs.trainers.basic.train import train as basic_train
from gs.visualization.Viewer import Viewer


def train(
    model: GaussianModel, cameras: List[KnownView], c: GridTrainConfig, _viewer=None
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
            basic_train(grid_model, visible_cameras, c_cell, viewer, global_iteration)
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

            # # Ask user to continue
            # input("Press Enter to continue...")

    # Merge model from grid
    merged = grid_model.grid_merge()

    print("Training complete!")
    print(f"Iterations per cell: {c.iterations}")
    print(f"Total cells trained: {len(grid_model.grid_cells)}")
    print(f"Total Gaussians: {len(merged)}")

    return merged
