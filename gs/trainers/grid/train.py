from typing import List
from gs.core.GaussianModel import GaussianModel
from gs.core.View import KnownView
from gs.trainers.grid.config import GridTrainConfig
from gs.trainers.grid.GridGaussianModel import GridGaussianModel
from gs.trainers.basic.train import train as basic_train
from gs.visualization.Viewer import Viewer

def train(
        model: GaussianModel, 
        cameras: List[KnownView], 
        c: GridTrainConfig,
        _viewer = None
    ):

    if _viewer is None:
        viewer = Viewer(auto_start=False)
    else:
        viewer = _viewer
    viewer.set_model(model)

    # Split model into grid on its original device
    model.to(c.model_store_device)
    grid_model = GridGaussianModel.from_gaussian_model(model, cameras, c.grid, c.model_store_device, c.model_train_device, min_gaussians=c.min_gaussians, default_extra_cell_compensation="last")

    # We train each cell in the grid for sync_interval iterations
    while all(cell.current_iter < c.iterations for cell in grid_model.grid_iter()): # While there are cells that have not reached the target iteration

        for cell in filter(lambda cell: cell.current_iter < c.iterations, grid_model.grid_iter()): # For each cell that has not reached the target iteration

            print(f"Training cell {cell.index} for {c.sync_interval} iterations")

            # Configure cell to train for sync_interval iterations, starting from its current iteration
            target_iteration = min(cell.current_iter + c.sync_interval, c.iterations) 
            c_cell = GridTrainConfig(**c.__dict__)
            c_cell.starting_iter = cell.current_iter
            c_cell.ending_iter = target_iteration

            bounding_box_viz = viewer.add_cell_bounary(cell)

            # Train the cell
            grid_model.set_active_cell_index(cell.index)
            visible_cameras = grid_model.get_visible_cameras_from_cell(cell.index) # Get the cameras that can see the cell
            basic_train(
                grid_model,
                visible_cameras,
                c_cell,
                viewer
            )

            cell.clean_model_edges()
            bounding_box_viz.remove()

            # Pre-render the cell if required
            if c.extra_cell_compensation != "disabled":
                if c.extra_cell_compensation == "last":
                    grid_model.cull_active_cell_prerenders(target_iteration)
                grid_model.prerender_active_cell(target_iteration)

            # Update the cell to notifiy that it has been trained
            cell.current_iter = target_iteration

            # # Ask user to continue
            # input("Press Enter to continue...")

    # Merge model from grid
    merged = grid_model.merge()

    if _viewer is None:
        viewer.start()

    return merged


