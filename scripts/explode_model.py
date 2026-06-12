import os
import torch
import tyro
from typing import Optional

from gs.core.GaussianModel import GaussianModel
from gs.geometry.grid import Grid
from gs.trainers.grid.grid_utils import merge_model, split_model
from gs.visualization.Viewer import Viewer


def explode_model(
    model_path: str,
    grid_size: float,
    gap: float,
    output: Optional[str] = None,
    save_cells_dir: Optional[str] = None,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
    grid_origin_z: float = 0.0,
    min_gaussians: int = 1,
    no_gui: bool = False,
):
    print(f"Loading model: {model_path}")
    model = GaussianModel.from_ply(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    device = model.positions.device

    grid = Grid(
        grid_size=grid_size,
        grid_origin=torch.tensor(
            [grid_origin_x, grid_origin_y, grid_origin_z], device=device
        ),
    )
    print(
        f"Grid size: {grid_size}, origin: ({grid_origin_x}, {grid_origin_y}, {grid_origin_z})"
    )

    cells = split_model(model, grid, min_gaussians)
    print(f"Found {len(cells)} cells (min_gaussians={min_gaussians})")
    if len(cells) == 0:
        print("No cells with gaussians found. Try lowering --min-gaussians.")
        return

    model_center = model.calculate_bounding_box().center

    shown = min(20, len(cells))
    for i, (cell_index, (cell_model, cell_bb)) in enumerate(cells.items()):
        cell_center = cell_bb.center
        direction = cell_center - model_center
        norm = torch.norm(direction)
        if norm > 1e-6:
            offset = (direction / norm) * gap
            cell_model.positions.data += offset

        if i < shown:
            num = cell_model.positions.size(0)
            print(f"  Cell {cell_index}: {num} gaussians, offset applied")

    if len(cells) > shown:
        print(f"  ... and {len(cells) - shown} more cells")

    exploded_model = merge_model(list(cells.values()), device, clean=False)
    total_gs = exploded_model.positions.size(0)
    print(f"Merged exploded model: {total_gs} gaussians across {len(cells)} cells")

    if output is not None:
        exploded_model.save_ply(output)
        print(f"Saved exploded model to {output}")

    if save_cells_dir is not None:
        os.makedirs(save_cells_dir, exist_ok=True)
        for cell_index, (cell_model, _) in cells.items():
            path = os.path.join(save_cells_dir, f"{cell_index.to_string_id()}.ply")
            cell_model.save_ply(path)
        print(f"Saved {len(cells)} cell PLYs to {save_cells_dir}")

    if no_gui:
        print("Skipping viewer (--no-gui)")
        return

    print("Launching viewer...")
    viewer = Viewer(auto_start=False)
    viewer.set_model(exploded_model)

    for _, (_, cell_bb) in cells.items():
        viewer.add_bounding_box_boundary(cell_bb, color=(100, 100, 255), line_width=2)

    viewer.start(threaded=False)


if __name__ == "__main__":
    tyro.cli(explode_model)
