import os
import random
import time
import warnings
from typing import List, Union
import torch
from tqdm import tqdm
from gs.core.View import KnownView
from gs.core.GaussianModel import GaussianModel
from gs.helpers.formatting import format_number
from gs.helpers.image import torch_to_cv2, torch_to_numpy, torch_to_pil
from gs.helpers.loss import mix_l1_ssim_loss, l2_loss
from gs.helpers.scene import estimate_scene_scale
from gs.helpers.training import get_expon_lr_func
from gs.profiling import log_iteration, log_tensor_set
from gs.trainers.basic.config import BasicTrainConfig
from gs.trainers.basic.dynamic_parameters import (
    densify,
    prune,
    prune_opacity_only,
    reset_opacities,
)
from gs.visualization.Viewer import Viewer

import cv2


# Check if display is available for OpenCV preview
def _has_display():
    """Check if a display is available for OpenCV GUI functions."""
    return os.environ.get("DISPLAY") is not None or os.name == "nt"


_HAS_DISPLAY = _has_display()


def train(
    model: GaussianModel,
    cameras: List[KnownView],
    c: BasicTrainConfig,
    _viewer: Union[
        None, Viewer
    ] = None,  # If this training loop is chained with another, we can pass the viewer to avoid creating a new one.
    global_iteration_offset: int = 0,
):
    """
    This is the most basic trainer for Gaussian splatting. It mirrors the original training logic.
    """

    if c.model_train_device is not None:
        model.to(c.model_train_device)

    for name in [
        "positions",
        "sh_coefficients_0",
        "sh_coefficients_rest",
        "rotations",
        "scales",
        "opacities",
        "_gradient_accumulator",
        "_gradient_accumulator_denominator",
        "max_radii2D",
    ]:
        tensor = getattr(model, name)
        if isinstance(tensor, torch.Tensor):
            log_tensor_set(f"model.{name}", tensor, role="parameter")

    # Prepare model visualizer
    if _viewer is None:
        viewer = Viewer(auto_start=False)
    else:
        viewer = _viewer
    viewer.set_model(model)

    # We estimate the scene size, such that a larger scene will have a larger learning rate. It is a heuristic defined in the original code.
    if c.scene_scale is None:
        scene_scale = estimate_scene_scale(cameras).item()
    else:
        scene_scale = c.scene_scale

    # We set different learning rates for each parameter type in a Gaussian.
    lr_groups = [
        {
            "params": [model.positions],
            "lr": c.positions_lr_init * scene_scale,
            "name": "positions",
        },
        {"params": [model.rotations], "lr": c.rotations_lr, "name": "rotations"},
        {"params": [model.scales], "lr": c.scales_lr, "name": "scales"},
        {"params": [model.opacities], "lr": c.opacities_lr, "name": "opacities"},
        {
            "params": [model.sh_coefficients_0],
            "lr": c.sh_coefficients_lr,
            "name": "sh_coefficients_0",
        },
        {
            "params": [model.sh_coefficients_rest],
            "lr": c.sh_coefficients_lr / 20.0,
            "name": "sh_coefficients_rest",
        },
    ]

    # With all this set, we can define the optimizer.
    optimizer = torch.optim.Adam(lr_groups, lr=0.0, eps=1e-15)

    # We define the learning rate scheduler for the positions parameters, such that initially it is high and decays exponentially.
    position_lr_scheduler = get_expon_lr_func(
        lr_init=c.positions_lr_init * scene_scale,
        lr_final=c.positions_lr_final * scene_scale,
        lr_delay_mult=c.position_lr_delay_mult,
        max_steps=c.position_lr_max_steps,
    )

    # We define a list of cameras to train on. If randomize is True, we shuffle the cameras. It will be filled whenever it is empty.
    train_cameras: List[KnownView] = []

    # We set the active SH degree to 0. For this basic trainer, each Gaussian will just have a constant color.
    active_sh_degree = 0

    # RAM backup in case of out of CUDA memory
    max_memory_reached = False
    max_gaussians_reached = False

    # Create range of iterations, modifiable by starting_iter and ending_iter
    pbar = tqdm(
        range(c.starting_iter, c.iterations if c.ending_iter is None else c.ending_iter)
    )
    for i in pbar:
        if (
            (i % c.up_sh_interval == 0)
            and (active_sh_degree < model.sh_degree)
            and (i > 0)
        ):
            active_sh_degree += 1

        # If we have no cameras to train on, we fill the list with all cameras.
        if len(train_cameras) == 0:
            train_cameras += reversed(cameras)
            if c.randomize:
                random.shuffle(train_cameras)

        # We update the learning rate for the positions parameters according to the scheduler.
        for group in optimizer.param_groups:
            if "positions" in group["name"]:
                group["lr"] = position_lr_scheduler(i + 1)
                break

        # We get the next camera to train on.
        camera = train_cameras.pop()
        if c.camera_train_device is not None:
            camera.to(c.camera_train_device)
        if hasattr(camera, "image") and camera.image is not None:
            log_tensor_set(f"cam_{camera.id}.image", camera.image, role="buffer")

        # print(f"Training on camera {camera.id}, Iteration {i}")

        # We perform a forward pass and compute the loss.

        rendered, depth, alpha = model.forward(
            camera, active_sh_degree=active_sh_degree
        )

        # Fast MSE loss only (LPIPS/L1+SSIM disabled for speed)
        loss = l2_loss(rendered, camera.image)

        # Add a loss for transparency so the Gaussians fill the screen. We want alpha to be 1 everywhere.
        if c.transparency_loss_weight > 0:
            loss += c.transparency_loss_weight * torch.mean((alpha - 1.0) ** 2)

        # Add a loss for the uncertainty of the opacities. We want the opacities to be either 0 or 1.
        if c.opacity_uncertainty_penalty > 0:
            # This probably makes training unstable, because this regularizes all Gaussians when during each iteration we only see a few.
            # This results in Gaussians being pruned too early.
            opacity_uncertainty = torch.sin(model.opacities * 3.14159) ** 2
            loss += c.opacity_uncertainty_penalty * torch.mean(opacity_uncertainty)

        # We perform a backward pass and update the parameters.
        loss.backward()
        model.backprop_stats()

        # Show rendered image as its training
        if c.preview_camera is not None:
            if c.preview_camera == "all" or c.preview_camera == camera.id:
                show_image(rendered)
            else:
                with torch.no_grad():
                    c.preview_camera.to(c.camera_train_device)
                    preview_render, _, _ = model.forward(
                        c.preview_camera, active_sh_degree=active_sh_degree
                    )
                    show_image(preview_render)

        if c.camera_store_device is not None:
            camera.to(c.camera_store_device)
        if hasattr(camera, "image") and camera.image is not None:
            log_tensor_set(f"cam_{camera.id}.image", camera.image, role="buffer")

        with torch.no_grad():
            # Densification and culling
            if c.densify_from_iter < i < c.densify_until_iter and not (
                max_memory_reached or max_gaussians_reached
            ):
                if i % c.densify_interval == 0:
                    densify(
                        model,
                        optimizer,
                        scene_scale,
                        c.densify_grad_threshold,
                        split_n_samples=c.split_n_samples,
                        split_shrink_factor=c.split_shrink_factor,
                    )
                    if i > c.opacity_reset_interval:
                        prune(
                            model,
                            optimizer,
                            scene_scale,
                            c.opacity_threshold,
                            c.screen_size_threshold,
                            c.world_size_threshold_multiplier,
                        )
                    else:
                        prune_opacity_only(model, optimizer, c.opacity_threshold)

            # Opacity reset
            if (i % c.opacity_reset_interval == 0) and (i > c.densify_from_iter):
                reset_opacities(model, optimizer, c.reset_to_opacity)

            # We perform the optimization step and zero the gradients
            optimizer.step()
            optimizer.zero_grad(
                set_to_none=True
            )  # We zero the gradients so they do not accumulate to the next iteration.

            log_iteration(i + global_iteration_offset)

            viewer.render_once()

            pbar.set_description(
                f"Loss: {loss.item()}, Num splats: {format_number(model.positions.size(0))}"
            )
            torch.cuda.empty_cache()  # We empty the cache to avoid memory leaks.

            # Check if we have reached the memory limit or the maximum number of Gaussians
            if (
                c.max_memory is not None
                and torch.cuda.memory_allocated() > c.max_memory
            ):
                max_memory_reached = True
                print("Out of memory")
            if (
                c.max_gaussians is not None
                and model.positions.size(0) > c.max_gaussians
            ):
                max_gaussians_reached = True
                print("Max Gaussians reached")

    return model


def show_image(preview_render, name="Rendered"):
    if not _HAS_DISPLAY:
        warnings.warn(
            "No display detected. Set preview_camera=None to disable OpenCV preview attempts."
        )
        return
    try:
        cv2.imshow(name, torch_to_cv2(preview_render.detach().cpu()))
        cv2.waitKey(5)
    except cv2.error as e:
        if "could not connect to display" in str(e) or "could not load" in str(e):
            warnings.warn(
                "Could not connect to display for OpenCV preview. Set preview_camera=None to disable this warning."
            )
        else:
            raise e
