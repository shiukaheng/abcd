import random
from typing import List
import torch
from tqdm import tqdm
from gs.core.View import KnownView
from gs.core.GaussianModel import GaussianModel
from gs.helpers.formatting import format_number
from gs.helpers.loss import mix_l1_ssim_loss
from gs.helpers.scene import estimate_scene_scale
from gs.helpers.training import get_expon_lr_func
from gs.trainers.basic.config import BasicTrainConfig
from gs.trainers.basic.dynamic_parameters import densify, prune, prune_opacity_only, reset_opacities
from gs.visualization.Viewer import Viewer

def train(
        model: GaussianModel,
        cameras: List[KnownView],
        c: BasicTrainConfig,
    ):
    """
    This is the most basic trainer for Gaussian splatting. It mirrors the original training logic.
    """

    if c.model_train_device is not None:
        model.to(c.model_train_device)

    # Prepare model visualizer
    # viewer = Viewer(model, auto_start=False)

    # We estimate the scene size, such that a larger scene will have a larger learning rate. It is a heuristic defined in the original code.
    if c.scene_scale is None:
        scene_scale = estimate_scene_scale(cameras).item()
    else:
        scene_scale = c.scene_scale

    # We set different learning rates for each parameter type in a Gaussian.
    lr_groups = [
        {"params": [model.positions], "lr": c.positions_lr_init * scene_scale, "name": "positions"},
        {"params": [model.rotations], "lr": c.rotations_lr, "name": "rotations"},
        {"params": [model.scales], "lr": c.scales_lr, "name": "scales"},
        {"params": [model.opacities], "lr": c.opacities_lr, "name": "opacities"},
        {"params": [model.sh_coefficients_0], "lr": c.sh_coefficients_lr, "name": "sh_coefficients_0"},
        {"params": [model.sh_coefficients_rest], "lr": c.sh_coefficients_lr / 20.0, "name": "sh_coefficients_rest"},
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
    pbar = tqdm(range(c.starting_iter, c.iterations if c.ending_iter is None else c.ending_iter))
    for i in pbar:

        if (i % c.up_sh_interval == 0) and (active_sh_degree < model.sh_degree) and (i > 0):
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

        # print(f"Training on camera {camera.id}, Iteration {i}")

        # We perform a forward pass and compute the loss.

        rendered, _, _ = model.forward(camera, active_sh_degree=active_sh_degree)
        loss = mix_l1_ssim_loss(rendered, camera.image)

        # We perform a backward pass and update the parameters.
        loss.backward()
        model.backprop_stats()

        if c.camera_store_device is not None:
            camera.to(c.camera_store_device)

        with torch.no_grad():

            # Densification and culling
            if c.densify_from_iter < i < c.densify_until_iter and not (max_memory_reached or max_gaussians_reached):
                if i % c.densify_interval == 0:
                    densify(model, optimizer, scene_scale, c.densify_grad_threshold)
                    # model.assert_validity()
                    if i > c.opacity_reset_interval:
                        prune(model, optimizer, scene_scale, c.opacity_threshold, c.screen_size_threshold, c.world_size_threshold_multiplier)
                        # model.assert_validity()
                    else:
                        prune_opacity_only(model, optimizer, c.opacity_threshold)
                        # model.assert_validity()

            # Opacity reset
            if (i % c.opacity_reset_interval == 0) and (i > c.densify_from_iter):
                reset_opacities(model, optimizer, c.reset_to_opacity)

            # We perform the optimization step and zero the gradients
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) # We zero the gradients so they do not accumulate to the next iteration.

            # viewer.render_once()

            pbar.set_description(f"Loss: {loss.item()}, Num splats: {format_number(model.positions.size(0))}")
            torch.cuda.empty_cache() # We empty the cache to avoid memory leaks.

            # Check if we have reached the memory limit or the maximum number of Gaussians
            if c.max_memory is not None and torch.cuda.memory_allocated() > c.max_memory:
                max_memory_reached = True
                print("Out of memory")
            if c.max_gaussians is not None and model.positions.size(0) > c.max_gaussians:
                max_gaussians_reached = True
                print("Max Gaussians reached")

    # viewer.start()