import torch
from torch import nn

from abcd.core.GaussianModel import GaussianModel
from abcd.helpers.transforms import quat_to_rot
from abcd.profiling import log_tensor_set

"""
This module contains helper functions for densifying and pruning Gaussian models, as well as adjusting the opacities. 
These are not class methods for GaussianModel since it requires direct access to the optimizer.
"""


def densify(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    scene_scale: float,
    gradient_threshold: float,
    percent_dense: float = 0.01,
    split_n_samples: int = 2,
    split_shrink_factor: float = 0.8,
) -> None:
    """
    Densifies the Gaussian model by cloning and splitting Gaussians based on the gradient magnitude and the size of the Gaussian.
    """
    gradients = model.mean_gradient_magnitude
    exceed_gradient_mask = torch.where(
        gradients > gradient_threshold, True, False
    ).squeeze(1)
    if model.scales_range is None:
        large_gaussian_mask = (
            torch.max(model.scales_activation(model.scales), dim=1).values
            > percent_dense * scene_scale
        )
    else:
        large_gaussian_mask = (
            torch.max(model.scales_activation(model.scales), dim=1).values
            > model.scales_range[1] * 0.75
        )
    clone_mask = torch.logical_and(exceed_gradient_mask, ~large_gaussian_mask)
    clone_gaussians(model, optimizer, clone_mask)
    if model.scales_range is None:
        split_mask = torch.logical_and(exceed_gradient_mask, large_gaussian_mask)
    else:
        # Split if exceeding 90% of the upper limit regardless of gradient
        split_mask = large_gaussian_mask
    padded_split_mask = pad_mask(split_mask, model, model.positions.device)
    split_gaussians(
        model,
        optimizer,
        padded_split_mask,
        n_samples=split_n_samples,
        split_shrink_factor=split_shrink_factor,
    )


def prune(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    scene_scale: float,
    opacity_threshold: float,
    screen_size_threshold: float,
    world_size_threshold_multiplier: float = 0.1,
) -> None:
    """
    Prunes the Gaussian model by removing Gaussians based on opacity, screen size and world size.
    """
    opacity_mask = model.opacities_activation(model.opacities) < opacity_threshold
    screen_size_mask = model.max_radii2D > screen_size_threshold
    world_size_mask = (
        model.scales_activation(model.scales).max(dim=1).values
        > world_size_threshold_multiplier * scene_scale
    )
    final_mask = opacity_mask.squeeze(1).logical_or_(
        screen_size_mask.squeeze(1).logical_or_(world_size_mask)
    )
    cull_gaussians(model, optimizer, final_mask)


def prune_opacity_only(
    model: GaussianModel, optimizer: torch.optim.Adam, opacity_threshold: float
) -> None:
    """
    Prunes the Gaussian model by removing Gaussians based on opacity only.
    """
    opacity_mask = model.opacities_activation(model.opacities) < opacity_threshold
    cull_gaussians(model, optimizer, opacity_mask.squeeze())


def append_new_gaussians(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
) -> None:
    """
    Appends new Gaussians to the model and optimizer.
    """
    device = model.positions.device
    extension_lookup = {
        "positions": positions,
        "rotations": rotations,
        "scales": scales,
        "opacities": opacities,
        "sh_coefficients_0": sh_coefficients_0,
        "sh_coefficients_rest": sh_coefficients_rest,
    }
    for group in optimizer.param_groups:
        if len(group["params"]) != 1:
            raise ValueError(
                f"Unexpected number of parameters in optimizer group. Only one parameter is expected, as initialized in the GaussianModel. Parameter group: {group}"
            )
        if group["name"] not in extension_lookup:
            raise ValueError(
                f"Unexpected parameter name {group['name']} in optimizer group. Expected one of 'positions', 'rotations', 'scales', 'opacities', 'sh_coefficients_0', 'sh_coefficients_rest'."
            )
        extension_params = extension_lookup[group["name"]]
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat(
                (stored_state["exp_avg"], torch.zeros_like(extension_params)), dim=0
            )
            stored_state["exp_avg_sq"] = torch.cat(
                (stored_state["exp_avg_sq"], torch.zeros_like(extension_params)), dim=0
            )
            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                torch.cat(
                    (group["params"][0], extension_params), dim=0
                ).requires_grad_()
            )
            optimizer.state[group["params"][0]] = stored_state
            setattr(model, group["name"], group["params"][0])
        else:
            group["params"][0] = nn.Parameter(
                torch.cat(
                    (group["params"][0], extension_params), dim=0
                ).requires_grad_()
            )
            setattr(model, group["name"], group["params"][0])
    model._gradient_accumulator = torch.zeros(
        (model.positions.shape[0], 1), device=device
    )
    model._gradient_accumulator_denominator = torch.zeros(
        (model.positions.shape[0], 1), device=device
    )
    model.max_radii2D = torch.zeros(
        (model.positions.shape[0]), device=device
    ).unsqueeze(1)

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


def check_mask_validity(mask: torch.Tensor, model: GaussianModel) -> None:
    """
    Checks if a mask is valid for the model or will lead to out-of-bounds errors.
    """
    if mask.size(0) != model.positions.size(0):
        print("Mask length does not match model length.")


def clone_gaussians(
    model: GaussianModel, optimizer: torch.optim.Adam, mask: torch.Tensor
) -> None:
    """
    Clones Gaussians based on a mask.
    """
    # check_mask_validity(mask, model)
    positions = model.positions[mask]
    rotations = model.rotations[mask]
    scales = model.scales[mask]
    opacities = model.opacities[mask]
    sh_coefficients_0 = model.sh_coefficients_0[mask]
    sh_coefficients_rest = model.sh_coefficients_rest[mask]
    append_new_gaussians(
        model,
        optimizer,
        positions,
        rotations,
        scales,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
    )


def cull_gaussians(
    model: GaussianModel, optimizer: torch.optim.Adam, mask: torch.Tensor
) -> None:
    """
    Removes Gaussians based on a mask.
    """
    # check_mask_validity(mask, model)
    keep_mask = ~mask
    for group in optimizer.param_groups:
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][keep_mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][keep_mask]
            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                group["params"][0][keep_mask].requires_grad_()
            )
            optimizer.state[group["params"][0]] = stored_state
            setattr(model, group["name"], group["params"][0])
        else:
            group["params"][0] = nn.Parameter(
                group["params"][0][keep_mask].requires_grad_()
            )
            setattr(model, group["name"], group["params"][0])
    model._gradient_accumulator = model._gradient_accumulator[keep_mask]
    model._gradient_accumulator_denominator = model._gradient_accumulator_denominator[
        keep_mask
    ]
    model.max_radii2D = model.max_radii2D[keep_mask]

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


def split_gaussians(
    model: GaussianModel,
    optimizer: torch.optim.Adam,
    mask: torch.Tensor,
    n_samples: int = 2,
    split_shrink_factor: float = 0.8,
) -> None:
    """
    Splits Gaussians based on a mask.
    """
    # check_mask_validity(mask, model)
    device = model.positions.device
    positions = model.positions[mask]
    rotations = model.rotations[mask]
    scales = model.scales[mask]
    opacities = model.opacities[mask]
    sh_coefficients_0 = model.sh_coefficients_0[mask]
    sh_coefficients_rest = model.sh_coefficients_rest[mask]

    # We sample from a normal distribution with a standard deviation of 80% of the original scale.
    sds = model.scales_activation(scales).repeat(n_samples, 1)
    means = torch.zeros((sds.size(0), 3), device=device)
    samples = torch.normal(means, sds)
    p_rotations = quat_to_rot(rotations).repeat(n_samples, 1, 1)

    # We create new Gaussians with the sampled positions. Scale is divided by 0.8 * n_samples of the original scale.
    new_positions = torch.bmm(p_rotations, samples.unsqueeze(-1)).squeeze(
        -1
    ) + positions.repeat(n_samples, 1)
    new_rotations = rotations.repeat(n_samples, 1)
    if model.scales_range is None:
        new_scales = model.scales_inverse_activation(
            model.scales_activation(scales).repeat(n_samples, 1)
            / (split_shrink_factor * n_samples)
        )
    else:
        new_scales = model.scales_inverse_activation(
            torch.clamp(
                model.scales_activation(scales).repeat(n_samples, 1)
                / (split_shrink_factor * n_samples),
                model.scales_range[0],
                model.scales_range[1],
            )
        )
        # Because of quirks of inverse_activation, -inf will be returned when the scale is exactly at the lower bound, and nan when at the upper bound.
        # Thus,
    new_opacities = opacities.repeat(n_samples, 1)
    new_sh_coefficients_0 = sh_coefficients_0.repeat(n_samples, 1, 1)
    new_sh_coefficients_rest = sh_coefficients_rest.repeat(n_samples, 1, 1)

    append_new_gaussians(
        model,
        optimizer,
        new_positions,
        new_rotations,
        new_scales,
        new_opacities,
        new_sh_coefficients_0,
        new_sh_coefficients_rest,
    )

    padded_mask = pad_mask(mask, model, device)
    cull_gaussians(model, optimizer, padded_mask)  # Remove the original Gaussians


def pad_mask(
    mask: torch.Tensor, model: GaussianModel, device: torch.device
) -> torch.Tensor:
    """
    Pads a mask to the length of the model.
    """
    mask_length = mask.size(0)
    model_length = model.positions.size(0)
    n_new_gaussians = model_length - mask_length
    return torch.cat(
        (mask, torch.zeros(n_new_gaussians, dtype=torch.bool, device=device))
    )


def replace_tensor_to_optimizer(
    optimizer: torch.optim.Adam, tensor: torch.Tensor, name: str
) -> dict:
    """
    Replaces a tensor in the optimizer with a new tensor.
    E.g. replace_tensor_to_optimizer(optimizer, new_tensor, "positions")
    Used for updating the model parameters in the optimizer.
    """
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group["name"] == name:
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def reset_opacities(
    model: GaussianModel, optimizer: torch.optim.Adam, opacity: float = 0.01
) -> None:
    """
    Resets the opacities of the model to a specific value.
    """
    new_opacities = model.opacities_inverse_activation(
        torch.min(
            model.opacities_activation(model.opacities),
            torch.ones_like(model.opacities_activation(model.opacities)) * opacity,
        )
    )
    if torch.isnan(new_opacities).any():
        raise ValueError("NaNs in new opacities.")
    params = replace_tensor_to_optimizer(optimizer, new_opacities, "opacities")
    model.opacities = params["opacities"]
