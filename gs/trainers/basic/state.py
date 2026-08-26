from dataclasses import dataclass, field

import torch


@dataclass
class BasicTrainState:
    """Serializable state needed to continue training one partition."""

    next_iteration: int = 0
    active_sh_degree: int | None = None
    optimizer: dict[str, dict[str, object]] = field(default_factory=dict)
    densification_stopped_for_memory: bool = False
    densification_stopped_for_count: bool = False

    def restore_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        groups = {group["name"]: group for group in optimizer.param_groups}
        unknown = set(self.optimizer) - set(groups)
        if unknown:
            raise ValueError(f"Unknown optimizer groups in training state: {unknown}")

        for name, saved_state in self.optimizer.items():
            parameter = groups[name]["params"][0]
            restored = {}
            for key, value in saved_state.items():
                if isinstance(value, torch.Tensor):
                    if value.ndim > 0 and value.shape != parameter.shape:
                        raise ValueError(
                            f"Optimizer state {name}.{key} has shape {value.shape}; "
                            f"expected {parameter.shape}"
                        )
                    target_device = parameter.device if value.ndim > 0 else value.device
                    target_dtype = parameter.dtype if value.ndim > 0 else value.dtype
                    restored[key] = value.to(
                        device=target_device, dtype=target_dtype
                    ).clone()
                else:
                    restored[key] = value
            optimizer.state[parameter] = restored

    def capture_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        captured = {}
        for group in optimizer.param_groups:
            parameter = group["params"][0]
            state = optimizer.state.get(parameter)
            if not state:
                continue
            captured[group["name"]] = {
                key: value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in state.items()
            }
        self.optimizer = captured

    def subset(self, mask: torch.Tensor) -> None:
        """Apply a Gaussian-row mask to all row-shaped optimizer tensors."""

        cpu_mask = mask.detach().cpu()
        row_count = cpu_mask.numel()
        for group_state in self.optimizer.values():
            for key, value in list(group_state.items()):
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == row_count
                ):
                    group_state[key] = value[cpu_mask].clone()
