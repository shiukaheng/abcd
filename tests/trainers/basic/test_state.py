import torch

from gs.trainers.basic.state import BasicTrainState


def make_optimizer(parameter):
    return torch.optim.Adam(
        [{"params": [parameter], "lr": 0.1, "name": "positions"}], eps=1e-15
    )


def take_step(parameter, optimizer):
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = 2 * parameter.detach()
    optimizer.step()


def test_optimizer_state_survives_new_parameter_objects():
    original = torch.nn.Parameter(torch.tensor([[1.0], [2.0]]))
    optimizer = make_optimizer(original)
    take_step(original, optimizer)

    state = BasicTrainState(next_iteration=1, active_sh_degree=1)
    state.capture_optimizer(optimizer)

    restored_parameter = torch.nn.Parameter(original.detach().clone())
    restored_optimizer = make_optimizer(restored_parameter)
    state.restore_optimizer(restored_optimizer)

    restored_state = restored_optimizer.state[restored_parameter]
    original_state = optimizer.state[original]
    torch.testing.assert_close(restored_state["exp_avg"], original_state["exp_avg"])
    torch.testing.assert_close(
        restored_state["exp_avg_sq"], original_state["exp_avg_sq"]
    )
    torch.testing.assert_close(restored_state["step"], original_state["step"])


def test_state_subset_keeps_matching_optimizer_rows():
    state = BasicTrainState(
        optimizer={
            "positions": {
                "step": torch.tensor(4.0),
                "exp_avg": torch.tensor([[1.0], [2.0], [3.0]]),
                "exp_avg_sq": torch.tensor([[10.0], [20.0], [30.0]]),
            }
        }
    )

    state.subset(torch.tensor([True, False, True]))

    torch.testing.assert_close(
        state.optimizer["positions"]["exp_avg"], torch.tensor([[1.0], [3.0]])
    )
    torch.testing.assert_close(
        state.optimizer["positions"]["exp_avg_sq"],
        torch.tensor([[10.0], [30.0]]),
    )
    assert state.optimizer["positions"]["step"].item() == 4
