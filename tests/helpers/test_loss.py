import torch

import abcd.helpers.loss as losses


class FakeLpips:
    def __init__(self):
        self.inputs = None

    def __call__(self, predicted, target):
        self.inputs = predicted, target
        return torch.tensor([2.0])


def test_lpips_adds_batch_dimension_and_normalizes_to_signed_range():
    fake = FakeLpips()
    previous = losses.lpips_func
    losses.lpips_func = fake
    try:
        value = losses.lpips_loss(torch.ones((3, 2, 2)), torch.zeros((3, 2, 2)))
    finally:
        losses.lpips_func = previous

    predicted, target = fake.inputs
    assert predicted.shape == (1, 3, 2, 2)
    assert target.shape == (1, 3, 2, 2)
    assert torch.all(predicted == 1)
    assert torch.all(target == -1)
    assert value.item() == 2
