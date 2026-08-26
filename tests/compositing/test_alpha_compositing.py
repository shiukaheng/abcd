import pytest
import torch

from abcd.compositing.alpha_compositing import composite_images_rgbda


def layer(color, alpha, depth=1.0):
    alpha_tensor = torch.tensor([[[alpha]]], dtype=torch.float32)
    # Rasterizer RGB is premultiplied by alpha.
    rgb = torch.tensor(color, dtype=torch.float32).reshape(3, 1, 1) * alpha_tensor
    return rgb, torch.tensor([[[depth]]]), alpha_tensor


def test_front_layer_is_composited_over_back_layer():
    back = layer([0.0, 0.0, 1.0], 1.0, depth=2.0)
    front = layer([1.0, 0.0, 0.0], 0.25, depth=1.0)

    rgb, _, alpha = composite_images_rgbda([back, front])

    torch.testing.assert_close(rgb, torch.tensor([[[0.25]], [[0.0]], [[0.75]]]))
    torch.testing.assert_close(alpha, torch.ones_like(alpha))


def test_gradient_only_flows_through_active_layer_when_cache_is_detached():
    back = tuple(value.detach() for value in layer([0.0, 0.0, 1.0], 1.0))
    active_rgb = torch.tensor([[[0.4]], [[0.0]], [[0.0]]], requires_grad=True)
    active_alpha = torch.tensor([[[0.5]]])
    active = active_rgb, torch.ones((1, 1, 1)), active_alpha

    rgb, _, _ = composite_images_rgbda([back, active])
    rgb.sum().backward()

    assert active_rgb.grad is not None
    assert all(value.grad is None for value in back)


def test_empty_composite_is_rejected():
    with pytest.raises(ValueError, match="At least one"):
        composite_images_rgbda([])
