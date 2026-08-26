import torch

from gs.geometry.bounding_box import BoundingBox
from gs.geometry.frustum import Frustum
from gs.geometry.plane import Plane


def box(minimum, maximum):
    return BoundingBox(
        torch.tensor(minimum, dtype=torch.float32),
        torch.tensor(maximum, dtype=torch.float32),
    )


def axis_aligned_frustum():
    return Frustum(
        left=Plane(torch.tensor([1.0, 0.0, 0.0]), 1.0),
        right=Plane(torch.tensor([-1.0, 0.0, 0.0]), 1.0),
        top=Plane(torch.tensor([0.0, -1.0, 0.0]), 1.0),
        bottom=Plane(torch.tensor([0.0, 1.0, 0.0]), 1.0),
        near=Plane(torch.tensor([0.0, 0.0, 1.0]), -1.0),
        far=Plane(torch.tensor([0.0, 0.0, -1.0]), 10.0),
    )


def test_frustum_intersects_inside_and_enclosing_boxes():
    frustum = axis_aligned_frustum()
    assert frustum.intersects_bounding_box(box([-0.5, -0.5, 2], [0.5, 0.5, 3]))
    assert frustum.intersects_bounding_box(box([-2, -2, 0], [2, 2, 11]))


def test_frustum_rejects_boxes_outside_planes():
    frustum = axis_aligned_frustum()
    assert not frustum.intersects_bounding_box(box([2, -0.5, 2], [3, 0.5, 3]))
    assert not frustum.intersects_bounding_box(box([-0.5, -0.5, 0], [0.5, 0.5, 0.5]))
    assert not frustum.intersects_bounding_box(box([-0.5, -0.5, -2], [0.5, 0.5, -1]))


def test_frustum_treats_tangent_boxes_as_visible():
    frustum = axis_aligned_frustum()
    assert frustum.intersects_bounding_box(box([1, -0.5, 2], [2, 0.5, 3]))
    assert frustum.intersects_bounding_box(box([-0.5, -0.5, 0], [0.5, 0.5, 1]))
