import math
from typing import Generic, TypeVar, Union
import numpy as np
import torch
import torch.nn as nn
from gs.geometry.bounding_box import BoundingBox
from gs.geometry.frustum import Frustum
from gs.helpers.transforms import get_projection_matrix, get_world_to_view

ZFAR = 100.0
ZNEAR = 0.01

class View(nn.Module):
    """
    Class to represent the extrinsics and intrinsics of a camera.
    """

    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    center: torch.Tensor # (3,)
    look_at: torch.Tensor # (3,)

    def __init__(
        self,
        R: np.ndarray,
        t: np.ndarray,
        fov_x: float,
        fov_y: float
    ):
        super().__init__()
        world_view_transform = torch.tensor(get_world_to_view(R, t)).transpose(0, 1)
        projection_matrix = get_projection_matrix(znear=ZNEAR, zfar=ZFAR, fovX=fov_x, fovY=fov_y).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        look_at = world_view_transform.inverse()[:3, 2]

        self.register_buffer("world_view_transform", world_view_transform)
        self.register_buffer("full_proj_transform", full_proj_transform)
        self.register_buffer("center", camera_center)
        self.register_buffer("look_at", look_at)

        self.R = R
        self.t = t
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.frustum = Frustum.from_projection_matrix(full_proj_transform)
        self.aspect_ratio = math.tan(fov_x / 2) / math.tan(fov_y / 2)

        # Assert that all tensors dont have nan, inf, or -inf
        assert not torch.isnan(self.world_view_transform).any()
        assert not torch.isnan(self.full_proj_transform).any()
        assert not torch.isnan(self.center).any()
        assert not torch.isnan(self.look_at).any()
        assert not torch.isinf(self.world_view_transform).any()
        assert not torch.isinf(self.full_proj_transform).any()
        assert not torch.isinf(self.center).any()
        assert not torch.isinf(self.look_at).any()


    def __repr__(self):
        return f"CameraPose(center={self.center}, look_at={self.look_at}), fov_x={self.fov_x}, fov_y={self.fov_y}"
    
    def __str__(self):
        return self.__repr__()

class ViewWithRes(View):
    """
    Class to represent the extrinsics and intrinsics of a camera with a defined resolution.
    """

    def __init__(
        self,
        R: np.ndarray,
        t: np.ndarray,
        fov_x: float,
        fov_y: float,
        image_height: int,
        image_width: int
    ):
        super().__init__(R, t, fov_x, fov_y)
        self.image_height = image_height
        self.image_width = image_width

    def __repr__(self):
        return f"CameraPoseWithResolution(center={self.center}, look_at={self.look_at}, image_height={self.image_height}, image_width={self.image_width})"
    
    def __str__(self):
        return self.__repr__()
    
T = TypeVar('T')
    
class KnownView(ViewWithRes, Generic[T]):
    """
    Class representing a known camera, with an image and id. ID will be of type T.
    """
    
    image: torch.Tensor

    def __init__(
        self,
        R: np.ndarray,
        t: np.ndarray,
        fov_x: float,
        fov_y: float,
        image_height: int,
        image_width: int,
        id: T,
        image: torch.Tensor
    ):
        super().__init__(R, t, fov_x, fov_y, image_height, image_width)
        self.id = id
        self.register_buffer("image", image)

    def __repr__(self):
        return f"KnownCamera(center={self.center}, look_at={self.look_at}, image_height={self.image_height}, image_width={self.image_width}, id={self.id})"
    
    def __str__(self):
        return self.__repr__()