from typing import List
import numpy as np
import torch
from gs.core.View import KnownView


# class COLMAPView(KnownView):
#     def __init__(
#             self,
#             image_height: int,
#             image_width: int,
#             fov_x: float,
#             fov_y: float,
#             R: torch.Tensor,
#             t: torch.Tensor,
#             image: torch.Tensor,
#             image_path: str,
#             id: int,
#             point3d_ids: List[int],
#     ):
#         super().__init__(image_height, image_width, fov_x, fov_y, R, t, id, image)
#         self.image_path = image_path
#         self.point3d_ids = point3d_ids

class COLMAPView(KnownView[int]):
    def __init__(
            self,
            R: np.ndarray,
            t: np.ndarray,
            fov_x: float,
            fov_y: float,
            image_height: int,
            image_width: int,
            id: int,
            image: torch.Tensor,
            image_path: str,
            point3d_ids: List[int],
    ):
        super().__init__(R, t, fov_x, fov_y, image_height, image_width, id, image)
        self.image_path = image_path
        self.point3d_ids = point3d_ids