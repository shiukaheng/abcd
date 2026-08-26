from typing import List, Tuple

import torch

from abcd.geometry.bounding_box import BoundingBox
from abcd.geometry.plane import Plane


class Frustum:
    """
    A viewing frustum, represented by six bounding planes.
    """

    def __init__(
        self,
        left: Plane,
        right: Plane,
        top: Plane,
        bottom: Plane,
        near: Plane,
        far: Plane,
    ):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.near = near
        self.far = far

    def contains_point(self, point: torch.Tensor) -> bool:
        """
        Check if a point is inside the frustum.
        """
        return bool(
            self.left.signed_distance(point) >= 0
            and self.right.signed_distance(point) >= 0
            and self.top.signed_distance(point) >= 0
            and self.bottom.signed_distance(point) >= 0
            and self.near.signed_distance(point) >= 0
            and self.far.signed_distance(point) >= 0
        )

    def intersects_bounding_box(self, box: BoundingBox) -> bool:
        planes = [self.left, self.right, self.top, self.bottom, self.near, self.far]

        for plane in planes:
            p_vertex, _ = box.get_opposing_vertices(plane.normal)
            if plane.signed_distance(p_vertex) < 0:
                return False

        return True

    def get_corners(self):
        """
        Calculate the intersection points of the frustum planes, which form its corners.
        """
        # List of corners, each is the intersection of three planes
        corners = []
        plane_combinations = [
            (self.near, self.top, self.right),
            (self.near, self.top, self.left),
            (self.near, self.bottom, self.right),
            (self.near, self.bottom, self.left),
            (self.far, self.top, self.right),
            (self.far, self.top, self.left),
            (self.far, self.bottom, self.right),
            (self.far, self.bottom, self.left),
        ]

        for planes in plane_combinations:
            A = torch.stack([tuple(p.normal) for p in planes])
            b = torch.tensor([-float(p.d) for p in planes])
            corner = torch.linalg.solve(A, b)
            corners.append(tuple(corner))

        return corners

    def get_edges(
        self,
    ) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Using the corners of the frustum, define the edges.
        """
        corners = self.get_corners()
        # Edges defined by pairs of corner indices
        edge_indices = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),  # Near plane
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),  # Far plane
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Connecting edges
        ]
        edges = [
            (tuple(corners[start]), tuple(corners[end])) for start, end in edge_indices
        ]
        return edges

    def __str__(self) -> str:
        return (
            f"Frustum(left={self.left}, right={self.right}, top={self.top}, "
            + f"bottom={self.bottom}, near={self.near}, far={self.far})"
        )

    def to_projection_matrix(self) -> torch.Tensor:
        """
        Converts the frustum to a 4x4 projection matrix.
        """
        P = torch.zeros(4, 4)
        P[0] = (self.right.normal + self.left.normal) / (self.right.d + self.left.d)
        P[1] = (self.top.normal + self.bottom.normal) / (self.top.d + self.bottom.d)
        P[2] = (self.far.normal + self.near.normal) / (self.far.d + self.near.d)
        P[3] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        return P

    @staticmethod
    def from_projection_matrix(matrix: torch.Tensor) -> "Frustum":
        """
        Creates a frustum from a projection matrix.
        Assumes matrix is a 4x4 matrix combining camera intrinsics and extrinsics.
        """
        # Decompose the matrix into the six planes
        m = matrix
        planes = []

        # Define plane extraction according to typical graphics conventions
        indices = [
            (3, 0),  # left
            (3, 0),  # right
            (3, 1),  # bottom
            (3, 1),  # top
            (3, 2),  # near
            (3, 2),  # far
        ]
        signs = [
            -1,  # left
            +1,  # right
            -1,  # bottom
            +1,  # top
            +1,  # near
            -1,  # far
        ]

        for (c, i), sign in zip(indices, signs):
            if sign == -1:
                plane = m[:, c] - m[:, i]
            else:
                plane = m[:, c] + m[:, i]

            normal = plane[:3]
            norm = torch.norm(normal)
            normal /= norm
            d = plane[3] / norm

            planes.append(Plane(normal, d))

        return Frustum(*planes)
