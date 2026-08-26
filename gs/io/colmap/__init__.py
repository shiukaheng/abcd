import os
import struct
import warnings
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from gs.helpers.image import pil_to_torch
from gs.helpers.transforms import qvec_to_rotmat
from gs.io.colmap.camera_parsing import (
    get_fov,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from gs.io.colmap.COLMAPPointCloud import COLMAPPointCloud
from gs.io.colmap.COLMAPView import COLMAPView
from gs.io.colmap.sparse_parsing import (
    fetchPly,
    read_points3D_binary,
    read_points3D_text,
    storePly,
)
from gs.profiling import log_tensor_set

"""
This module contains the main functions for loading COLMAP models.
"""


def load(
    path: str, images_subdir: str = "images"
) -> Tuple[List[COLMAPView], COLMAPPointCloud]:
    """
    Loads a COLMAP model from a path, returns (List[Camera], PointCloud)

    Expects folder structure:
    <path>/
        sparse/0/
            cameras.bin OR cameras.txt
            images.bin OR images.txt
            points3D.bin OR points3D.txt
        <images_subdir>/
            <image_name>.jpg
            ...
    """
    cameras = load_cameras(path, images_subdir)
    sparse_points = load_sparse_points(path)
    return cameras, sparse_points


def load_cameras(path: str, images_subdir: str = "images") -> List[COLMAPView]:
    binary_extrinsics = os.path.join(path, "sparse/0", "images.bin")
    binary_intrinsics = os.path.join(path, "sparse/0", "cameras.bin")
    text_extrinsics = os.path.join(path, "sparse/0", "images.txt")
    text_intrinsics = os.path.join(path, "sparse/0", "cameras.txt")
    if os.path.exists(binary_extrinsics) and os.path.exists(binary_intrinsics):
        try:
            camera_extrinsics = read_extrinsics_binary(binary_extrinsics)
            camera_intrinsics = read_intrinsics_binary(binary_intrinsics)
        except (OSError, ValueError, struct.error) as error:
            raise ValueError(f"Failed to read binary COLMAP model at {path}") from error
    elif os.path.exists(text_extrinsics) and os.path.exists(text_intrinsics):
        try:
            camera_extrinsics = read_extrinsics_text(text_extrinsics)
            camera_intrinsics = read_intrinsics_text(text_intrinsics)
        except (OSError, ValueError) as error:
            raise ValueError(f"Failed to read text COLMAP model at {path}") from error
    else:
        raise FileNotFoundError(
            f"No complete COLMAP camera model found under {path}/sparse/0"
        )

    # Now, we use the intermediate format to create Camera objects
    images_folder = os.path.join(path, images_subdir)
    cameras = []

    pbar = tqdm(camera_extrinsics, desc="Loading cameras")
    for idx, key in enumerate(pbar):
        extrinsics = camera_extrinsics[key]  # We first get the extrinsics
        intrinsics = camera_intrinsics[
            extrinsics.camera_id
        ]  # Then we get the intrinsics

        R = np.transpose(qvec_to_rotmat(extrinsics.qvec))
        t = np.array(extrinsics.tvec)

        fov_x, fov_y = get_fov(intrinsics)
        image_path = os.path.join(images_folder, os.path.basename(extrinsics.name))
        # replace .arw/.rd with .jpg, and handle case where .JPG needs to become .jpg
        # but the file actually exists with uppercase extension
        image_path_original = image_path
        image_path = image_path.replace(".arw", ".jpg").replace(".rd", ".jpg")
        # If lowercase path doesn't exist, try the original (handles .JPG files)
        if not os.path.exists(image_path):
            image_path = image_path_original
        with warnings.catch_warnings(), Image.open(image_path) as pil_image:
            warnings.simplefilter("ignore")
            image = pil_to_torch(pil_image)
            image_height = pil_image.height
            image_width = pil_image.width
        log_tensor_set(f"cam_{idx}.image", image, role="buffer")

        # Point indexes = indexes of extrinsics.points3D_ids that are not -1
        point_indexes = np.where(extrinsics.point3D_ids != -1)[0]

        # Originally, we convert to CameraInfo, but this is so convoluted. Lets just directly convert to image
        camera = COLMAPView(
            R,
            t,
            fov_x,
            fov_y,
            image_height,
            image_width,
            idx,
            image,
            image_path,
            point_indexes,
        )

        cameras.append(camera)
    pbar.close()
    return cameras


def load_sparse_points(path: str) -> COLMAPPointCloud:
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        if os.path.exists(bin_path):
            xyz, rgb, _, point3d_ids = read_points3D_binary(bin_path)
        elif os.path.exists(txt_path):
            xyz, rgb, _, point3d_ids = read_points3D_text(txt_path)
        else:
            raise FileNotFoundError(
                f"No COLMAP sparse points found at {bin_path} or {txt_path}"
            )
        storePly(ply_path, xyz, rgb, point3d_ids)
    try:
        pcd = fetchPly(ply_path)
    except Exception as error:
        raise ValueError(f"Failed to read COLMAP point cloud {ply_path}") from error
    return pcd
