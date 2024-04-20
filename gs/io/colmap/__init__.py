import os
from typing import List, Tuple
import warnings
import numpy as np
from tqdm import tqdm
from gs.io.colmap.COLMAPView import COLMAPView
from gs.helpers.image import pil_to_torch
from gs.helpers.transforms import qvec_to_rotmat
from gs.io.colmap.COLMAPPointCloud import COLMAPPointCloud
from gs.io.colmap.camera_parsing import get_fov, read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text
from PIL import Image

from gs.io.colmap.sparse_parsing import fetchPly, read_points3D_binary, read_points3D_text, storePly

"""
This module contains the main functions for loading COLMAP models.
"""

def load(path: str) -> Tuple[List[COLMAPView], COLMAPPointCloud]:
    '''
    Loads a COLMAP model from a path, returns (List[Camera], PointCloud)

    Expects folder structure:
    <path>/
        sparse/0/
            cameras.bin OR cameras.txt
            images.bin OR images.txt
            points3D.bin OR points3D.txt
        images/
            <image_name>.jpg
            ...
    '''
    cameras = load_cameras(path)
    sparse_points = load_sparse_points(path)
    return cameras, sparse_points

def load_cameras(path: str) -> List[COLMAPView]:
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        camera_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        camera_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception as e:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        camera_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        camera_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    # Now, we use the intermediate format to create Camera objects
    images_folder = os.path.join(path, "images")
    cameras = []

    pbar = tqdm(camera_extrinsics, desc="Loading cameras")
    for idx, key in enumerate(pbar):
        extrinsics = camera_extrinsics[key] # We first get the extrinsics
        intrinsics = camera_intrinsics[extrinsics.camera_id] # Then we get the intrinsics

        R = np.transpose(qvec_to_rotmat(extrinsics.qvec))
        t = np.array(extrinsics.tvec)

        fov_x, fov_y = get_fov(intrinsics)
        image_path = os.path.join(images_folder, os.path.basename(extrinsics.name))
        # replace .arw with .jpg (hack)
        image_path = image_path.replace(".arw", ".jpg").replace(".rd", ".jpg")
        with warnings.catch_warnings(): # Suppress EXIF warnings .. "Corrupt EXIF data"
            warnings.simplefilter("ignore")
            pil_image = Image.open(image_path)
        image = pil_to_torch(pil_image)
        
        # Point indexes = indexes of extrinsics.points3D_ids that are not -1
        point_indexes = np.where(extrinsics.point3D_ids != -1)[0]

        # Originally, we convert to CameraInfo, but this is so convoluted. Lets just directly convert to image
        camera = COLMAPView(
            R, t, fov_x, fov_y, pil_image.height, pil_image.width, idx, image, image_path, point_indexes
        )

        cameras.append(camera)
    pbar.close()
    return cameras

def load_sparse_points(path: str) -> COLMAPPointCloud:
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        # No .ply file. We will convert the .bin or .txt file to .ply
        try:
            xyz, rgb, errors, point3d_ids = read_points3D_binary(bin_path)
        except:
            xyz, rgb, errors, point3d_ids = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb, point3d_ids)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    return pcd