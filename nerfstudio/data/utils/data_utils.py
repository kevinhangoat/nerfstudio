# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions to allow easy re-use of common operations across dataloaders"""
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
import open3d as o3d
from sklearn.neighbors import KDTree
import pdb

def get_image_mask_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
    if len(mask_tensor.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return mask_tensor


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype="int64").view(1, 1, -1)
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.NEAREST)
    semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [width, height, 1].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        if "image" not in locals() or image is None:
            image = np.zeros((height, width))
        else:
            image = image.astype(np.float64) * scale_factor
            image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image[:, :, np.newaxis])

def get_points3d_from_depth_batch(
    depth_img,
    camera_params,
    extrinsics_mat,
):

    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float64) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image[:, :, np.newaxis])

def depth_to_pcd(
    depth_img,
    camera_params,
    extrinsics_mat,
):
    """Map from 2D depth map to 3D point cloud."""
    f = camera_params[0]
    cx = camera_params[2]
    cy = camera_params[3]
    width = depth_img.shape[1]
    height = depth_img.shape[0]
    xw = np.tile(list(range(width)), (height, 1)) - cx
    yw = np.tile(list(range(height)), (width, 1)).T - cy
    point_x = (xw * depth_img).reshape(width * height) / f
    point_y = (yw * depth_img).reshape(width * height) / f
    point_z = depth_img.reshape(width * height)
    point = np.stack((point_x, point_y, point_z))
    point = point[:, ~np.all(point == 0, axis=0)]
    point = np.vstack((point, np.ones(point.shape[1])))
    pcd_array = np.array(np.matmul(np.linalg.inv(extrinsics_mat), point))
    return pcd_array


def fit_plane(pcd_array, prior_plane_model=[]):
    """Use ransac to fit the points of floor/ground to a plane.

    This is used to avoid noise in depth map for the ground.
    """
    pcd_array = np.delete(pcd_array, 3, 0)
    pcd_array = pcd_array.T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    if len(pcd_array) < 10:
        # If there are less than 10 pixels of ground
        # remove all of them since ransac cannot work well
        return prior_plane_model
    plane_model, _ = pcd.segment_plane(
        distance_threshold=1, 
        ransac_n=3, 
        num_iterations=1000,
    )

    return plane_model
