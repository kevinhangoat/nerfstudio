#!/usr/bin/env python
"""Processes a video or image sequence to a nerfstudio compatible dataset."""

import json
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Tuple
from dataclasses import dataclass
from sklearn.neighbors import KDTree
import appdirs
import numpy as np
import requests
import tyro
from rich.console import Console
from rich.progress import track
import pdb
from nerfstudio.process_data.process_data_utils import CameraModel
from nerfstudio.process_data.colmap_utils import (
    read_cameras_binary, read_images_binary, qvec2rotmat)

@dataclass
class ColmapCameraModel:
    """Camera model"""

    model_id: int
    """Model identifier"""
    model_name: str
    """Model name"""
    num_params: int
    """Number of parameters"""

COLMAP_CAMERA_MODELS = [
    ColmapCameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    ColmapCameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    ColmapCameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    ColmapCameraModel(model_id=3, model_name="RADIAL", num_params=5),
    ColmapCameraModel(model_id=4, model_name="OPENCV", num_params=8),
    ColmapCameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    ColmapCameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    ColmapCameraModel(model_id=7, model_name="FOV", num_params=5),
    ColmapCameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    ColmapCameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    ColmapCameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
]

class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"


CAMERA_MODELS = {
    "perspective": CameraModel.OPENCV,
    "fisheye": CameraModel.OPENCV_FISHEYE,
    "other": None
}


def images_selection(images, image_center):
    """Generate a dictionary of images for easier post processing.

    output:
    images_dict: key is the image name, value is a dict containing id, tvec,
    neighbors, depth_map and depth_map_nbr.
    """
    images_list = []
    images_dict = {}
    tvec_list = []
    center_index = 0
    for _, value in images.items():
        images_list.append(value)
        tvec_list.append(value.tvec)
        if image_center == value.name:
            center_index = len(images_list) - 1
    tvec_array = np.array(tvec_list)
    kdt = KDTree(tvec_array, leaf_size=30, metric="euclidean")
    dists, indices = kdt.query(tvec_array, k=50, return_distance=True)
    
    selected_images = {}
    for i in indices[center_index]:
        selected_images[i] = images_list[i]
    return selected_images


def colmap_to_json(
    cameras_path: Path,
    images_path: Path,
    output_dir: Path,
    camera_model: CameraModel,
    camera_mask_path: Optional[Path] = None,
) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        cameras_path: Path to the cameras.bin file.
        images_path: Path to the images.bin file.
        output_dir: Path to the output directory.
        camera_mask_path: Path to the camera mask.
        camera_model: Camera model used.

    Returns:
        The number of registered images.
    """

    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)

    # Only supports one camera
    camera_params = cameras[1].params

    frames = []

    dynamic_mask_dir = output_dir / "dynamic_mask_dilated"
    depth_maps_processed_dir = output_dir / "depth_maps_nbr_consistent"
    semantic_dir = output_dir / "pan_mask/semantics"
    for _, im_data in images.items():
        if im_data.name[:17] == "fac2865d-c461dd89":
            if int(im_data.name[18:25]) <= 457:
                pass
            else:
                continue
        if im_data.name[:17] == "109fc76c-4981de2c":
            if 331 <= int(im_data.name[18:25]) <= 793:
                pass
            else:
                continue
        rotation = qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system to ours
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = Path(f"./images/{im_data.name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
        }
        if camera_mask_path is not None:
            frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()
        image_name = im_data.name
        png_name = image_name.replace(".jpg", ".png")
        frame["mask_path"] = (dynamic_mask_dir / png_name).as_posix()
        frame["depth_file_path"] = (depth_maps_processed_dir / png_name).as_posix()
        frame["semantics_file_path"] = (semantic_dir / png_name).as_posix()
        frames.append(frame)
    out = {
        "fl_x": float(camera_params[0]),
        "fl_y": float(camera_params[0]),
        "cx": float(camera_params[1]),
        "cy": float(camera_params[2]),
        "w": cameras[1].width,
        "h": cameras[1].height,
    }

    if camera_model == CameraModel.OPENCV:
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "p1": float(camera_params[6]),
                "p2": float(camera_params[7]),
            }
        )
    if camera_model == CameraModel.OPENCV_FISHEYE:
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "k3": float(camera_params[6]),
                "k4": float(camera_params[7]),
            }
        )

    out["frames"] = frames
    out["panoptic_classes"] = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/panoptic_classes.json"
    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)

def main():
    output_dir = Path("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/95115_11/")
    colmap_dir = output_dir
    camera_type = "other"
    camera_model = CAMERA_MODELS[camera_type]
    num_matched_frames = colmap_to_json(
        cameras_path=colmap_dir / "sparse" / "cameras.bin",
        images_path=colmap_dir / "sparse" / "images.bin",
        output_dir=output_dir,
        camera_model=camera_model,
    )

if __name__ == "__main__":
    main()