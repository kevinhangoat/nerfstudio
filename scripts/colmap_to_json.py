#!/usr/bin/env python
"""Processes a video or image sequence to a nerfstudio compatible dataset."""

import json
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Tuple

import appdirs
import numpy as np
import requests
import tyro
from rich.console import Console
from rich.progress import track
import pdb
from nerfstudio.utils import colmap_utils, install_checks

class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"

CAMERA_MODELS = {
    "perspective": CameraModel.OPENCV,
    "fisheye": CameraModel.OPENCV_FISHEYE,
    "other": None
}

def colmap_to_json(cameras_path: Path, images_path: Path, output_dir: Path, camera_model: CameraModel) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        cameras_path: Path to the cameras.bin file.
        images_path: Path to the images.bin file.
        output_dir: Path to the output directory.
        camera_model: Camera model used.

    Returns:
        The number of registered images.
    """

    cameras = colmap_utils.read_cameras_binary(cameras_path)
    images = colmap_utils.read_images_binary(images_path)

    # Only supports one camera
    camera_params = cameras[1].params

    frames = []
    for _, im_data in images.items():
        rotation = colmap_utils.qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system to ours
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = Path(f"./images_8/{im_data.name}")

        frame = {
            "file_path": str(name),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)
    out = {
        "fl_x": float(camera_params[0]),
        "fl_y": float(camera_params[0]),
        "cx": float(camera_params[1]),
        "cy": float(camera_params[2]),
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
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

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)

def main():
    output_dir = Path("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/bicycle")
    colmap_dir = output_dir
    camera_type = "other"
    camera_model = CAMERA_MODELS[camera_type]
    num_matched_frames = colmap_to_json(
        cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
        images_path=colmap_dir / "sparse" / "0" / "images.bin",
        output_dir=output_dir,
        camera_model=camera_model,
    )

if __name__ == "__main__":
    main()