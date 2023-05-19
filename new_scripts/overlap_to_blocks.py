#!/usr/bin/env python
"""Processes a video or image sequence to a nerfstudio compatible dataset."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Tuple
from dataclasses import dataclass
from sklearn.neighbors import KDTree
import numpy as np
import requests
import tyro
import cv2
from rich.console import Console
from rich.progress import track
import pdb
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from nerfstudio.process_data.process_data_utils import CameraModel
from nerfstudio.process_data.colmap_utils import (
    read_cameras_binary, read_images_binary, qvec2rotmat)
from create_dynamic_mask import create_pan_mask_dict, get_pan_mask
from tqdm import tqdm

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

def check_exposure(img_path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, std_dev = cv2.meanStdDev(gray)

    # Check if the standard deviation is too low or too high
    
    if std_dev < 30 or std_dev > 150 or mean < 60 or mean > 160:
        return False
    else:
        return True


def image_centers_blocks_selection(images, image_centers):

    images_list = []
    tvec_list = []
    center_indices = []

    for _, value in images.items():
        images_list.append(value)
        tvec_list.append(value.tvec)
        if value.name in image_centers:
            center_indices.append(len(images_list) - 1)

    tvec_array = np.array(tvec_list)
    kdt = KDTree(tvec_array, leaf_size=30, metric="euclidean")
    dists, indices = kdt.query(tvec_array, k=50, return_distance=True)
    image_indices = kdt.query_radius(tvec_array, r=50)
    
    blocks = []
    for i in center_indices:
        block = []
        for image_index in image_indices[i]:
            block.append(images_list[image_index])
        blocks.append(block)
    return blocks


def build_cuboid_blocks(images):

    images_list = []
    tvec_list = []

    for _, value in images.items():
        images_list.append(value)
        tvec_list.append(value.tvec)
    
    tvec_array = np.asarray(tvec_list)

    max_coords = np.max(tvec_array, axis=0) + 5
    min_coords = np.min(tvec_array, axis=0) - 5
    cube_size = np.array([80,max_coords[1]-min_coords[1],80])

    n_cubes = np.ceil((max_coords - min_coords) / cube_size).astype(int)
    indices = np.indices(n_cubes).reshape(3, -1).T
    centers = min_coords + cube_size / 2 + indices * cube_size

    centers_array = np.asarray(centers)
    kdt = KDTree(tvec_array, leaf_size=30, metric="euclidean")
    image_indices = kdt.query_radius(centers_array, r=50)

    blocks = []
    for i in range(len(centers)):
        block = []
        for image_index in image_indices[i]:
            block.append(images_list[image_index])
        blocks.append(block)
    
    kdt_centers = KDTree(centers_array, leaf_size=30, metric="euclidean")
    
    for block in blocks:
        if 0 < len(block) < 20:
            cur_tvec_list = []
            for img in block:
                cur_tvec_list.append(img.tvec)
            cur_tvec_array = np.asarray(cur_tvec_list)
            _, cur_center_indices = kdt_centers.query(cur_tvec_array, k=3)
            for i, _ in enumerate(block):
                second_closest = cur_center_indices[i][1]
                if len(blocks[second_closest]) > 20:
                    blocks[second_closest].append(block[i])
    
    final_blocks = []
    for block in blocks:
        if len(block) > 20:
            final_blocks.append(block)
    return final_blocks


def colmap_to_blocks_json(
    cameras_path: Path,
    images_path: Path,
    output_dir: Path,
    camera_model: CameraModel,
) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        cameras_path: Path to the cameras.bin file.
        images_path: Path to the images.bin file.
        output_dir: Path to the output directory.

    Returns:
        The number of registered images.
    """

    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)

    # Only supports one camera
    camera_params = cameras[1].params

    selected_blocks = build_cuboid_blocks(images)
    images_dir = output_dir / "images"
    blocks_dir = output_dir / "blocks"
    blocks_dir.mkdir(exist_ok=True)

    dynamic_mask_dir = output_dir / "dynamic_mask_dilated"
    depth_maps_processed_dir = output_dir / "depth_maps_nbr_consistent"
    semantic_dir = output_dir / "pan_mask/semantics"
    for block_ind, block in enumerate(selected_blocks):
        block_dir = blocks_dir / str(block_ind)
        block_dir.mkdir(exist_ok=True)
        frames = []
        for ind, im_data in enumerate(block):
            image_name = im_data.name

            block_images_dir = block_dir / "images"
            block_images_dir.mkdir(exist_ok=True)

            block_removed_images_dir = block_dir / "images_removed"
            block_removed_images_dir.mkdir(exist_ok=True)

            if check_exposure(images_dir / image_name):
                os.system(f"cp {images_dir / image_name} {block_images_dir}")
            else:
                os.system(f"cp {images_dir / image_name} {block_removed_images_dir}")

            rotation = qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system to ours
            c2w[0:3, 1:3] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[2, :] *= -1

            
            frame = {
                "file_path": (images_dir / image_name).as_posix(),
                "transform_matrix": c2w.tolist(),
            }
            png_name = image_name.replace(".jpg", ".png")
            frame["mask_path"] = (dynamic_mask_dir / png_name).as_posix()
            frame["depth_file_path"] = (depth_maps_processed_dir / png_name).as_posix()
            frame["semantics_file_path"] = (semantic_dir / png_name).as_posix()
            frames.append(frame)
        if len(camera_params) == 3:
            fx, cx, cy = camera_params
            fy = fx
        else:
            fx, fy, cx, cy = camera_params

        out = {
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
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
        with open(block_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)    
    

def main():
    parser = argparse.ArgumentParser(
        description="Turn bdd100k overlaps into blocks for nerf"
    )
    parser.add_argument(
        "--dense_path",
        type=str,
        help="Dense path",
    )
    args = parser.parse_args()

    DESTINATION_PATH = Path("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data")

    dense_path = args.dense_path
    recon_name = dense_path.split('/')[-3]
    postcode = dense_path.split('/')[-6]
    blocks_name = postcode + "_" + recon_name

    new_folder_path = DESTINATION_PATH / blocks_name
    new_folder_path.mkdir(exist_ok=True)
        

    original_images_path = Path(dense_path) / "images"
    if not (DESTINATION_PATH / "images").exists():
        os.system(f"cp -r {original_images_path} {new_folder_path}")

    original_pan_mask_path = Path(dense_path).parents[1] / "pan_mask"
    if not (DESTINATION_PATH / "pan_mask").exists():
        os.system(f"cp -r {original_pan_mask_path} {new_folder_path}")
    json_path = new_folder_path / "pan_mask" / "pan_seg.json"
    pan_masks_dict = create_pan_mask_dict(json_path)
    semantics_save_path = new_folder_path / "pan_mask" / "semantics"
    semantics_save_path.mkdir(exist_ok=True)
    for img in tqdm(pan_masks_dict):
        semantics = pan_masks_dict[img]['total'][0]
        image_name_png = img.replace('jpg', 'png')
        semantics = semantics.astype(np.uint16)
        cv2.imwrite(
            os.path.join(str(semantics_save_path), image_name_png),
            semantics,
        )

    original_depth_maps_nbr_consistent_path = Path(dense_path) / "depth_maps_nbr_consistent"
    if not (DESTINATION_PATH / "depth_maps_nbr_consistent").exists():
        os.system(f"cp -r {original_depth_maps_nbr_consistent_path} {new_folder_path}")

    original_sparse_path = Path(dense_path) / "sparse"
    if not (DESTINATION_PATH / "sparse").exists():
        os.system(f"cp -r {original_sparse_path} {new_folder_path}")


    if not (new_folder_path / "dynamic_mask_dilated").exists():
        print("Create dynamic mask dilated")
        mask_save_path = str(new_folder_path / "dynamic_mask_dilated")
        (new_folder_path / "dynamic_mask_dilated").mkdir(exist_ok=True)
        for image_name in tqdm(pan_masks_dict):   
            pan_mask_dict = pan_masks_dict[image_name]
            shape = pan_mask_dict["total"][0].shape
            
            pan_mask = get_pan_mask(shape, pan_mask_dict)
            cur_mask_save_path = os.path.join(mask_save_path, image_name.replace("jpg", "png"))
            cv2.imwrite(cur_mask_save_path, pan_mask.astype(np.uint8))
    

    camera_type = "other"
    camera_model = CAMERA_MODELS[camera_type]

    colmap_to_blocks_json(
        cameras_path=new_folder_path / "sparse" / "cameras.bin",
        images_path=new_folder_path / "sparse" / "images.bin",
        output_dir=new_folder_path,
        camera_model=camera_model,
    )

if __name__ == "__main__":
    main()