#!/usr/bin/env python
"""
render.py
"""

from nerfstudio.utils.io import load_from_json
import os
import json
import logging
import sys
from pathlib import Path
from typing import Literal, Optional
import pdb
import numpy as np
import torch
import tyro
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import assert_never
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn
from PIL import Image
import argparse
import matplotlib.cm as cm
import matplotlib.colors as colors
import pylab as plt
import cv2

"""
python scripts/test.py --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/outputs/-scratch_net-biwidl208-yuthan-wisp_data-nerf-360_v2-garden/nerfacto/2022-10-30_174811/config.yml \
    --data-name d389c316-c71f7a5e \
    --save-name nerfacto
"""
logging.basicConfig(format="[%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)

def plot_depth(
    depth_map: np.ndarray,
    save_path: str = "",
    title: str = "",
    visualize: bool = True,
):
    """Visualize depth map"""
    depth_map_visual = np.ma.masked_where(depth_map == 0, depth_map)
    cmap = cm.Blues_r
    cmap.set_bad(color="gray")
    plt.figure(figsize=(30, 20))
    plt.imshow(depth_map_visual, cmap=cmap)
    plt.colorbar()
    plt.title(title)

    if visualize:
        plt.show()
    else:
        plt.imsave(save_path, depth_map_visual, cmap=cmap)
    plt.close()
    
def _render_rgb(
    pipeline: Pipeline,
    cameras: Cameras,
    output_directory: Path,
    rendered_resolution_scaling_factor: float = 1.0,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_directory: Name of the output file.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
    """
    images = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    
    for camera_idx in range(cameras.size):
        camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx).to(pipeline.device)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        rgb = outputs["rgb"].cpu().numpy() * 255
        rgb = Image.fromarray(rgb.astype(np.uint8))
        rgb.save(f"{output_directory}/rgb_{camera_idx}.jpeg")
        
        depth = np.squeeze((outputs["depth"].cpu().numpy() * 100)).astype(np.uint16)
        min_depth, max_depth = np.percentile(
            depth, [5, 95]
        )
        depth[depth < min_depth] = min_depth
        depth[depth > max_depth] = max_depth
        plot_depth(
            depth,
            f"{output_directory}/vis_depth_{camera_idx}.png",
            camera_idx,
            False,
        )
        cv2.imwrite(
            f"{output_directory}/depth_{camera_idx}.png",
            depth,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="load configs"
    )
    parser.add_argument(
        "--base-path",
        "-p",
        type=str,
        default="/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/",
        help="Base path",
    )
    parser.add_argument(
        "--load-config",
        "-c",
        type=str,
        help="config file",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        help="data name",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        help="save name",
    )
    parser.add_argument(
        "--aabb-scale",
        type=str,
        default=16,
        help="aabb scale",
    )
    args = parser.parse_args()
    load_config = args.load_config
    json_path = os.path.join(args.base_path, "data", args.data_name, "transforms.json")
    eval_num_rays_per_chunk = 4000
    output_directory = Path(os.path.join(args.base_path, "renders", args.data_name, args.save_name))
    output_directory.mkdir(parents=True, exist_ok=True)
    aabb_scale = args.aabb_scale

    _, pipeline, _ = eval_setup(
        Path(load_config),
        eval_num_rays_per_chunk=eval_num_rays_per_chunk,
    )
    camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
    meta = load_from_json(Path(json_path))
    poses = []
    for frame in meta["frames"]:
        poses.append(np.array(frame["transform_matrix"]))
    poses = np.array(poses).astype(np.float32)
    camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform
    # pdb.set_trace()
    camera_to_world[:, :, 3] /= (aabb_scale)
    # camera_to_world[:, :, 3]
    if "k1" in meta:
        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta["k1"]), k2=float(meta["k2"]), p1=float(meta["p1"]), p2=float(meta["p2"])
        )
        cameras = Cameras(
            fx=float(meta["fl_x"]),
            fy=float(meta["fl_y"]),
            cx=float(meta["cx"]),
            cy=float(meta["cy"]),
            distortion_params=distortion_params,
            height=int(meta["h"]),
            width=int(meta["w"]),
            camera_to_worlds=camera_to_world,
            camera_type=CameraType.PERSPECTIVE,
        )
    else:
        cameras = Cameras(
            fx=camera_start.fx[0],
            fy=camera_start.fy[0],
            cx=camera_start.cx[0],
            cy=camera_start.cx[0],
            camera_to_worlds=camera_to_world,
            camera_type=CameraType.PERSPECTIVE,
        )
    _render_rgb(
        pipeline,
        cameras,
        output_directory,
    )
