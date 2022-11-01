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
        # pdb.set_trace()

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
        "--aabb-scale",
        type=str,
        default=16,
        help="aabb scale",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        help="save name",
    )
    args = parser.parse_args()
    load_config = args.load_config
    eval_num_rays_per_chunk = 4000
    output_directory = Path(os.path.join(args.base_path, "renders", args.data_name, args.save_name))
    output_directory.mkdir(parents=True, exist_ok=True)

    _, pipeline, _ = eval_setup(
        Path(load_config),
        eval_num_rays_per_chunk=eval_num_rays_per_chunk,
    )
    _render_rgb(
        pipeline,
        pipeline.datamanager.train_dataset.dataparser_outputs.cameras,
        output_directory,
    )
