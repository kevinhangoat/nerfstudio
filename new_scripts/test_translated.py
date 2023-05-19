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
from typing import Any, Dict, Optional, Tuple
from typing_extensions import assert_never
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
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

def get_translated_path(
    camera: Cameras,
    steps: int = 30,
    rots: int = 2,
) -> Cameras:
    """
    Returns a list of camera in a spiral trajectory.

    Args:
        camera: The camera to start the spiral from.
        steps: The number of cameras in the generated path.
        rots: The number of rotations to apply to the camera.

    Returns:
        A spiral camera path.
    """

    new_c2ws = []
    for local_c2wh in camera.camera_to_worlds:
        local_c2wh[2,3] += 0.03
        # c2wh = torch.matmul(c2wh_global, local_c2wh)
        new_c2ws.append(local_c2wh)
    new_c2ws = torch.stack(new_c2ws, dim=0)

    times = None
    if camera.times is not None:
        times = torch.linspace(0, 1, steps)[:, None]
    return Cameras(
        fx=camera.fx[0],
        fy=camera.fy[0],
        cx=camera.cx[0],
        cy=camera.cy[0],
        camera_to_worlds=new_c2ws,
        times=times,
    )

def get_interpolated_path(
    camera: Cameras,
    steps: int = 30,
    fps: int = 60,
) -> Cameras:
    """
    Returns a list of camera in a spiral trajectory.

    Args:
        camera: The camera to start the spiral from.
        steps: The number of cameras in the generated path.
        rots: The number of rotations to apply to the camera.

    Returns:
        A spiral camera path.
    """
    num_interpolated = fps / 5
    new_c2ws = []
    for idx, local_c2wh in enumerate(camera.camera_to_worlds):
        new_c2ws.append(local_c2wh)
        
        if idx < len(camera.camera_to_worlds) - 1:
            interpolated_frame = local_c2wh.clone()
            diff = camera.camera_to_worlds[idx+1] - camera.camera_to_worlds[idx]
            diff[:3,:3] = torch.zeros(3,3)
            for i in range(int(num_interpolated) - 1 ): 
                new_c2ws.append(interpolated_frame + (diff / num_interpolated) * float(i + 1))
    new_c2ws = torch.stack(new_c2ws, dim=0)

    times = None
    if camera.times is not None:
        times = torch.linspace(0, 1, steps)[:, None]
    return Cameras(
        fx=camera.fx[0],
        fy=camera.fy[0],
        cx=camera.cx[0],
        cy=camera.cy[0],
        camera_to_worlds=new_c2ws,
        times=times,
    )

def _render_results(
    pipeline: Pipeline,
    data,
    output_directory: Path,
    rendered_resolution_scaling_factor: float = 1.0,
) -> None:
    """Helper function to create a video of the spiral trajectory.
    Args:
        pipeline: Pipeline to evaluate with.
        data: Cameras to render.
        output_directory: Name of the output file.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
    """
    cameras = get_interpolated_path(data.cameras, steps=30)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    
    image_filenames = data.image_filenames
    for camera_idx in range(cameras.size):
        camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx).to(pipeline.device)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        rgb = outputs["rgb"].cpu().numpy() * 255
        rgb = Image.fromarray(rgb.astype(np.uint8))
        filename = str(image_filenames[int(camera_idx // 12)].name)[:-4] + f"_{str(int(12 - camera_idx % 12))}"
        rgb.save(f"{output_directory}/rgb_{filename}.jpeg")
        
        depth = np.squeeze((outputs["depth"].cpu().numpy() * 256)).astype(np.uint16)
        min_depth, max_depth = np.percentile(
            depth, [5, 95]
        )
        depth[depth < min_depth] = min_depth
        depth[depth > max_depth] = max_depth
        # np.save(f"{output_directory}/depth_{filename}.npy", depth)
        plot_depth(
            depth,
            f"{output_directory}/vis_depth_{filename}.png",
            camera_idx,
            False,
            vmax=150
        )
        if "semantics_colormap" in outputs:
            semantic = outputs["semantics_colormap"].cpu().numpy() * 255
            semantic = Image.fromarray(semantic.astype(np.uint8))
            semantic.save(f"{output_directory}/semantic_{filename}.jpeg")
        # cv2.imwrite(
        #     f"{output_directory}/depth_{camera_idx}.png",
        #     depth,
        # )
        # pdb.set_trace()

def plot_depth(
    depth_map,
    save_path: str = "",
    title: str = "",
    visualize: bool = True,
    vmin: float = 0,
    vmax: float = 80,
) -> None:
    """Visualize depth map."""
    mask = np.logical_or((depth_map == vmin), (depth_map > vmax))
    depth_map_visual = np.ma.masked_where(mask, depth_map)
    cmap = cm.viridis
    cmap.set_bad(color="gray")
    plt.figure(figsize=(30, 20))
    plt.imshow(depth_map_visual, cmap=cmap, vmin=vmin, vmax=vmax)
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
        default="/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data",
        help="Base path",
    )
    parser.add_argument(
        "--load-config",
        "-c",
        type=str,
        help="config file",
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
    render_path = os.path.join(args.base_path, "renders")
    os.makedirs(render_path, exist_ok=True)
    output_directory = Path(render_path, args.save_name)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    _, pipeline, _ = eval_setup(
        Path(load_config),
        eval_num_rays_per_chunk=eval_num_rays_per_chunk,
    )
    

    _render_results(
        pipeline,
        pipeline.datamanager.train_dataparser_outputs,
        output_directory,
    )