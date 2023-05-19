#!/usr/bin/env python

import os
import json
import sys
import pdb
import numpy as np
import torch
import tyro
from PIL import Image
import argparse
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
import pylab as plt
import cv2

"""
python scripts/test.py --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/outputs/-scratch_net-biwidl208-yuthan-wisp_data-nerf-360_v2-garden/nerfacto/2022-10-30_174811/config.yml \
    --data-name d389c316-c71f7a5e \
    --save-name nerfacto
"""

MY_RGB = np.array([
    [0,0,0],[255,255,128],[147,211,203],[150,100,100],[168,171,172],[146,112,198],[210,170,100],
    [225,199,255],[218,88,184],[241,129,0],[217,17,255],[124,74,181],[70,70,70],[255,228,255],[154,208,0],
    [193,0,92],[76,91,113],[255,180,195],[106,154,176],[230,150,140],[60,143,255],[128,64,128],[92,82,55],
    [254,212,124],[73,77,174],[255,160,98],[255,255,255],[104,84,109],[169,164,131],[92,136,89],
    [137,54,74],[135,158,223],[7,246,231],[107,255,200],[58,41,149],[183,121,142],[255,73,97],
    [107,142,35],[190,153,153],[146,139,141],[70,130,180]
]) / 255
MT_SEMANTICS = [
    "unlabeled","dynamic","ego vehicle","ground","static","parking","rail track","road","sidewalk",
    "bridge","building","fence","garage","guard rail","tunnel","wall","banner","billboard","lane divider",
    "parking sign","pole","polegroup","street light","traffic cone","traffic device","traffic light",
    "traffic sign","traffic sign frame","terrain","vegetation","sky","person","rider","bicycle","bus",
    "car","caravan","motorcycle","trailer","train","truck"
  ]
def plot_label(
    label: np.ndarray,
    image: str,
    save_path: str = "",
    title: str = "",
    visualize: bool = True,
) -> None:
    """Visualize depth map"""
    norm = colors.Normalize(vmin=0, vmax=40)
    my_cmap = ListedColormap(MY_RGB, name='my_cmap')

    # fig, ax = plt.subplots(figsize=(30, 20))
    plt.figure(figsize=(30, 20))
    plt.imshow(image)
    plt.imshow(label, cmap=my_cmap, norm=norm, alpha=0.65)
    
    cbar = plt.colorbar()

    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(MT_SEMANTICS):
        cbar.ax.text(2, (7.83 * j ) / 8.0 + 0.4, lab, ha='center', va='center')
    cbar.ax.get_yaxis().labelpad = 15
    
    plt.title(title)
    
    if visualize:
        plt.show()
    else:
        plt.imsave(save_path, depth_map_visual, cmap=cmap)
    pdb.set_trace()
    plt.close()
    

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
    # semantics = np.load("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs/11216_3/semantic-nerfacto/2023-02-15_120959/semantics.npy")
    # semantic_max = np.zeros((720, 1280))
    # for i in range(len(semantics)):
    #     for j in range(len(semantics[0])):
    #         semantic_max[i][j] = np.argmax(semantics[i][j])
    # plot_label(semantic_max)
    
    # filepath = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_overlap/ego_mask_new/d5fe7b01-bff75d31-0000607.png"
    # masks_path = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/94102_1/dynamic_mask"
    # images_path = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/94102_1/blocks/1/images"
    # for image_name in os.listdir(images_path):
    #     image_path = os.path.join(images_path, image_name)
    #     mask_path = os.path.join(masks_path, image_name.replace('jpg', 'png'))
    #     pil_image = Image.open(mask_path)
    #     mask_array = np.array(pil_image, dtype="int64")
    #     plot_depth(mask_array)
    # pdb.set_trace()
    
    filepath = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/KITTI-360-raw-single/depth_priors_lidar/0000003380.png"
    pil_image = Image.open(filepath)
    depth = np.array(pil_image, dtype="int64") /256
    # pdb.set_trace()
    
    
    # img = Image.open("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/94102_1/dynamic_mask_dilated/a85df27e-a31526ad-0000847.png")
    # mask = torch.from_numpy(np.array(img, dtype="int64"))[..., None]
    # semantics = torch.load(filepath)
    # plot_depth(mask)
    pdb.set_trace()