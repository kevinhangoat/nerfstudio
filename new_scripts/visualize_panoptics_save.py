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
from tqdm import tqdm
import pdb

def load_panoptics_class_from_json(pan_class_path):
    with open(pan_class_path, encoding="UTF-8") as file:
        content = json.load(file)
    return content['semantics_colors'], content['semantics']

MY_RGB, MY_SEMANTICS = load_panoptics_class_from_json("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/panoptic_classes.json")
MY_RGB = np.array(MY_RGB) / 255
# MY_RGB = np.array([
#     [0,0,0],[255,255,128],[147,211,203],[150,100,100],[168,171,172],[146,112,198],[210,170,100],
#     [225,199,255],[218,88,184],[241,129,0],[217,17,255],[124,74,181],[70,70,70],[255,228,255],[154,208,0],
#     [193,0,92],[76,91,113],[255,180,195],[106,154,176],[230,150,140],[60,143,255],[128,64,128],[92,82,55],
#     [254,212,124],[73,77,174],[255,160,98],[255,255,255],[104,84,109],[169,164,131],[92,136,89],
#     [137,54,74],[135,158,223],[7,246,231],[107,255,200],[58,41,149],[183,121,142],[255,73,97],
#     [107,142,35],[190,153,153],[146,139,141],[70,130,180]
# ]) / 255
# MY_SEMANTICS = [
#     "unlabeled","dynamic","ego vehicle","ground","static","parking","rail track","road","sidewalk",
#     "bridge","building","fence","garage","guard rail","tunnel","wall","banner","billboard","lane divider",
#     "parking sign","pole","polegroup","street light","traffic cone","traffic device","traffic light",
#     "traffic sign","traffic sign frame","terrain","vegetation","sky","person","rider","bicycle","bus",
#     "car","caravan","motorcycle","trailer","train","truck"
#   ]

def plot_label(
    label: np.ndarray,
    save_path: str = "",
    title: str = "",
    visualize: bool = True,
) -> None:
    """Visualize depth map"""
    norm = colors.Normalize(vmin=0, vmax=40)
    my_cmap = ListedColormap(MY_RGB, name='my_cmap')

    # fig, ax = plt.subplots(figsize=(30, 20))
    plt.figure(figsize=(30, 20))
    plt.imshow(label, cmap=my_cmap, norm=norm)
    
    cbar = plt.colorbar()

    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(MY_SEMANTICS):
        cbar.ax.text(2, (7.83 * j ) / 8.0 + 0.4, lab, ha='center', va='center')
    cbar.ax.get_yaxis().labelpad = 15
    
    plt.title(title)
    
    if visualize:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_label_with_rgb(
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
    plt.imshow(label, cmap=my_cmap, norm=norm, alpha=0.7)
    
    cbar = plt.colorbar()

    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(MY_SEMANTICS):
        cbar.ax.text(2, (7.83 * j ) / 8.0 + 0.4, lab, ha='center', va='center')
    cbar.ax.get_yaxis().labelpad = 15
    
    plt.title(title)
    
    if visualize:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
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
    plt.figure()
    plt.imshow(depth_map_visual, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)

    if visualize:
        plt.show()
    else:
        plt.imsave(save_path, depth_map_visual, cmap=cmap)
    plt.close()


if __name__ == "__main__":
    
    pan_dir = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_overlap/pan_mask/semantics/"
    save_dir = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_overlap/pan_mask/vis"
    img_dir = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_overlap/images/"
    for sem_file in tqdm(os.listdir(pan_dir)):
        sem_path = os.path.join(pan_dir, sem_file)
        save_path = os.path.join(save_dir, sem_file)
        save_path_only = os.path.join(save_dir, "onlyLabel_"+sem_file)
        img_path = os.path.join(img_dir, sem_file.replace("png", "jpg"))

        pil_image = Image.open(sem_path)
        
        img = Image.open(img_path)
        semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
        plot_label_with_rgb(semantics, img, save_path, visualize=False)
        plot_label(semantics, save_path_only, visualize=False)
    pdb.set_trace()