#!/usr/bin/env python
import numpy as np
import os
from glob import glob
from numpy.lib.twodim_base import mask_indices
import imageio
from multiprocessing import Pool
import copy
import pdb
import json
from pathlib import Path
from .utils import create_pan_mask_dict, plot_depth
import cv2
from tqdm import tqdm

# def get_egovehicle_mask(
#     shape, pan_mask_dict
# ):
#     ego_mask = np.zeros(shape)
#     if len(pan_mask_dict["ego vehicle"]) != 0:
#         ego_mask += pan_mask_dict["ego vehicle"][0]
#     if len(pan_mask_dict["static"]) != 0:
#         ego_mask += pan_mask_dict["static"][0]
        
#     ego_mask = 1 - ego_mask

#     return ego_mask  # type: ignore

def get_egovehicle_mask(
    shape, pan_mask_dict
):
    ego_mask = np.zeros(shape)
    if len(pan_mask_dict["ego vehicle"]) != 0:
        ego_mask += pan_mask_dict["ego vehicle"][0]

    kernel = np.ones((50,50), np.uint8)
    ego_mask_dilation = 1 - cv2.dilate(ego_mask, kernel, iterations=1)
    # if len(pan_mask_dict["unlabeled"]) != 0:
    #     unlabeled = pan_mask_dict["unlabeled"][0]
    #     mask[unlabeled==1] = 0
    return ego_mask_dilation  # type: ignore

def get_pan_mask(
    shape, pan_mask_dict
):
    pan_mask = np.zeros(shape)
    if len(pan_mask_dict["ego vehicle"]) != 0:
        pan_mask += pan_mask_dict["ego vehicle"][0]
    if len(pan_mask_dict["static"]) != 0:
        pan_mask += pan_mask_dict["static"][0]
    if len(pan_mask_dict["unlabeled"]) != 0:
        pan_mask += pan_mask_dict["unlabeled"][0]
    if len(pan_mask_dict["static"]) != 0:
        pan_mask += pan_mask_dict["static"][0]
    transient_instances = (
        pan_mask_dict["car"]
        + pan_mask_dict["bus"]
        + pan_mask_dict["truck"]
        + pan_mask_dict["person"]
        + pan_mask_dict["rider"]
        + pan_mask_dict["bicycle"]
    )
    if len(transient_instances) != 0:
        for instance_mask in transient_instances:
            pan_mask += instance_mask
    # kernel = np.ones((50,50), np.uint8)
    # pan_mask = 1 - cv2.dilate(pan_mask, kernel, iterations=1)
    return 1-pan_mask  # type: ignore


if __name__ == "__main__":
    json_path = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/1177c1ab-463886b6/pan_mask/pan_seg.json"
    mask_save_path = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/1177c1ab-463886b6/dynamic_mask"
    os.makedirs(mask_save_path, exist_ok=True)
    pan_masks_dict = create_pan_mask_dict(json_path)
    for image_name in tqdm(pan_masks_dict):
        
        pan_mask_dict = pan_masks_dict[image_name]
        shape = pan_mask_dict["total"][0].shape
        
        ego_mask = get_egovehicle_mask(shape, pan_mask_dict)
        cur_mask_save_path = os.path.join(mask_save_path, image_name.replace("jpg", "png"))
        cv2.imwrite(cur_mask_save_path, ego_mask.astype(np.uint8))
    pdb.set_trace()
