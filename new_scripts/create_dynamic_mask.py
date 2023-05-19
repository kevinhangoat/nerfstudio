#!/usr/bin/env python
import numpy as np
import os
from glob import glob
from numpy.lib.twodim_base import mask_indices
import imageio
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple
import copy
import pdb
import json
from pathlib import Path
from scalabel.label.transforms import rle_to_mask
from visualize_panoptics import plot_depth
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

def create_pan_mask_dict(
    pan_json_path: str, shape: Tuple[int, int] = (720, 1280)
):
    """Create a dictionary for panoptic segmentation."""
    if not os.path.exists(pan_json_path):
        return None
    with open(pan_json_path, "rb") as fp:
        fp_content = json.load(fp)
    frames = fp_content["frames"]
    result = {}
    for frame in frames:
        img_name = frame["name"]
        labels = frame["labels"]
        pan_dict: Dict[str, List[NDArrayU8]] = {
            "person": [],
            "rider": [],
            "bicycle": [],
            "bus": [],
            "car": [],
            "caravan": [],
            "motorcycle": [],
            "trailer": [],
            "train": [],
            "truck": [],
            "dynamic": [],
            "ego vehicle": [],
            "ground": [],
            "static": [],
            "parking": [],
            "rail track": [],
            "road": [],
            "sidewalk": [],
            "bridge": [],
            "building": [],
            "fence": [],
            "garage": [],
            "guard rail": [],
            "tunnel": [],
            "wall": [],
            "banner": [],
            "billboard": [],
            "lane divider": [],
            "parking sign": [],
            "pole": [],
            "polegroup": [],
            "street light": [],
            "traffic cone": [],
            "traffic device": [],
            "traffic light": [],
            "traffic sign": [],
            "traffic sign frame": [],
            "terrain": [],
            "vegetation": [],
            "sky": [],
            "unlabeled": [],
            "total": [],
        }
        sem_id = {
            "person": 31,
            "rider": 32,
            "bicycle": 33,
            "bus": 34,
            "car": 35,
            "caravan": 36,
            "motorcycle": 37,
            "trailer": 38,
            "train": 39,
            "truck": 40,
            "dynamic": 1,
            "ego vehicle": 2,
            "ground": 3,
            "static": 4,
            "parking": 5,
            "rail track": 6,
            "road": 7,
            "sidewalk": 8,
            "bridge": 9,
            "building": 10,
            "fence": 11,
            "garage": 12,
            "guard rail": 13,
            "tunnel": 14,
            "wall": 15,
            "banner": 16,
            "billboard": 17,
            "lane divider": 18,
            "parking sign": 19,
            "pole": 20,
            "polegroup": 21,
            "street light": 22,
            "traffic cone": 23,
            "traffic device": 24,
            "traffic light": 25,
            "traffic sign": 26,
            "traffic sign frame": 27,
            "terrain": 28,
            "vegetation": 29,
            "sky": 30,
            "unlabeled": 0,
        }
        result[img_name] = pan_dict
        pan_seg_total = np.zeros(shape)
        for label in labels:
            cur_label_mask = rle_to_mask(label["rle"])
            result[img_name][label["category"]].append(cur_label_mask)
            pan_seg_total += cur_label_mask * sem_id[label["category"]]
        result[img_name]["total"] = [pan_seg_total]  # type: ignore
    return result
    
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
    kernel = np.ones((30,30), np.uint8)
    pan_mask = 1 - cv2.dilate(pan_mask, kernel, iterations=1)

    ego_mask = np.zeros(shape)
    if len(pan_mask_dict["ego vehicle"]) != 0:
        ego_mask += pan_mask_dict["ego vehicle"][0]
    ego_kernel = np.ones((50,50), np.uint8)
    ego_mask_dilation = 1 - cv2.dilate(ego_mask, ego_kernel, iterations=1)
    mask = np.logical_and(ego_mask_dilation, pan_mask)
    return mask  # type: ignore


if __name__ == "__main__":
    json_path = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/pan_mask/pan_seg.json"
    mask_save_path = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/dynamic_mask_dilated"
    os.makedirs(mask_save_path, exist_ok=True)
    pan_masks_dict = create_pan_mask_dict(json_path)
    for image_name in tqdm(pan_masks_dict):
        
        pan_mask_dict = pan_masks_dict[image_name]
        shape = pan_mask_dict["total"][0].shape
        
        # ego_mask = get_egovehicle_mask(shape, pan_mask_dict)
        # cur_mask_save_path = os.path.join(mask_save_path, image_name.replace("jpg", "png"))
        pan_mask = get_pan_mask(shape, pan_mask_dict)
        cur_mask_save_path = os.path.join(mask_save_path, image_name.replace("jpg", "png"))
        cv2.imwrite(cur_mask_save_path, pan_mask.astype(np.uint8))
    