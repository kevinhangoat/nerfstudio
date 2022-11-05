#!/usr/bin/env python
"""Processes a video or image sequence to a nerfstudio compatible dataset."""

import math
import random
import json
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple
import cv2
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pylab as plt
import pdb
from nerfstudio.utils import colmap_utils, install_checks
import os
import numpy as np
from PIL import Image
from bdd100k.utils import load_bdd100k_config
from scalabel.label.io import load
from scalabel.vis.label import LabelViewer
from scripts.visualize import plot_depth
from scalabel.label.transforms import rle_to_mask

STUFF_CLASSES = ["dynamic", "ego vehicle", "ground", "static", "parking", "rail track", "road", "sidewalk", "bridge", "building", "fence", "garage", "guard rail", "tunnel", "wall", "banner", "billboard", "lane divider", "parking sign", "pole", "polegroup", "street light", "traffic cone", "traffic device", "traffic light", "traffic sign", "traffic sign frame", "terrain", "vegetation", "sky", "unlabeled"]
THING_CLASSES = ["person", "rider", "bicycle", "bus", "car", "caravan", "motorcycle", "trailer", "train", "truck"]


def vis_panoptics():
    frames = load('/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/d389c316-c71f7a5e/pan_seg.json').frames
    IMG_DIR = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/d389c316-c71f7a5e/images"
    VIS_DIR = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/d389c316-c71f7a5e/segmentations/panoptic_vis"
    pan_seg = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/scripts/bdd100k/config/pan_seg.toml"
    viewer = LabelViewer(label_cfg=load_bdd100k_config(pan_seg))
    pdb.set_trace()
    for frame in frames:
        img = np.array(Image.open(os.path.join(IMG_DIR, frame.name)))
        viewer.draw(img, frame)
        viewer.save(os.path.join(VIS_DIR, frame.name))

def class_to_json(save_path):
    stuff_colors = []
    for i in range(len(STUFF_CLASSES)):
        color = []
        for i in range(3):
            color.append(math.floor(random.random()*255))
        stuff_colors.append(color)
    thing_colors = []
    for i in range(len(THING_CLASSES)):
        color = []
        for i in range(3):
            color.append(math.floor(random.random()*255))
        thing_colors.append(color)
    class_dict = {
        "stuff": STUFF_CLASSES,
        "stuff_colors":stuff_colors,
        "thing": THING_CLASSES,
        "thing_colors":thing_colors
    }
    with open(save_path / "panoptic_classes.json", "w", encoding="utf-8") as f:
        json.dump(class_dict, f, indent=4)


def save_seg_mask(pan_seg_path, save_path):
    stuff_path= "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/friends/TBBT-big_living_room/segmentations/stuff/TBBT_S12E01_00000087.png"
    stuff = cv2.imread(stuff_path, -1).astype(np.float32)
    thing_path= "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/friends/TBBT-big_living_room/segmentations/thing/TBBT_S12E01_00000087.png"
    thing = cv2.imread(thing_path, -1).astype(np.float32)

    with open(pan_seg_path, "rb") as fp:
        fp_content = json.load(fp)
    frames = fp_content["frames"]
    for frame in frames:
        img_name = frame['name']
        labels = frame["labels"]
        thing = np.zeros(labels[0]["rle"]["size"])
        stuff = np.zeros(labels[0]["rle"]["size"])
        for label in labels:
            if label["category"] in STUFF_CLASSES:
                stuff_id = int(label["id"]) - 10 + 1
                stuff += stuff_id * rle_to_mask(label["rle"])
            if label["category"] in THING_CLASSES:
                thing_id = int(label["id"])%1000 + 1
                thing += thing_id * rle_to_mask(label["rle"])
        new_img_name = img_name.replace('jpg', 'png')
        thing_uint16 = thing.astype(np.uint16)
        stuff_uint16 = stuff.astype(np.uint16)
        Image.save(
        # cv2.imwrite(
        #     str(save_path / "segmentations" / "thing" / new_img_name),
        #     thing_uint16,
        # )
        # cv2.imwrite(
        #     str(save_path / "segmentations" / "stuff" / new_img_name),
        #     stuff_uint16,
        # )
    # return result
    
def main():
    
    print("wait")

if __name__ == "__main__":
    save_path = Path('/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/d389c316-c71f7a5e/')
    pan_seg_path = Path('/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/d389c316-c71f7a5e/pan_seg.json')
    # save_seg_mask(pan_seg_path, save_path)
    # class_to_json(save_path)

    path = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/friends/TBBT-big_living_room/segmentations/stuff/TBBT_S12E01_00000087.png"
    thing = np.array(Image.open(path), dtype="int32")
    pdb.set_trace()