#!/usr/bin/env python
import numpy as np
import os
from glob import glob
from numpy.lib.twodim_base import mask_indices
import imageio
from multiprocessing import Pool
import copy
import torch
import pdb
import json
from pathlib import Path

if __name__ == "__main__":
    json_path = open("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/KITTI-360-raw-single/transforms.json")
    semantic_home = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/KITTI-360-raw-single/pan_mask/semantics"
    json_content = json.load(json_path)
    new_frames = []
    for frame in json_content["frames"]:
        new_frame = frame
        file_name = new_frame["file_path"].split("/")[-1]
        camera_id = new_frame["file_path"].split("/")[2][-2:]
        # file_name = camera_id + "_" + file_name
        # semantic_name = "semantic_class_" + file_name.split('_')[1]
        semantic_name = "00_" + file_name.split('.')[0]+".png"
        semantic_path = os.path.join(semantic_home, semantic_name)
        if not os.path.exists(os.path.join(semantic_home, semantic_name)):
            pdb.set_trace()
        new_frame["semantics_file_path"] = semantic_path
        new_frames.append(new_frame)
    json_content["frames"] = new_frames
    json_content["panoptic_classes"] = "panoptic_classes.json"
    output_dir = Path("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/KITTI-360-raw-single/")
    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(json_content, f, indent=4)
