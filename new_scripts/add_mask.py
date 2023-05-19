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
    json_path = open("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11101_1177c1ab-463886b6/transforms.json")
    mask_home = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11101_1177c1ab-463886b6/dynamic_mask"
    json_content = json.load(json_path)
    new_frames = []
    for frame in json_content["frames"]:
        new_frame = frame
        file_name = new_frame["file_path"].split("/")[-1]
        # camera_id = new_frame["file_path"].split("/")[2][-2:]
        # file_name = camera_id + "_" + file_name
        mask_name = file_name.split('.')[0]+".png"
        mask_path = os.path.join("./dynamic_mask", mask_name)
        if not os.path.exists(os.path.join(mask_home, mask_name)):
            pdb.set_trace()
        new_frame["mask_path"] = mask_path
        new_frames.append(new_frame)
    json_content["frames"] = new_frames
    output_dir = Path("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11101_1177c1ab-463886b6/")
    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(json_content, f, indent=4)
