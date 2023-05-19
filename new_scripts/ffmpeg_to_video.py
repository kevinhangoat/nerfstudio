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
import glob
"""
python scripts/test.py --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/outputs/-scratch_net-biwidl208-yuthan-wisp_data-nerf-360_v2-garden/nerfacto/2022-10-30_174811/config.yml \
    --data-name d389c316-c71f7a5e \
    --save-name nerfacto
"""



if __name__ == "__main__":    
    path = "/scratch/yuthan/master_thesis/view_synthesis"
    name = "rgb"
    image_format = "jpeg"
    
    destination_path = os.path.join(path, name)
    os.makedirs(os.path.join(path, name), exist_ok=True)
    images = glob.glob(f"{os.path.join(path, name)}*.{image_format}")
    images.sort()
    for i, image in enumerate(images):
        
        new_name = f"{str(i).zfill(6)}.{image_format}"
        destination_file = os.path.join(destination_path, new_name)
        os.system(f"cp {image} {destination_file}")
    target_images = os.path.join(destination_path, f"*.{image_format}")
    target_video = os.path.join(destination_path, f"{name}.mp4")
    
    os.system(f"cd {destination_path}")
    os.system(f"ffmpeg -framerate 5 -pattern_type glob -i '{target_images}' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {target_video}")
    os.system(f"rm {target_images}")
    