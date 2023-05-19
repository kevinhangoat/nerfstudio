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
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NeRF for BDD100K"
    )
    parser.add_argument(
        "--blocks_path",
        type=str,
        help="Dense path",
    )
    args = parser.parse_args()
    
    blocks_path = args.blocks_path
    
    for block in os.listdir(blocks_path):
        block_path = os.path.join(blocks_path, block)
        image_path = os.path.join(block_path, "images")
        num_steps = 600 * len(os.listdir(image_path))
        os.system(
            "CUDA_VISIBLE_DEVICES=0 ns-train semantic-nerfacto --vis tensorboard "
            f"--output_dir {blocks_path}/{block} "
            f"--data {blocks_path}/{block} "
            f"--max_num_iterations {num_steps} "
            "semantic-data"
        )