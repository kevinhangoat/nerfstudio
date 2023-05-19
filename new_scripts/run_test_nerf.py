#!/usr/bin/env python
import numpy as np
import os
import torch
import pdb
import open3d as o3d
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export NeRF pointcloud for BDD100K"
    )
    parser.add_argument(
        "--blocks_path",
        type=str,
        help="Dense path",
    )
    args = parser.parse_args()
    blocks_path = args.blocks_path
    pcd_path_list = []
    for block in os.listdir(blocks_path):
        block_path = os.path.join(blocks_path, block)
        
        result_path = os.path.join(block_path, block, "semantic-nerfacto")
        if not os.path.exists(result_path):
            continue
        result_list = os.listdir(result_path)
        result_list.sort()
        target_path = os.path.join(result_path, result_list[-1])
        os.system(
            "CUDA_VISIBLE_DEVICES=0 python ../../clean_nerfstudio/nerfstudio/new_scripts/test_original.py "
            f"--base-path {target_path} "
            f"--load-config {target_path}/config.yml "
            f"--save-name renders"
        )