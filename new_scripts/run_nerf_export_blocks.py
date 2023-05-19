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
        pcd_path = os.path.join(target_path, "point_cloud.ply")
        if not os.path.exists(pcd_path) and os.path.exists(os.path.join(target_path,"nerfstudio_models")):
            os.system(
                "CUDA_VISIBLE_DEVICES=0 ns-export pointcloud "
                f"--load-config {target_path}/config.yml "
                f"--output-dir {target_path} "
            )
        if os.path.exists(pcd_path):
            pcd_path_list.append(pcd_path)
        else:
            print(f"Missing point cloud for {block_path}")
    for cur_pcd_path in pcd_path_list:
        cur_pcd = o3d.io.read_point_cloud(cur_pcd_path)
        if "pcd_combined" not in locals():
            pcd_combined = cur_pcd
        else:
            pcd_combined += cur_pcd
    o3d.io.write_point_cloud(os.path.join(blocks_path, "merged_point_cloud.ply"), pcd_combined)

