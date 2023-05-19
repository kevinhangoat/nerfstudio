#!/usr/bin/env python
import numpy as np
import os
import pdb
import open3d as o3d

if __name__ == "__main__":
    # Load the two point clouds
    pcd1 = o3d.io.read_point_cloud("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/0/0/semantic-nerfacto/2023-03-21_170252/point_cloud_semantic.ply")
    pcd2 = o3d.io.read_point_cloud("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/1/1/semantic-nerfacto/2023-04-15_204051/point_cloud_semantic.ply")
    pcd3 = o3d.io.read_point_cloud("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/2/2/semantic-nerfacto/2023-04-19_115204/point_cloud_semantic.ply")
    pcd4 = o3d.io.read_point_cloud("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/3/3/semantic-nerfacto/2023-03-28_192658/point_cloud_semantic.ply")
    pcd_car = o3d.io.read_point_cloud("/scratch-second/yuthan/lola-b9950-ano-2000-museo-falonso/source/CAR02.ply")

    # Merge the two point clouds
    pcd_combined = pcd1 + pcd2 + pcd3 + pcd4 + pcd_car

    # Save the merged point cloud to a new .ply file
    o3d.io.write_point_cloud("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/merged_point_cloud_semantic.ply", pcd_combined)