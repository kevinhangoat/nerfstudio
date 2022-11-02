#!/usr/bin/env python

import open3d as o3d
import cv2
import pdb
import numpy as np
depth_path = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/renders/waymo/block_0/nerfacto/depth_22.png"
depth = cv2.imread(depth_path, -1).astype(np.float32) / 100.0
depth_img = o3d.geometry.Image(depth)
intrinsic = o3d.camera.PinholeCameraIntrinsic(1382, 1216,1250, 1250, 691.0, 608.0)


pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_img, intrinsic)
pdb.set_trace()
