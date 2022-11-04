#!/usr/bin/env python

import torch   
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pylab as plt
import pdb
import pickle 
from nerfstudio.utils.misc import create_pan_mask_dict, get_transient_mask
from pathlib import Path


def plot_depth(
    depth_map: np.ndarray,
    save_path: str = "",
    title: str = "",
    visualize: bool = True,
):
    """Visualize depth map"""
    depth_map_visual = np.ma.masked_where(depth_map == 0, depth_map)
    cmap = cm.Blues_r
    cmap.set_bad(color="gray")
    plt.figure(figsize=(30, 20))
    plt.imshow(depth_map_visual, cmap=cmap)
    plt.colorbar()
    plt.title(title)

    if visualize:
        plt.show()
    else:
        plt.imsave(save_path, depth_map_visual, cmap=cmap)
    plt.close()

if __name__ == "__main__":
    # maskplot_depth = torch.load("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/mask.pt")
    # pdb.set_trace()

    # pan_path = Path("/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/11201_48/pan_seg.json")
    # pan_seg_dict = create_pan_mask_dict(pan_path)
    # pdb.set_trace()
    batch = torch.load("pair.pt")
    pdb.set_trace()
    # mask = plot_depth(masks[0].to_numpy())