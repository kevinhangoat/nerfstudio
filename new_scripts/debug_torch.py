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
from visualize_panoptics import plot_depth, plot_label


if __name__ == "__main__":
    file_path = "/srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/depth_image.pt" 
    tmp = torch.load(file_path)
    # tmp[torch.isnan(tmp)]
    # torch.where(torch.isnan(tmp) == True)
    pdb.set_trace()