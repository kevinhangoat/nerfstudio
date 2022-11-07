# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Miscellaneous helper code.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from pathlib import Path
import json
import numpy as np
from scalabel.label.transforms import rle_to_mask
import pdb

def get_dict_to_torch(stuff: Any, device: Union[torch.device, str] = "cpu", exclude: Optional[List[str]] = None):
    """Set everything in the dict to the specified torch device.

    Args:
        stuff: things to convert to torch
        device: machine to put the "stuff" on
        exclude: list of keys to skip over transferring to device
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            if exclude and k in exclude:
                stuff[k] = v
            else:
                stuff[k] = get_dict_to_torch(v, device)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.to(device)
    return stuff


def get_dict_to_cpu(stuff: Any):
    """Set everything in the dict to CPU.

    Args:
        stuff: things to place onto cpu
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            stuff[k] = get_dict_to_cpu(v)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.detach().cpu()
    return stuff


def get_masked_dict(d, mask):
    """Return a masked dictionary.
    TODO(ethan): add more asserts/checks so this doesn't have unpredictable behavior.

    Args:
        d: dict to process
        mask: mask to apply to values in dictionary
    """
    masked_dict = {}
    for key, value in d.items():
        masked_dict[key] = value[mask]
    return masked_dict


class IterableWrapper:  # pylint: disable=too-few-public-methods
    """A helper that will allow an instance of a class to return multiple kinds of iterables bound
    to different functions of that class.

    To use this, take an instance of a class. From that class, pass in the <instance>.<new_iter_function>
    and <instance>.<new_next_function> to the IterableWrapper constructor. By passing in the instance's
    functions instead of just the class's functions, the self argument should automatically be accounted
    for.

    Args:
        new_iter: function that will be called instead as the __iter__() function
        new_next: function that will be called instead as the __next__() function
        length: length of the iterable. If -1, the iterable will be infinite.


    Attributes:
        new_iter: object's pointer to the function we are calling for __iter__()
        new_next: object's pointer to the function we are calling for __next__()
        length: length of the iterable. If -1, the iterable will be infinite.
        i: current index of the iterable.

    """

    i: int

    def __init__(self, new_iter: Callable, new_next: Callable, length: int = -1):
        self.new_iter = new_iter
        self.new_next = new_next
        self.length = length

    def __next__(self):
        if self.length != -1 and self.i >= self.length:
            raise StopIteration
        self.i += 1
        return self.new_next()

    def __iter__(self):
        self.new_iter()
        self.i = 0
        return self


def scale_dict(dictionary: Dict[Any, Any], coefficients: Dict[str, float]) -> Dict[Any, Any]:
    """Scale a dictionary in-place given a coefficients dictionary.

    Args:
        dictionary: input dict to be scaled.
        coefficients: scalar dict config for holding coefficients.

    Returns:
        Input dict scaled by coefficients.
    """
    for key in dictionary:
        if key in coefficients:
            dictionary[key] *= coefficients[key]
    return dictionary


def step_check(step, step_size, run_at_zero=False) -> bool:
    """Returns true based on current step and step interval."""
    if step_size == 0:
        return False
    return (run_at_zero or step != 0) and step % step_size == 0


def update_avg(prev_avg: float, new_val: float, step: int) -> float:
    """helper to calculate the running average

    Args:
        prev_avg (float): previous average value
        new_val (float): new value to update the average with
        step (int): current step number

    Returns:
        float: new updated average
    """
    return (step * prev_avg + new_val) / (step + 1)

def create_pan_mask_dict(pan_json_path: Path) -> Dict[str, np.array]:
    if not pan_json_path.is_file():
        return None
    with open(pan_json_path, "rb") as fp:
        fp_content = json.load(fp)
    frames = fp_content["frames"]
    result = dict()
    for frame in frames:
        img_name = frame['name']
        labels = frame["labels"]
        pan_dict = {
            "person":[], "rider":[], "bicycle":[], "bus":[], "car":[], 
            "caravan":[], "motorcycle":[], "trailer":[], "train":[], 
            "truck":[], "dynamic":[], "ego vehicle":[], "ground":[], 
            "static":[], "parking":[], "rail track":[], "road":[], 
            "sidewalk":[], "bridge":[], "building":[], "fence":[], 
            "garage":[], "guard rail":[], "tunnel":[], "wall":[], 
            "banner":[], "billboard":[], "lane divider":[], 
            "parking sign":[], "pole":[], "polegroup":[], "street light":[], 
            "traffic cone":[], "traffic device":[], "traffic light":[], 
            "traffic sign":[], "traffic sign frame":[], "terrain":[], 
            "vegetation":[], "sky":[], "unlabeled":[]
        }
        result[img_name] = pan_dict
        for label in labels:
            result[img_name][label["category"]].append(rle_to_mask(label["rle"]))
    return result


def get_transient_mask(pan_seg_dict, image_name, shape):
    """
    Create transient mask that contains transient objects(car, bike so on) and ego vehicle
    """
    mask = np.zeros(shape)
    if not pan_seg_dict:
        # If no panoptic masks are found, return matrix of ones
        return 1- mask
    if len(pan_seg_dict[image_name]['ego vehicle']) != 0:
        mask += pan_seg_dict[image_name]['ego vehicle'][0]
    if len(pan_seg_dict[image_name]['unlabeled']) != 0:
        mask += pan_seg_dict[image_name]['unlabeled'][0]
    transient_instances = pan_seg_dict[image_name]['car'] + pan_seg_dict[image_name]['bus'] + pan_seg_dict[image_name]['truck'] + \
                          pan_seg_dict[image_name]['person'] + pan_seg_dict[image_name]['rider'] + pan_seg_dict[image_name]['bicycle']
    if len(transient_instances)!=0:    
        for instance_mask in transient_instances:
            mask += instance_mask 
    return 1-mask
