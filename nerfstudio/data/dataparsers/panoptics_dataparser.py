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

"""Data parser for Panoptics dataset"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.console import Console

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
import pdb
console = Console()

MAX_AUTO_RESOLUTION = 1600

def get_semantics_and_masks(image_idx: int, semantics: Semantics):
    """function to process additional semantics and mask information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """
    # handle mask
    car_index = semantics.thing_classes.index("car")
    truck_index = semantics.thing_classes.index("truck")
    bus_index = semantics.thing_classes.index("bus")
    person_index = semantics.thing_classes.index("person")

    thing_image_filename = semantics.thing_filenames[image_idx]
    pil_image = Image.open(thing_image_filename)
    thing_semantics = torch.from_numpy(np.array(pil_image, dtype="int32"))[..., None]
    car_mask = (thing_semantics != car_index).to(torch.float32)  # 1 where valid
    truck_mask = (thing_semantics != truck_index).to(torch.float32)
    bus_mask = (thing_semantics != bus_index).to(torch.float32)
    mask = (thing_semantics != person_index).to(torch.float32)
    # mask = torch.logical_and(torch.logical_and(car_mask, truck_mask), bus_mask)
    # handle semantics
    # stuff
    stuff_image_filename = semantics.stuff_filenames[image_idx]
    pil_image = Image.open(stuff_image_filename)
    stuff_semantics = torch.from_numpy(np.array(pil_image, dtype="int32"))[..., None]
    # thing
    thing_image_filename = semantics.thing_filenames[image_idx]
    pil_image = Image.open(thing_image_filename)
    thing_semantics = torch.from_numpy(np.array(pil_image, dtype="int32"))[..., None]
    return {"mask": mask, "semantics_stuff": stuff_semantics, "semantics_thing": thing_semantics}


@dataclass
class PanopticsDataParserConfig(DataParserConfig):
    """Panoptics dataset parser config"""

    _target: Type = field(default_factory=lambda: Panoptics)
    """target class to instantiate"""
    data: Path = Path("data/friends/TBBT-big_living_room")
    """Directory specifying location of data."""
    include_semantics: bool = True
    """whether or not to include loading of semantics data"""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    orientation_method: Literal["pca", "up"] = "up"
    """The method to use for orientation."""
    train_split_percentage: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""

    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the Panoptics axis-aligned bbox will be scaled to this value.
    """


@dataclass
class Panoptics(DataParser):
    """Panoptics Dataset"""

    config: PanopticsDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements

        meta = load_from_json(self.config.data / "transforms.json")
        image_filenames = []
        poses = []

        images_folder = f"images"
        segmentations_folder = f"segmentations"

        num_skipped_image_filenames = 0
        pan_path = self.config.data / "pan_seg.json"
        # pan_seg_dict = create_pan_mask_dict(pan_path)
        image_shape = (int(meta['h']), int(meta['w']))
        for idx, frame in enumerate(meta["frames"]):
            if "\\" in frame["file_path"]:
                filepath = PureWindowsPath(frame["file_path"])
            else:
                filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath)
            if not fname:
                num_skipped_image_filenames += 1
            else:
                image_filenames.append(fname)
                poses.append(np.array(frame["transform_matrix"]))
            # frame_mask = get_transient_mask(pan_seg_dict, filepath.name, image_shape)
            # masks_all.append(torch.from_numpy(frame_mask)[..., None])
        if num_skipped_image_filenames >= 0:
            logging.info("Skipping %s files in dataset split %s.", num_skipped_image_filenames, split)
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.config.train_split_percentage)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses = camera_utils.auto_orient_poses(poses, method=self.config.orientation_method)

        # Scale poses
        scale_factor = 1.0 / torch.max(torch.abs(poses[:, :3, 3]))
        poses[:, :3, 3] *= scale_factor * self.config.scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        poses = poses[indices]
        # masks_all = torch.stack(masks_all)
        # masks = masks_all[indices]
        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        # --- semantics ---
        semantics = None
        if self.config.include_semantics:
            thing_filenames = [
                Path(
                    str(image_filename)
                    .replace(f"/{images_folder}/", f"/{segmentations_folder}/thing/")
                    .replace(".jpg", ".png")
                )
                for image_filename in image_filenames
            ]
            stuff_filenames = [
                Path(
                    str(image_filename)
                    .replace(f"/{images_folder}/", f"/{segmentations_folder}/stuff/")
                    .replace(".jpg", ".png")
                )
                for image_filename in image_filenames
            ]
            panoptic_classes = load_from_json(self.config.data / "panoptic_classes.json")
            stuff_classes = panoptic_classes["stuff"]
            stuff_colors = torch.tensor(panoptic_classes["stuff_colors"], dtype=torch.float32) / 255.0
            thing_classes = panoptic_classes["thing"]
            thing_colors = torch.tensor(panoptic_classes["thing_colors"], dtype=torch.float32) / 255.0
            semantics = Semantics(
                stuff_classes=stuff_classes,
                stuff_colors=stuff_colors,
                stuff_filenames=stuff_filenames,
                thing_classes=thing_classes,
                thing_colors=thing_colors,
                thing_filenames=thing_filenames,
            )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta["k1"]) if "k1" in meta else 0.0,
            k2=float(meta["k2"]) if "k2" in meta else 0.0,
            k3=float(meta["k3"]) if "k3" in meta else 0.0,
            k4=float(meta["k4"]) if "k4" in meta else 0.0,
            p1=float(meta["p1"]) if "p1" in meta else 0.0,
            p2=float(meta["p2"]) if "p2" in meta else 0.0,
        )

        cameras = Cameras(
            fx=float(meta["fl_x"]),
            fy=float(meta["fl_y"]),
            cx=float(meta["cx"]),
            cy=float(meta["cy"]),
            distortion_params=distortion_params,
            height=int(meta["h"]),
            width=int(meta["w"]),
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs={"semantics": {"func": get_semantics_and_masks, "kwargs": {"semantics": semantics}}},
            semantics=semantics,
        )
        return dataparser_outputs

    def _get_fname(self, filepath):
        """Get the filename of the image file."""

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(self.config.data / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (self.config.data / f"images_{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                console.print(f"Auto image downscale factor of {2**df}")
                self.downscale_factor = 2**df
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return self.config.data / f"images_{self.downscale_factor}" / filepath.name
        return self.config.data / filepath
