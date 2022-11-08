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

from typing import Dict

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Depths
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_from_path


class DepthDataset(InputDataset):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs):
        super().__init__(dataparser_outputs)
        assert "depths" in dataparser_outputs.metadata.keys() and isinstance(
            dataparser_outputs.metadata["depths"], Depths
        )
        self.depths = dataparser_outputs.metadata["depths"]
        self.depth_scale_factor = dataparser_outputs.depth_scale_factor

    def get_metadata(self, data: Dict) -> Dict:
        # handle mask
        filepath = self.depths.filenames[data["image_idx"]]
        depth = get_depth_from_path(
            filepath=filepath, depth_scale_factor=self.depth_scale_factor
        )
        return {"depth": depth}
