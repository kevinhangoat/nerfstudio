# Copyright 2022 The Plenoptix Team. All rights reserved.
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
The field module baseclass.
"""
from abc import abstractmethod

from torch import nn
from torchtyping import TensorType


class FieldModule(nn.Module):
    """Field modules that can be combined to store and compute the fields."""

    def __init__(self) -> None:
        """Default initialization of module"""
        super().__init__()
        self.in_dim = 0
        self.out_dim = 0

    def build_nn_modules(self) -> None:
        """Function instantiates any torch.nn members within the module.
        If none exist, do nothing."""

    def set_in_dim(self, in_dim: int) -> None:
        """Sets input dimension of encoding

        Args:
            in_dim (int): input dimension
        """
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding.

        Returns:
            int: output dimension
        """
        if not hasattr(self, "out_dim"):
            raise ValueError("Output dimension has not been set")
        if self.out_dim <= 0:
            raise ValueError("Output dimension should be greater than zero")
        return self.out_dim

    @abstractmethod
    def forward(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """
        Args:
            in_tensor (TensorType[..., "input_dim"]): Input tensor to process

        Returns:
            TensorType[..., "output_dim"]: Processed tensor
        """
        raise NotImplementedError
