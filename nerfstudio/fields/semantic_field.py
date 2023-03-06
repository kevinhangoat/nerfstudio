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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import Encoding, HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    PredNormalsFieldHead,
    RGBFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass
import pdb

def get_normalized_directions(directions: TensorType["bs":..., 3]) -> TensorType["bs":..., 3]:
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


class SemanticField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        hidden_dim: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        num_semantic_classes: number of semantic classes
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
    ) -> None:
        super().__init__()

        self.geo_feat_dim = geo_feat_dim

        self.pass_semantic_gradients = pass_semantic_gradients

        # semantics
        self.mlp_semantics = tcnn.Network(
            n_input_dims=self.geo_feat_dim,
            n_output_dims=hidden_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        self.field_head_semantics = SemanticFieldHead(
            in_dim=self.mlp_semantics.n_output_dims, num_classes=num_semantic_classes
        )


    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # semantics
        semantics_input = density_embedding.view(-1, self.geo_feat_dim)
        if not self.pass_semantic_gradients:
            semantics_input = semantics_input.detach()

        x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
        outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        return outputs
    
    def forward(
        self,
        ray_samples: RaySamples,
        density_embedding: Optional[TensorType] = None,
    ) -> Dict[FieldHeadNames, TensorType]:
        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        
        return field_outputs