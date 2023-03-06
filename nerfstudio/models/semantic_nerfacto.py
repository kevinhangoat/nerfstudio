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
Nerfacto augmented with depth supervision and semantic head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.losses import DepthLossType, depth_loss
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.model_components.renderers import SemanticRenderer, UncertaintyRenderer
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.fields.semantic_field import SemanticField
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
import pdb

@dataclass
class SemanticNerfactoModelConfig(NerfactoModelConfig):
    """Additional parameters for depth supervision."""

    _target: Type = field(default_factory=lambda: SemanticNerfactoModel)
    depth_loss_mult: float = 1e-3
    """Lambda of the depth loss."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.01
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    should_decay_sigma: bool = False
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""
    depth_loss_type: DepthLossType = DepthLossType.DS_NERF
    """Depth loss type."""
    use_transient_embedding: bool = False
    """Whether to use transient embedding."""
    semantic_loss_weight: float = 1e-3
    """Lambda of the depth loss."""
    pass_semantic_gradients: bool = True
    """Use semantic gradients or not."""


class SemanticNerfactoModel(NerfactoModel):
    """Depth and Semantic loss augmented nerfacto model.

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: SemanticNerfWModelConfig

    def __init__(self, config: SemanticNerfWModelConfig, metadata: Dict, **kwargs) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics = metadata["semantics"]
        super().__init__(config=config, **kwargs)
        self.colormap = self.semantics.colors.clone().detach().to(self.device)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.config.depth_sigma])

        scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_transient_embedding=self.config.use_transient_embedding,
            use_semantics=False,
            num_semantic_classes=len(self.semantics.classes),
        )

        self.semantic_networks = SemanticField(
            num_semantic_classes=len(self.semantics.classes),
            pass_semantic_gradients=self.config.pass_semantic_gradients,
        )

        # renderers
        self.renderer_uncertainty = UncertaintyRenderer()
        self.renderer_semantics = SemanticRenderer()

        # losses
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def get_outputs(self, ray_bundle: RayBundle):
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples)

        if self.training and self.config.use_transient_embedding:
            density = field_outputs[FieldHeadNames.DENSITY] + field_outputs[FieldHeadNames.TRANSIENT_DENSITY]
            weights = ray_samples.get_weights(density)
            weights_static = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            rgb_static_component = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            rgb_transient_component = self.renderer_rgb(
                rgb=field_outputs[FieldHeadNames.TRANSIENT_RGB], weights=weights
            )
            rgb = rgb_static_component + rgb_transient_component
        else:
            weights_static = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            weights = weights_static
            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        weights_list.append(weights_static)
        ray_samples_list.append(ray_samples)

        semantic_outputs = self.semantic_networks(ray_samples, self.field.density_embedding)
        
        depth = self.renderer_depth(weights=weights_static, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights_static)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        outputs["weights_list"] = weights_list
        outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # transients
        if self.training and self.config.use_transient_embedding:
            weights_transient = ray_samples.get_weights(field_outputs[FieldHeadNames.TRANSIENT_DENSITY])
            uncertainty = self.renderer_uncertainty(field_outputs[FieldHeadNames.UNCERTAINTY], weights_transient)
            outputs["uncertainty"] = uncertainty + 0.03  # NOTE(ethan): this is the uncertainty min
            outputs["density_transient"] = field_outputs[FieldHeadNames.TRANSIENT_DENSITY]

        # semantics
        semantic_weights = weights_static
        if not self.config.pass_semantic_gradients:
            semantic_weights = semantic_weights.detach()
        outputs["semantics"] = self.renderer_semantics(
            semantic_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )

        # semantics colormaps
        semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            metrics_dict["depth_loss"] = 0.0
            sigma = self._get_sigma().to(self.device)
            termination_depth = batch["depth_image"].to(self.device)
            for i in range(len(outputs["weights_list"])):
                metrics_dict["depth_loss"] += depth_loss(
                    weights=outputs["weights_list"][i],
                    ray_samples=outputs["ray_samples_list"][i],
                    termination_depth=termination_depth,
                    predicted_depth=outputs["depth"],
                    sigma=sigma,
                    directions_norm=outputs["directions_norm"],
                    is_euclidean=self.config.is_euclidean_depth,
                    depth_loss_type=self.config.depth_loss_type,
                ) / len(outputs["weights_list"])

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        image = batch["image"].to(self.device)
        if self.training:
            assert metrics_dict is not None and "depth_loss" in metrics_dict
            loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]

        # transient loss
        if self.training and self.config.use_transient_embedding:
            betas = outputs["uncertainty"]
            loss_dict["uncertainty_loss"] = 3 + torch.log(betas).mean()
            loss_dict["density_loss"] = 0.01 * outputs["density_transient"].mean()
            loss_dict["rgb_loss"] = (((image - outputs["rgb"]) ** 2).sum(-1) / (betas[..., 0] ** 2)).mean()
        else:
            loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        # semantic loss
        loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.cross_entropy_loss(
            outputs["semantics"], batch["semantics"][..., 0].long()
        )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Appends ground truth depth to the depth image."""
        metrics, images = super().get_image_metrics_and_images(outputs, batch)
        ground_truth_depth = batch["depth_image"]
        if not self.config.is_euclidean_depth:
            ground_truth_depth = ground_truth_depth * outputs["directions_norm"]

        ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
        predicted_depth_colormap = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=torch.min(ground_truth_depth),
            far_plane=torch.max(ground_truth_depth),
        )
        images["depth"] = torch.cat([ground_truth_depth_colormap, predicted_depth_colormap], dim=1)
        depth_mask = ground_truth_depth > 0
        metrics["depth_mse"] = torch.nn.functional.mse_loss(
            outputs["depth"][depth_mask], ground_truth_depth[depth_mask]
        )

        # semantics
        semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        images["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        # valid mask
        images["mask"] = batch["mask"].repeat(1, 1, 3)

        return metrics, images

    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(  # pylint: disable=attribute-defined-outside-init
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma
