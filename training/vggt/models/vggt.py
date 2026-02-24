# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import List
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import SimplifiedCameraHead, CameraHead, ProgressiveCameraHead, DeeperCameraHead
from vggt.heads.dpt_head import DPTHead, DPTSharedHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_cam_point=False,
        enable_depth=True,
        enable_track=True,
        depth=24, # Aggregator
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        first_cam=True, # Aggregator
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pose_encoding_type="absT_quaR_FoV",
        share_dpt_head: bool = False,
        dpt_frames_chunk_size: int = 8,
        conf_logit_max: float | None = None,
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, num_register_tokens=num_register_tokens,
            qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias, patch_embed=patch_embed,
            aa_order=aa_order, aa_block_size=aa_block_size, qk_norm=qk_norm, rope_freq=rope_freq, init_values=init_values,
            first_cam=first_cam
        )


        # self.camera_head = ProgressiveCameraHead(dim_in=2 * embed_dim, intermediate_layer_idx=intermediate_layer_idx) if enable_camera else None
        # self.camera_head = SimplifiedCameraHead(dim_in=2 * embed_dim) if enable_camera else None
        # self.camera_head = DeeperCameraHead(dim_in=2 * embed_dim, depth=4) if enable_camera else None
        if enable_camera:
            if pose_encoding_type is None or pose_encoding_type == "absT_quaR_FoV":
                self.camera_head = CameraHead(dim_in=2 * embed_dim)
            elif pose_encoding_type == "absT_quaR_logK":
                self.camera_head = CameraHead(dim_in=2 * embed_dim, pose_encoding_type="absT_quaR_logK")
            else:
                raise TypeError(f"Unsupported camera_head type: {type(pose_encoding_type)}")
        else:
            self.camera_head = None

        self.shared_dpt = None
        self.point_head = None
        self.cam_point_head = None
        self.depth_head = None
        self.conf_logit_max = conf_logit_max

        if share_dpt_head and (enable_point or enable_cam_point or enable_depth):
            self.shared_dpt = DPTHead(
                dim_in=2 * embed_dim,
                patch_size=patch_size,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
                conf_logit_max=conf_logit_max,
                intermediate_layer_idx=intermediate_layer_idx,
                feature_only=True,
            )
            if enable_point:
                self.point_head = DPTSharedHead(
                    feature_dim=self.shared_dpt.features,
                    output_dim=4,
                    activation="inv_log",
                    conf_activation="expp1",
                    conf_logit_max=conf_logit_max,
                )
            if enable_cam_point:
                self.cam_point_head = DPTSharedHead(
                    feature_dim=self.shared_dpt.features,
                    output_dim=4,
                    activation="inv_log",
                    conf_activation="expp1",
                    conf_logit_max=conf_logit_max,
                )
            if enable_depth:
                self.depth_head = DPTSharedHead(
                    feature_dim=self.shared_dpt.features,
                    output_dim=2,
                    activation="exp",
                    conf_activation="expp1",
                    conf_logit_max=conf_logit_max,
                )
        else:
            self.point_head = DPTHead(dim_in=2 * embed_dim, patch_size=patch_size, output_dim=4, activation="inv_log", conf_activation="expp1", conf_logit_max=conf_logit_max, intermediate_layer_idx=intermediate_layer_idx) if enable_point else None
            self.cam_point_head = DPTHead(dim_in=2 * embed_dim, patch_size=patch_size, output_dim=4, activation="inv_log", conf_activation="expp1", conf_logit_max=conf_logit_max, intermediate_layer_idx=intermediate_layer_idx) if enable_cam_point else None
            self.depth_head = DPTHead(dim_in=2 * embed_dim, patch_size=patch_size, output_dim=2, activation="exp", conf_activation="expp1", conf_logit_max=conf_logit_max, intermediate_layer_idx=intermediate_layer_idx) if enable_depth else None

        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        self.dpt_frames_chunk_size = dpt_frames_chunk_size

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            # predictions["pose_enc"] = pose_enc_list  # pose encoding of the last iteration
            predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
            # predictions["pose_enc_list"] = pose_enc_list

        with torch.amp.autocast(device_type='cuda', enabled=False):
            shared_features = None
            if self.shared_dpt is not None:
                shared_features = self.shared_dpt(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=self.dpt_frames_chunk_size,
                )

            if self.depth_head is not None:
                if shared_features is None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens_list,
                        images=images,
                        patch_start_idx=patch_start_idx,
                        frames_chunk_size=self.dpt_frames_chunk_size,
                    )
                else:
                    depth, depth_conf = self.depth_head(shared_features)
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                if shared_features is None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens_list,
                        images=images,
                        patch_start_idx=patch_start_idx,
                        frames_chunk_size=self.dpt_frames_chunk_size,
                    )
                else:
                    pts3d, pts3d_conf = self.point_head(shared_features)
                predictions["pts3d"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.cam_point_head is not None:
                if shared_features is None:
                    pts3d, pts3d_conf = self.cam_point_head(
                        aggregated_tokens_list,
                        images=images,
                        patch_start_idx=patch_start_idx,
                        frames_chunk_size=self.dpt_frames_chunk_size,
                    )
                else:
                    pts3d, pts3d_conf = self.cam_point_head(shared_features)
                predictions["pts3d_cam"] = pts3d
                predictions["cam_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["img"] = images  # store the images for visualization during inference

        return predictions
