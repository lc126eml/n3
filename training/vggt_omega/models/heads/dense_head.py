# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Inspired by https://github.com/DepthAnything/Depth-Anything-V2

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import create_uv_grid, position_grid_to_embed


class DenseHead(nn.Module):
    """Dense prediction head used by the released VGGT-Omega checkpoints."""

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 16,
        features: int = 256,
        out_channels: list[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: list[int] = [4, 11, 17, 23],
        enable_depth: bool = True,
        enable_point: bool = False,
        enable_cam_point: bool = False,
        conf_logit_max: float | None = None,
    ) -> None:
        super().__init__()

        if patch_size % 4 != 0:
            raise ValueError(
                "DenseHead expects patch_size divisible by 4 because the fused feature is decoded "
                f"from 1/4 scale. Got patch_size={patch_size}."
            )

        self.patch_size = patch_size
        self.intermediate_layer_idx = intermediate_layer_idx
        self.final_shuffle_factor = patch_size // 4
        self.enable_depth = enable_depth
        self.enable_point = enable_point
        self.enable_cam_point = enable_cam_point
        self.conf_logit_max = conf_logit_max
        self.norm = nn.LayerNorm(dim_in, eps=1e-5)

        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                _make_dense_resize_layer(channels=out_channels[0], resize_scale=4.0),
                _make_dense_resize_layer(channels=out_channels[1], resize_scale=2.0),
                _make_dense_resize_layer(channels=out_channels[2], resize_scale=1.0),
                _make_dense_resize_layer(channels=out_channels[3], resize_scale=0.5),
            ]
        )

        self.scratch = _make_scratch(out_channels, features)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        if enable_depth:
            self.proj = _make_prediction_head(
                features,
                self.final_shuffle_factor**2,
            )
            self.proj_conf = _make_prediction_head(
                features,
                self.final_shuffle_factor**2,
            )
            _init_small_conf_prediction_head(self.proj_conf)
        else:
            self.proj = None
            self.proj_conf = None

        self.point_proj = None
        self.point_proj_conf = None
        if enable_point:
            self.point_proj = _make_prediction_head(features, 3 * self.final_shuffle_factor**2)
            self.point_proj_conf = _make_prediction_head(features, self.final_shuffle_factor**2)
            _init_small_conf_prediction_head(self.point_proj_conf)

        self.cam_point_proj = None
        self.cam_point_proj_conf = None
        if enable_cam_point:
            self.cam_point_proj = _make_prediction_head(features, 3 * self.final_shuffle_factor**2)
            self.cam_point_proj_conf = _make_prediction_head(features, self.final_shuffle_factor**2)
            _init_small_conf_prediction_head(self.cam_point_proj_conf)

        if not (enable_depth or enable_point or enable_cam_point):
            raise ValueError("DenseHead requires at least one enabled prediction branch")

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_token_start: int,
        frames_chunk_size: int | None = 8,
    ) -> dict[str, torch.Tensor]:
        if patch_token_start is None:
            raise ValueError("patch_token_start is required for DenseHead")

        _, num_frames, _, _, _ = images.shape

        if frames_chunk_size is None or frames_chunk_size >= num_frames:
            return self._forward_impl(aggregated_tokens_list, images, patch_token_start)

        assert frames_chunk_size > 0

        prediction_chunks: dict[str, list[torch.Tensor]] = {}
        for frames_start_idx in range(0, num_frames, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, num_frames)
            chunk = self._forward_impl(
                aggregated_tokens_list,
                images,
                patch_token_start,
                frames_start_idx,
                frames_end_idx,
            )
            for key, value in chunk.items():
                prediction_chunks.setdefault(key, []).append(value)

        return {key: torch.cat(value, dim=1) for key, value in prediction_chunks.items()}

    def _forward_impl(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_token_start: int,
        frames_start_idx: int | None = None,
        frames_end_idx: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        batch_size, num_frames, _, height, width = images.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size

        multi_scale_features = []
        for feature_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx]
            if x is None:
                raise ValueError(f"Aggregator did not cache layer {layer_idx}, which DenseHead needs.")
            x = x[:, :, patch_token_start:]
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            if x.dtype != torch.float32:
                x = x.float()

            x = x.reshape(batch_size * num_frames, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[feature_idx](x)
            x = self._apply_pos_embed(x, width, height)
            x = self.resize_layers[feature_idx](x)
            multi_scale_features.append(x)

        fused = self.scratch_forward(multi_scale_features)
        fused = self._apply_pos_embed(fused, width, height)

        predictions = {}
        if self.enable_depth:
            depth, depth_conf = self._predict_branch(
                fused,
                self.proj,
                self.proj_conf,
                batch_size,
                num_frames,
                activation="exp",
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if self.enable_point:
            pts3d, pts3d_conf = self._predict_branch(
                fused,
                self.point_proj,
                self.point_proj_conf,
                batch_size,
                num_frames,
                activation="inv_log",
            )
            predictions["pts3d"] = pts3d
            predictions["world_points_conf"] = pts3d_conf

        if self.enable_cam_point:
            pts3d_cam, pts3d_cam_conf = self._predict_branch(
                fused,
                self.cam_point_proj,
                self.cam_point_proj_conf,
                batch_size,
                num_frames,
                activation="inv_log",
            )
            predictions["pts3d_cam"] = pts3d_cam
            predictions["cam_points_conf"] = pts3d_cam_conf

        for key, value in predictions.items():
            if value.dtype != torch.float32:
                raise TypeError(f"DenseHead output {key} must be fp32, got {value.dtype}")

        return predictions

    def _predict_branch(
        self,
        fused: torch.Tensor,
        value_proj: nn.Module,
        conf_proj: nn.Module,
        batch_size: int,
        num_frames: int,
        activation: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value_logits = value_proj(fused)
        value_logits = F.pixel_shuffle(value_logits, self.final_shuffle_factor)
        value_logits = value_logits.permute(0, 2, 3, 1)

        confidence_logits = conf_proj(fused)
        confidence_logits = F.pixel_shuffle(confidence_logits, self.final_shuffle_factor)
        confidence_logits = confidence_logits.permute(0, 2, 3, 1).squeeze(-1)

        if activation == "exp":
            values = torch.exp(value_logits)
        elif activation == "inv_log":
            values = _inverse_log_transform(value_logits)
        else:
            raise ValueError(f"Unknown dense branch activation: {activation}")

        if self.conf_logit_max is not None:
            confidence_logits = confidence_logits.clamp(max=self.conf_logit_max)
        confidence = 1.0 + torch.exp(confidence_logits)

        values = values.view(batch_size, num_frames, *values.shape[1:]).contiguous()
        confidence = confidence.view(batch_size, num_frames, *confidence.shape[1:]).contiguous()
        return values, confidence

    def _apply_pos_embed(self, x: torch.Tensor, width: int, height: int, ratio: float = 0.1) -> torch.Tensor:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=width / height, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        return self.scratch.refinenet1(out, layer_1_rn, size=layer_1_rn.shape[2:])


def _make_dense_resize_layer(channels: int, resize_scale: float) -> nn.Module:
    if resize_scale == 1.0:
        return nn.Identity()

    if resize_scale == 0.5:
        return nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    upsample_scale = int(resize_scale)
    return nn.ConvTranspose2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=upsample_scale,
        stride=upsample_scale,
        padding=0,
    )


def _make_prediction_head(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)


def _inverse_log_transform(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * torch.expm1(torch.abs(y))


def _init_small_conf_prediction_head(proj: nn.Module) -> None:
    if not isinstance(proj, nn.Conv2d):
        raise TypeError(f"Unsupported confidence projection layer: {type(proj)}")

    nn.init.zeros_(proj.weight)
    if proj.bias is None:
        raise ValueError("Small confidence init requires a bias term for proj_conf")

    # With expp1 confidence activation this starts from conf ~= 1.05.
    nn.init.constant_(proj.bias, math.log(1.05 - 1.0))


def _make_fusion_block(features: int, has_residual: bool = True) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=False),
        has_residual=has_residual,
    )


def _make_scratch(in_shape: list[int], out_shape: int) -> nn.Module:
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, activation: nn.Module, has_residual: bool = True) -> None:
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation)
        self.resConfUnit2 = ResidualConvUnit(features, activation)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None, size: Tuple[int, int] | None = None) -> torch.Tensor:
        output = x
        if self.has_residual:
            if residual is None:
                raise ValueError("FeatureFusionBlock requires a residual tensor when has_residual=True")
            output = output + self.resConfUnit1(residual)

        output = self.resConfUnit2(output)
        output = custom_interpolate(output, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(output)


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        if scale_factor is None:
            raise ValueError("custom_interpolate requires either size or scale_factor")
        size = (
            int(x.shape[-2] * scale_factor),
            int(x.shape[-1] * scale_factor),
        )

    if tuple(x.shape[-2:]) == tuple(size):
        return x

    int_max = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements <= int_max:
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)

    chunks = torch.chunk(x, chunks=(input_elements // int_max) + 1, dim=0)
    interpolated_chunks = [
        F.interpolate(
            chunk,
            size=size,
            mode=mode,
            align_corners=align_corners,
        )
        for chunk in chunks
    ]
    return torch.cat(interpolated_chunks, dim=0).contiguous()
