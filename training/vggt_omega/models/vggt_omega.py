# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and dense prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        patch_embed: str | None = None,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        num_global_register_tokens: int = 0,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
        enable_camera: bool = True,
        enable_point: bool = False,
        enable_cam_point: bool = False,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        rope_freq: int = 100,
        first_cam: bool = True,
        conf_logit_max: float | None = None,
        dpt_frames_chunk_size: int | None = 8,
        asg: bool = False,
        asg_max_hw: int = 512,
        loop: bool = True,
    ) -> None:
        super().__init__()

        self.aggregator = Aggregator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            patch_embed=patch_embed,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_register_tokens=num_register_tokens,
            num_global_register_tokens=num_global_register_tokens,
            register_attention_block_indices=register_attention_block_indices,
            cached_layer_indices=cached_layer_indices,
            rope_freq=rope_freq,
            first_cam=first_cam,
            asg=asg,
            asg_max_hw=asg_max_hw,
            loop=loop,
        )
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = (
            DenseHead(
                dim_in=2 * embed_dim,
                patch_size=patch_size,
                intermediate_layer_idx=list(cached_layer_indices),
                enable_depth=enable_depth,
                enable_point=enable_point,
                enable_cam_point=enable_cam_point,
                conf_logit_max=conf_logit_max,
            )
            if enable_depth or enable_point or enable_cam_point
            else None
        )
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None
        self.dpt_frames_chunk_size = dpt_frames_chunk_size

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                predictions.update(
                    self.dense_head(
                        aggregated_tokens_list,
                        images=images,
                        patch_token_start=patch_token_start,
                        frames_chunk_size=self.dpt_frames_chunk_size,
                    )
                )

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        if rope_embed is None:
            continue
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
