# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from vggt_omega.models.layers import Mlp, PatchEmbed, RopePositionEmbedding, SelfAttentionBlock, init_masked_qkv_bias_buffers
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer
from train_utils.patch_pos import PatchRowColRegressionCriterionDynamic


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

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
        rope_freq: int = 100,
        first_cam: bool = True,
        asg: bool = False,
        asg_max_hw: int = 512,
        loop: bool = True,
        grad_start_layer: int = 0,
    ) -> None:
        super().__init__()

        grad_start_layer = int(grad_start_layer)
        if not 0 <= grad_start_layer <= depth:
            raise ValueError(
                f"grad_start_layer must be in [0, {depth}], got {grad_start_layer}"
            )

        self.patch_token_start = 1 + num_register_tokens
        self.dinov3_hf_patch_embed = False
        self.__build_patch_embed__(
            patch_embed=patch_embed,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            embed_dim=embed_dim,
        )
        self.rope_embed = (
            RopePositionEmbedding(
                embed_dim=embed_dim,
                num_heads=num_heads,
                base=rope_freq,
                normalize_coords="max",
                dtype=torch.float32,
            )
            if rope_freq > 0
            else None
        )
        self.loop = loop
        self.grad_start_layer = grad_start_layer

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(1 if loop else depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(2 if loop else depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.first_cam = first_cam
        num_special_token_sets = 2 if first_cam else 1
        self.camera_token = nn.Parameter(torch.empty(1, num_special_token_sets, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, num_special_token_sets, num_register_tokens, embed_dim))
        self.num_global_register_tokens = num_global_register_tokens
        if num_global_register_tokens > 0:
            self.global_register_token = nn.Parameter(torch.empty(1, num_global_register_tokens, embed_dim))
        else:
            self.global_register_token = None

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()
        if asg:
            self.asg = PatchRowColRegressionCriterionDynamic(
                feat_dim=embed_dim,
                grid_h=asg_max_hw // self.patch_size,
                grid_w=asg_max_hw // self.patch_size,
                # loss_type="smooth_l1",
            )
        else:
            self.asg = None

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)
        if self.global_register_token is not None:
            nn.init.normal_(self.global_register_token, std=1e-3)
        init_masked_qkv_bias_buffers(self.frame_blocks)
        init_masked_qkv_bias_buffers(self.inter_frame_blocks)

    def __build_patch_embed__(
        self,
        patch_embed: str | None,
        patch_size: int,
        num_register_tokens: int,
        embed_dim: int,
    ) -> None:
        """
        Build the patch embed layer. If 'conv', use a simple PatchEmbed conv layer.
        If a DINOv3 checkpoint path/name is provided, load it through transformers.
        Otherwise, keep VGGT-Omega's built-in DINOv3-style patch embed.
        """
        if patch_embed is None:
            self.patch_embed = _build_local_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
            return

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=224,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
            )
            return

        if "dinov3" in patch_embed or "dino-ds" in patch_embed:
            from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

            self.patch_embed = DINOv3ViTModel.from_pretrained(patch_embed)
            hidden_size = getattr(self.patch_embed.config, "hidden_size", embed_dim)
            if hidden_size != embed_dim:
                raise ValueError(
                    f"DINOv3 patch_embed hidden_size ({hidden_size}) must match VGGTOmega embed_dim ({embed_dim})."
                )

            checkpoint_patch_size = getattr(self.patch_embed.config, "patch_size", patch_size)
            if checkpoint_patch_size != patch_size:
                raise ValueError(
                    f"DINOv3 patch_embed patch_size ({checkpoint_patch_size}) must match "
                    f"VGGTOmega patch_size ({patch_size})."
                )

            self.dinov3_hf_patch_embed = True
            self.patch_token_start = 1 + self.patch_embed.config.num_register_tokens + self.patch_token_start

            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)
            return

        raise ValueError(f"Unknown patch_embed type for VGGTOmega: {patch_embed!r}")

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], int]:
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)

        patch_tokens = self.patch_embed(images)
        if self.dinov3_hf_patch_embed:
            patch_tokens = patch_tokens.last_hidden_state
        elif isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        elif patch_tokens.ndim == 4:
            patch_tokens = patch_tokens.flatten(1, 2)

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape
        global_register_token = (
            self.global_register_token.expand(batch_size, -1, -1)
            if self.global_register_token is not None
            else None
        )

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        frame_rope = None
        if self.rope_embed is not None:
            with torch.no_grad():
                rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
                frame_rope = (
                    rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                    rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
                )

        outputs = []
        for block_idx in range(self.depth):
            block_grad_enabled = torch.is_grad_enabled() and (
                not self.training or block_idx >= self.grad_start_layer
            )
            with torch.set_grad_enabled(block_grad_enabled):
                tokens, frame_tokens = self._run_frame_block(
                    tokens,
                    batch_size,
                    num_frames,
                    num_tokens,
                    embed_dim,
                    block_idx,
                    frame_rope,
                )
                tokens, global_register_token = self._run_inter_frame_attention_block(
                    tokens,
                    batch_size,
                    num_frames,
                    num_tokens,
                    embed_dim,
                    block_idx,
                    self.inter_frame_attention_types[block_idx],
                    global_register_token,
                )
            if block_idx in self.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)

        if self.asg is not None:
            self.aux_loss = self.asg(tokens.flatten(0, -3)[:, self.patch_token_start:, :], patch_grid_size[0], patch_grid_size[1])
        return outputs, self.patch_token_start

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens = self.frame_blocks[0  if self.loop else block_idx](tokens, rope_sincos)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
        global_register_token: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            if self.num_global_register_tokens > 0:
                tokens = torch.cat([global_register_token, tokens], dim=1)
                tokens = self.inter_frame_blocks[0  if self.loop else block_idx](tokens, None)
                global_register_token = tokens[:, : self.num_global_register_tokens, :]
                tokens = tokens[:, self.num_global_register_tokens :, :].contiguous()
            else:
                tokens = self.inter_frame_blocks[0  if self.loop else block_idx](tokens, None)

            return tokens.view(batch_size, num_frames, num_tokens, embed_dim), global_register_token

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:]

        if self.num_global_register_tokens > 0:
            camera_and_register_tokens = torch.cat([global_register_token, camera_and_register_tokens], dim=1)
            camera_and_register_tokens = self.inter_frame_blocks[1  if self.loop else block_idx](camera_and_register_tokens, None)
            global_register_token = camera_and_register_tokens[:, : self.num_global_register_tokens, :]
            camera_and_register_tokens = camera_and_register_tokens[:, self.num_global_register_tokens :, :].contiguous()
        else:
            camera_and_register_tokens = self.inter_frame_blocks[1  if self.loop else block_idx](camera_and_register_tokens, None)

        camera_and_register_tokens = camera_and_register_tokens.view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2), global_register_token


def _build_local_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model



def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    if token_tensor.shape[1] == 1:
        return token_tensor.expand(batch_size, num_frames, *token_tensor.shape[2:]).reshape(
            batch_size * num_frames,
            *token_tensor.shape[2:],
        )

    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
