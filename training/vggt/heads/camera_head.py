# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose

class DeeperCameraHead(nn.Module):
    """
    Simplified CameraHead that predicts camera parameters in a single forward pass.

    This version removes the iterative refinement and adaptive modulation, making it
    a simpler and faster direct predictor.

    It now supports a stack of 'depth' residual MLP blocks for deeper processing.
    """
    def __init__(
        self,
        dim_in: int = 2048,
        pose_encoding_type: str = "absT_quaR_FoV",
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "sigmoid_range_fov",
        depth: int = 4, 
        mlp_hidden_dim_factor: int = 2,
    ):
        """
        Args:
            dim_in (int): Input feature dimension.
            pose_encoding_type (str): Type of pose encoding.
            trans_act (str): Activation for translation.
            quat_act (str): Activation for quaternion.
            fl_act (str): Activation for focal length / FoV.
            depth (int): Number of residual MLP blocks to apply.
            mlp_hidden_dim_factor (int): Factor to determine the hidden dimension
                                         of the residual MLPs (hidden_dim = dim_in * factor).
        """
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
            self.fl_act = "relu"
        elif pose_encoding_type == "absT_quaR_logK":
            self.target_dim = 11
            self.fl_act = "linear"
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.depth = depth

        # Initial normalization for the camera token.
        self.token_norm = nn.LayerNorm(dim_in)

        # --- Residual MLP Blocks ---
        # We create 'depth' number of (Norm + MLP) blocks for residual connections.
        self.pose_mlps = nn.ModuleList()
        self.pose_norms = nn.ModuleList()
        hidden_dim = dim_in * mlp_hidden_dim_factor

        for _ in range(self.depth):
            self.pose_norms.append(nn.LayerNorm(dim_in))
            self.pose_mlps.append(Mlp(
                in_features=dim_in,
                hidden_features=hidden_dim,
                out_features=dim_in,  # Output dim must match input dim for residual
                drop=0
            ))
        # --- End Residual MLP Blocks ---

        # Final MLP layer to project to the target pose dimension.
        # This is the same as the original self.pose_branch.
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0
        )

    def forward(self, aggregated_tokens_list: list) -> torch.Tensor:
        """
        Forward pass to predict camera parameters in a single shot.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                                           the last tensor is used for prediction.

        Returns:
            torch.Tensor: The predicted camera encoding (post-activation).
        """
        tokens = aggregated_tokens_list[-1]

        # Get the [POSE] token (assuming it's the first token)
        pose_tokens = tokens[:, :, 0]
        
        # Apply initial normalization
        pose_tokens = self.token_norm(pose_tokens)

        # Apply the stack of residual MLP blocks
        for i in range(self.depth):
            residual = pose_tokens
            x = self.pose_norms[i](pose_tokens)
            x = self.pose_mlps[i](x)
            pose_tokens = residual + x  # Add residual connection

        # Final projection to target dimension
        pred_pose_enc = self.pose_branch(pose_tokens)

        # Apply activations
        activated_pose = activate_pose(
            pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
        )
        
        return activated_pose
    
class SimplifiedCameraHead(nn.Module):
    """
    Simplified CameraHead that predicts camera parameters in a single forward pass.

    This version removes the iterative refinement and adaptive modulation, making it
    a simpler and faster direct predictor.
    """
    def __init__(
        self,
        dim_in: int = 2048,
        pose_encoding_type: str = "absT_quaR_FoV",
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "sigmoid_range_fov",
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
            self.fl_act = "relu"
        elif pose_encoding_type == "absT_quaR_logK":
            self.target_dim = 11
            self.fl_act = "linear"
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)

        # Final MLP layer to predict the pose encoding from the processed token.
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list: list) -> torch.Tensor:
        """
        Forward pass to predict camera parameters in a single shot.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                                           the last tensor is used for prediction.

        Returns:
            torch.Tensor: The predicted camera encoding (post-activation).
        """
        tokens = aggregated_tokens_list[-1]

        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        pred_pose_enc = self.pose_branch(pose_tokens)

        activated_pose = activate_pose(
            pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
        )
        
        return activated_pose

class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
            self.fl_act = "relu"
        elif pose_encoding_type == "absT_quaR_logK":
            self.target_dim = 11
            self.fl_act = "linear"
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift


class ProgressiveCameraHead(nn.Module):
    """
    A camera head that progressively refines camera parameter predictions.

    This version uses a "FUSE-then-REFINE" strategy. For each specified
    intermediate layer, it FUSES the layer's token with the current
    refined state (via addition) and then passes the result through a
    dedicated REFINEMENT block.

    The refinement block is an efficient FFN (MLP) with a residual connection
    (x = x + MLP(Norm(x))), which is zero-initialized for training stability.
    """
    def __init__(
        self,
        dim_in: int = 2048,
        pose_encoding_type: str = "absT_quaR_FoV",
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "sigmoid_range_fov",
        intermediate_layer_idx: List[int] = [-1],
        mlp_ratio: int = 4,
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.intermediate_layer_idx = intermediate_layer_idx

        # Normalization for incoming tokens from the backbone
        self.token_norm = nn.LayerNorm(dim_in)

        # Final MLP layer to project the refined token into a pose
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

        # --- Efficient Refinement Blocks ---
        # We create one FFN block (Norm + MLP) for each refinement step.
        hidden_dim_refine = int(dim_in * mlp_ratio)
        
        self.refine_mlps = nn.ModuleList(
            [
                Mlp(in_features=dim_in, hidden_features=hidden_dim_refine, out_features=dim_in)
                for _ in range(len(self.intermediate_layer_idx))
            ]
        )
        self.refine_norms = nn.ModuleList(
            [
                nn.LayerNorm(dim_in)
                for _ in range(len(self.intermediate_layer_idx))
            ]
        )
        
        # Apply stable weight initialization
        self.apply(self._init_weights)
        self._init_refinement_weights()

    def _init_weights(self, m):
        """Standard ViT/Timm-style initialization."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_refinement_weights(self):
        """
        Zero-init the last linear layer of each refinement MLP.
        This makes the entire residual FFN block (x = x + MLP(Norm(x)))
        start as an identity function (x = x + 0), which is crucial
        for stable training of deep residual structures.
        """
        for mlp in self.refine_mlps:
            # --- This assumes your Mlp has a final layer 'fc2' or 'proj' ---
            layer_to_zero = None
            if hasattr(mlp, 'fc2'):
                layer_to_zero = mlp.fc2
            elif hasattr(mlp, 'proj'):
                layer_to_zero = mlp.proj
            
            if layer_to_zero is not None:
                nn.init.constant_(layer_to_zero.weight, 0)
                nn.init.constant_(layer_to_zero.bias, 0)
            else:
                print(f"Warning: {self.__class__.__name__} could not find "
                      f"the last layer of an Mlp to zero-init.")


    def forward(self, aggregated_tokens_list: list) -> torch.Tensor:
        """
        Forward pass: FUSE (add) new tokens and then REFINE (FFN block).
        """
        refined_token = None # This will hold our progressive state

        for i, layer_idx in enumerate(self.intermediate_layer_idx):
            tokens = aggregated_tokens_list[layer_idx]
            pose_tokens = tokens[:, :, 0]
            
            # Normalize the raw token from the backbone
            pose_tokens = self.token_norm(pose_tokens) 

            # --- 1. FUSE ---
            # Add the new information to our current refined state.
            if i == 0:
                refined_token = pose_tokens
            else:
                refined_token = refined_token + pose_tokens

            # --- 2. REFINE ---
            # Pass the *fused* state through its dedicated refinement block.
            # This is the stable residual block: x = x + MLP(Norm(x))
            norm_token = self.refine_norms[i](refined_token)
            refined_token = refined_token + self.refine_mlps[i](norm_token)

        # --- 3. PROJECT ---
        # The loop is finished. Project the final refined token to a pose.
        pred_pose_enc = self.pose_branch(refined_token)

        activated_pose = activate_pose(
            pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
        )

        return activated_pose
