# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ResidualConvBlock and DecoderHead (conv mode) are adapted from MoGe
# (https://github.com/microsoft/MoGe), distributed by Microsoft under the MIT
# License. See THIRD_PARTY_LICENSES.md for the full license text.

"""Decoder heads for the DVLT model: spatial dense heads and camera head."""

from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class SimpleCameraHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, pose_dim=9):
        super().__init__()
        self.pose_dim = pose_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_pose = nn.Linear(hidden_dim, pose_dim)

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        patch_token_start: int,
    ) -> torch.Tensor:
        tokens = aggregated_tokens_list[-1]
        if tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which SimpleCameraHead needs.")

        num_tokens = tokens.shape[2]
        if patch_token_start is None:
            raise ValueError("patch_token_start is required for SimpleCameraHead")
        if patch_token_start > num_tokens:
            raise ValueError(f"patch_token_start ({patch_token_start}) exceeds token length ({num_tokens})")

        # The first aggregator token is the camera token for each frame.
        x = self.mlp(tokens[:, :, 0])

        with torch.amp.autocast("cuda", enabled=False):
            pose = self.fc_pose(x.float())
            pose = torch.cat([pose[..., :7], F.relu(pose[..., 7:])], dim=-1)
        return pose
