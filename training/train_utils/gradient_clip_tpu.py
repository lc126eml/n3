# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class GradientClipperTPU:
    """
    TPU-oriented gradient clipping.

    Unlike the GPU clipper, this intentionally performs a single global
    clip over all trainable parameters to avoid multiple large XLA norm
    reduction graphs.
    """

    def __init__(self, configs, *args, **kwargs):
        del args, kwargs
        if not configs:
            raise ValueError("GradientClipperTPU requires one config")
        if len(configs) != 1:
            raise ValueError("GradientClipperTPU supports exactly one global clip config")
        config = configs[0]
        module_names = config.get("module_name", ["*"])
        if isinstance(module_names, str):
            module_names = [module_names]
        if module_names != ["*"]:
            raise ValueError(
                "GradientClipperTPU supports only one global clip config with module_name: ['*']"
            )
        self.configs = [
            {
                "module_names": module_names,
                "max_norm": float(config["max_norm"]) if config.get("max_norm") is not None else None,
                "norm_type": config.get("norm_type", 2),
            }
        ]
        self.params_to_clip = None
        self.is_initialized = False

    def setup_clipping(self, model: nn.Module) -> None:
        self.params_to_clip = [param for param in model.parameters() if param.requires_grad]
        self.is_initialized = True

    def __call__(self, model: nn.Module):
        del model
        if not self.is_initialized:
            raise RuntimeError("GradientClipperTPU must be initialized with setup_clipping() before use")
        config = self.configs[0]
        if not self.params_to_clip or config["max_norm"] is None:
            return {}
        grad_norm = nn.utils.clip_grad_norm_(
            self.params_to_clip,
            max_norm=config["max_norm"],
            norm_type=config["norm_type"],
        )
        if grad_norm is None:
            return {}
        return {"*": grad_norm.detach()}
