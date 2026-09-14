# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from wcmatch import fnmatch
from functools import wraps
from typing import List
import math

import torch.nn as nn

# ------------------------------------------------------------
# Glob‑matching flags (behave like the Unix shell) 
# ------------------------------------------------------------
GLOB_FLAGS = (
    fnmatch.CASE       # case‑sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out‑of‑the‑box
)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Use canonical module names through compilation or parallel wrappers."""
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module
        else:
            return model


def select_unfreeze_modules(model: nn.Module, selectors) -> list[nn.Module]:
    """Resolve recursive globs and paired, end-exclusive aggregator block ranges."""
    model = unwrap_model(model)
    if not selectors or set(selectors) - {"patterns", "frame_block_ranges"}:
        raise ValueError("modules requires patterns and/or frame_block_ranges")
    patterns = selectors.get("patterns", [])
    ranges = selectors.get("frame_block_ranges", [])
    if isinstance(patterns, str) or isinstance(ranges, str):
        raise ValueError("patterns and frame_block_ranges must be lists")
    selected = {}
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("Module patterns must be nonempty strings")
        matches = {name: mod for name, mod in model.named_modules()
                   if fnmatch.fnmatch(name, pattern, flags=GLOB_FLAGS)}
        if not matches:
            raise ValueError(f"Unfreeze pattern matched nothing: {pattern!r}")
        selected.update(matches)
    for bounds in ranges:
        if (not hasattr(bounds, "__len__") or len(bounds) != 2
                or any(type(i) is not int for i in bounds)):
            raise ValueError(f"Block range must be [start, end] integers: {bounds}")
        start, end = bounds
        for name in ("frame_blocks", "inter_frame_blocks"):
            blocks = getattr(getattr(model, "aggregator", None), name, None)
            if blocks is None or not 0 <= start < end <= len(blocks):
                raise ValueError(f"Invalid range {bounds} for aggregator.{name}")
            for index in range(start, end):
                selected[f"aggregator.{name}.{index}"] = blocks[index]
    if not selected:
        raise ValueError("An unfreeze stage must select at least one module")
    return list(selected.values())


def validate_unfreeze_configs(model: nn.Module, configs) -> None:
    for index, conf in enumerate(configs):
        try:
            epoch = conf.get("epoch")
            if type(epoch) is not int or epoch < 0:
                raise ValueError("epoch must be a nonnegative integer")
            discount = float(conf["batch_cost_discount"])
            if not math.isfinite(discount) or discount <= 0:
                raise ValueError("batch_cost_discount must be finite and positive")
            select_unfreeze_modules(model, conf["modules"])
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            raise ValueError(f"Invalid optim.unfreeze_configs[{index}]: {exc}") from exc


def freeze_modules(model: nn.Module, patterns: List[str], recursive: bool = True) -> nn.Module:
    """Freeze (stop training) parts of *model* whose *name* matches *patterns*.

    Parameters
    ----------
    model : nn.Module
        The complete model you are working with.
    patterns : list[str]
        Glob patterns to match sub‑module names.  Example: ``["encoder.*", "cls_head"]``
    recursive : bool, default = True
        • ``True``  → also freeze every child of a matched module.
        • ``False`` → freeze only the matched module itself.

    Returns
    -------
    nn.Module
        The same model object, now with some parts frozen.

    Example
    -------
    >>> freeze_modules(model, ["encoder.*", "decoder.layer1"], recursive=True)
    """
    matched: set[str] = set()

    for name, mod in model.named_modules():
        # does *name* match ANY user pattern?
        if any(fnmatch.fnmatch(name, p, flags=GLOB_FLAGS) for p in patterns):
            matched.add(name)
            freeze(mod, recursive)

    _check_every_pattern_used(matched, patterns)
    return model


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def freeze(mod: nn.Module, recursive: bool = True) -> None:
    """
    Put *mod* in eval mode and disable gradients.
    Does NOT modify mod.train.
    """

    # 1. Set eval mode
    if recursive:
        mod.eval()  # affects entire subtree
    else:
        mod.training = False  # only this module

    # 2. Disable gradients
    param_iter = (
        mod.parameters()
        if recursive
        else mod.parameters(recurse=False)
    )
    for p in param_iter:
        p.requires_grad = False

def unfreeze_ignore(mod: nn.Module, recursive: bool = True, ignore_names: list[str] = ["patch_embeddings"]) -> None:
    """
    Put *mod* back in train mode and enable gradients.
    Does NOT restore previous state — just unfreezes.
    
    Args:
        mod: The neural network module to unfreeze.
        recursive: If True, affects the entire subtree. If False, only affects the top-level module.
        ignore_names: A list of substrings. If a parameter's name contains any of these, 
                      its gradient will remain frozen.
    """

    # 1. Set train mode
    if recursive:
        mod.train(True)  # affects entire subtree
    else:
        mod.training = True  # only this module

    # 2. Enable gradients selectively
    param_iter = (
        mod.named_parameters()
        if recursive
        else mod.named_parameters(recurse=False)
    )
    
    for name, p in param_iter:
        # Check if any string in ignore_names matches a portion of the parameter name
        if any(ignore_word in name for ignore_word in ignore_names):
            continue  # Keep this parameter frozen, skip to the next
            
        p.requires_grad = True

def unfreeze(mod: nn.Module, recursive: bool = True) -> None:
    """
    Put *mod* back in train mode and enable gradients.
    Does NOT restore previous state — just unfreezes.
    """

    # 1. Set train mode
    if recursive:
        mod.train(True)  # affects entire subtree
    else:
        mod.training = True  # only this module

    # 2. Enable gradients
    param_iter = (
        mod.parameters()
        if recursive
        else mod.parameters(recurse=False)
    )
    for p in param_iter:
        p.requires_grad = True

def _check_every_pattern_used(matched_names: set[str], patterns: List[str]):
    unused = [p for p in patterns if not any(fnmatch.fnmatch(n, p, flags=GLOB_FLAGS)
                                             for n in matched_names)]
    if unused:
        raise ValueError(f"These patterns matched nothing: {unused}")
