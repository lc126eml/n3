# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)

import copy
import torch
import torch.nn as nn
import os
from pathlib import Path
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf
from wcmatch import fnmatch





# ------------------------------------------------------------
# Glob‑matching flags (behave like the Unix shell) 
# ------------------------------------------------------------
GLOB_FLAGS = (
    fnmatch.CASE       # case‑sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out‑of‑the‑box
)




class DDPCheckpointSaver:
    def __init__(
        self,
        checkpoint_folder: str,
        checkpoint_names: List[str],
        rank: int,
        epoch: int,
    ):
        super().__init__()
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_names = checkpoint_names
        self.worker_id = rank
        self.epoch = epoch

    def save_model_checkpoint(
        self,
        model: nn.Module,
        ema_models: Optional[List[Any]] = None,
        skip_saving_parameters: Optional[List[str]] = None,
    ) -> None:
        checkpoint = {
            "model": exclude_params_matching_unix_pattern(
                patterns=skip_saving_parameters, state_dict=model.state_dict()
            )
        }
        if ema_models is not None:
            checkpoint["ema_models"] = [
                exclude_params_matching_unix_pattern(
                    patterns=skip_saving_parameters,
                    state_dict=ema_model.state_dict(),
                )
                for ema_model in ema_models
            ]

        # DDP model checkpoints are only saved on rank 0.
        # We CANNOT move this guard before the creation of the `state_dict`,
        # because some data loaders have implicit synchronization within their state.
        if self.worker_id != 0:
            return

        for ckpt_name in self.checkpoint_names:
            checkpoint_path = os.path.join(
                self.checkpoint_folder, f"{ckpt_name}.pt"
            )
            logging.info(
                f"Saving model checkpoint at epoch {self.epoch} to {checkpoint_path}"
            )
            robust_torch_save(checkpoint, checkpoint_path)

    def save_optimizer_checkpoint(self, optimizer_state_dicts: Any) -> None:
        if optimizer_state_dicts is None:
            return

        for ckpt_name in self.checkpoint_names:
            checkpoint_path = os.path.join(
                self.checkpoint_folder,
                f"{ckpt_name}_optimizer_{self.worker_id}.pt",
            )
            logging.info(
                f"Saving optimizer checkpoint at epoch {self.epoch} to {checkpoint_path}"
            )
            robust_torch_save(optimizer_state_dicts, checkpoint_path)

    def save_meta_checkpoint(self, meta: Dict[str, Any]) -> None:
        if self.worker_id != 0:
            return

        for ckpt_name in self.checkpoint_names:
            checkpoint_path = os.path.join(
                self.checkpoint_folder, f"{ckpt_name}_meta.pt"
            )
            logging.info(
                f"Saving meta checkpoint at epoch {self.epoch} to {checkpoint_path}"
            )
            robust_torch_save(meta, checkpoint_path)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer_state_dicts: Optional[Any] = None,
        meta: Optional[Dict[str, Any]] = None,
        ema_models: Optional[List[Any]] = None,
        skip_saving_parameters: Optional[List[str]] = None,
    ) -> None:
        self.save_model_checkpoint(
            model=model,
            ema_models=ema_models,
            skip_saving_parameters=skip_saving_parameters,
        )
        if optimizer_state_dicts is not None:
            self.save_optimizer_checkpoint(optimizer_state_dicts=optimizer_state_dicts)
        if meta is not None:
            self.save_meta_checkpoint(meta=meta)



def exclude_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns

    Args:
        patterns: the list of unix patterns to exclude
        state_dict: the dictionary to filter

    Returns:
        A new state dictionary
    """
    if len(patterns) == 0:
        return state_dict

    all_keys = list(state_dict.keys())
    excluded_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: v for k, v in state_dict.items() if k not in excluded_keys}






def unix_pattern_to_parameter_names(
    constraints: List[str], all_parameter_names: Sequence[str]
) -> Union[None, Set[str]]:
    """
    Go through the list of parameter names and select those that match
    any of the provided constraints
    """
    parameter_names = []
    for param_name in constraints:
        matching_parameters = set(
            fnmatch.filter(all_parameter_names, param_name, flags=GLOB_FLAGS)
        )
        assert (
            len(matching_parameters) > 0
        ), f"param_names {param_name} don't match any param in the given names."
        parameter_names.append(matching_parameters)
    return set.union(*parameter_names)




def robust_torch_save(checkpoint, checkpoint_path):
    """
    A more robust version of torch.save that works better with preemptions
    and corruptions if a job is preempted during save.
    """
    # Move the existing checkpoint to a backup location
    backup_checkpoint_path = checkpoint_path + ".bak"
    backup_checkpoint_path_saved = False
    if g_pathmgr.exists(checkpoint_path):
        assert not g_pathmgr.exists(
            backup_checkpoint_path
        ), f"this should not exist... {backup_checkpoint_path}"
        g_pathmgr.mv(checkpoint_path, dst_path=backup_checkpoint_path)
        backup_checkpoint_path_saved = True
    # Save the checkpoint
    with g_pathmgr.open(checkpoint_path, "wb") as f:
        torch.save(checkpoint, f)
    # Remove the backup checkpoint
    if backup_checkpoint_path_saved:
        g_pathmgr.rm(backup_checkpoint_path)


# ------------------------------------------------------------
# Resume helpers (shared by the launch scripts and Trainer)
# ------------------------------------------------------------

RESUME_CONFIG_SKIP_KEYS_HARDCODED = (
    "checkpoint.resume_checkpoint_path",
    "checkpoint.resume_config_skip_keys",
    "checkpoint.resume_optimizer",
    "logging.run_folder_name",
    "total_run_time_hr",
)


def resolve_resume_checkpoint_path(requested_path: str) -> Optional[str]:
    requested_path = str(requested_path)
    if not requested_path:
        return None
    if os.path.isfile(requested_path):
        return requested_path

    path_obj = Path(str(requested_path))
    parts = path_obj.parts
    if path_obj.is_absolute():
        if len(parts) < 4:
            return None
        search_root = Path(parts[0], parts[1], parts[2], parts[3], parts[4])
    else:
        if len(parts) < 3:
            return None
        search_root = Path(*parts[:3])

    if not search_root.is_dir():
        return None

    for found in search_root.rglob("checkpoint.pt"):
        if found.is_file():
            print(
                "resume_checkpoint_path '%s' is not a file; using discovered checkpoint '%s'"
                % (requested_path, found),
                flush=True,
            )
            return str(found)
    if parts[1] == "kaggle":
        search_root = Path(parts[0], parts[1], parts[2], parts[3], parts[5])
    for found in search_root.rglob("checkpoint.pt"):
        if found.is_file():
            print(
                "resume_checkpoint_path '%s' is not a file; using discovered checkpoint '%s'"
                % (requested_path, found),
                flush=True,
            )
            return str(found)
    return None


def load_resume_meta(cfg) -> Optional[Dict[str, Any]]:
    """Load the companion ``*_meta.pt`` for a full-resume checkpoint, if present.

    Returns None when no resume is configured, the path does not resolve, or no
    meta file exists (legacy single-file or weights-only checkpoints are handled
    by the Trainer fallback, avoiding a multi-GB full-file load at launch).
    """
    checkpoint_cfg = cfg.get("checkpoint")
    requested_path = checkpoint_cfg.get("resume_checkpoint_path") if checkpoint_cfg else None
    if not requested_path:
        return None
    ckpt_path = resolve_resume_checkpoint_path(str(requested_path))
    if ckpt_path is None or not str(ckpt_path).endswith(".pt"):
        return None
    meta_path = f"{str(ckpt_path)[:-3]}_meta.pt"
    if not g_pathmgr.exists(meta_path):
        return None
    with g_pathmgr.open(meta_path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)


def normalize_resume_skip_keys(cfg) -> List[str]:
    checkpoint_cfg = cfg.get("checkpoint", {})
    raw_skip_keys = checkpoint_cfg.get("resume_config_skip_keys", [])
    if OmegaConf.is_config(raw_skip_keys):
        raw_skip_keys = OmegaConf.to_container(raw_skip_keys, resolve=False)
    if raw_skip_keys is None:
        raw_skip_keys = []
    elif isinstance(raw_skip_keys, str):
        raw_skip_keys = [raw_skip_keys]
    elif not isinstance(raw_skip_keys, (list, tuple)):
        logging.warning(
            "checkpoint.resume_config_skip_keys must be a list/tuple of dot-path strings; "
            f"got {type(raw_skip_keys).__name__}. Ignoring it."
        )
        raw_skip_keys = []

    seen_skip_keys = set()
    resume_skip_keys: List[str] = []
    for key_path in [*RESUME_CONFIG_SKIP_KEYS_HARDCODED, *raw_skip_keys]:
        if not isinstance(key_path, str) or not key_path:
            print(f"Ignoring invalid resume_config_skip_keys entry: {key_path!r}", flush=True)
            continue
        if key_path in seen_skip_keys:
            continue
        seen_skip_keys.add(key_path)
        resume_skip_keys.append(key_path)
    return resume_skip_keys


def merge_resume_config(cfg, resume_meta: Dict[str, Any]):
    """Merge a resumed run's trainer_config over cfg, honoring skip keys.

    Idempotent: re-applying it to an already-merged config yields the same result.
    """
    resume_cfg = resume_meta.get("trainer_config") if isinstance(resume_meta, dict) else None
    if resume_cfg is None:
        print("Checkpoint does not contain trainer_config; cannot resume with minimal config.")
        return cfg
    merged = OmegaConf.create(resume_cfg)
    resume_skip_keys = normalize_resume_skip_keys(cfg)
    _missing = object()
    for key_path in resume_skip_keys:
        try:
            value = OmegaConf.select(cfg, key_path, default=_missing)
            if value is _missing:
                continue
            OmegaConf.update(merged, key_path, copy.deepcopy(value), merge=False)
        except Exception as exc:
            print(
                f"Failed to apply resume_config_skip_keys override for '{key_path}': {exc}",
                flush=True,
            )
    base_log_dir = merged.get("logging", {}).get("log_dir")
    new_run_folder = merged.get("logging", {}).get("run_folder_name")
    if base_log_dir and new_run_folder:
        base_parent = os.path.dirname(os.path.normpath(base_log_dir))
        merged.logging.log_dir = os.path.join(base_parent, str(new_run_folder))
        if merged.get("logging", {}).get("tensorboard_writer"):
            merged.logging.tensorboard_writer.path = os.path.join(
                merged.logging.log_dir, "tensorboard"
            )
        if merged.get("logging", {}).get("csv_writer"):
            merged.logging.csv_writer.path = os.path.join(merged.logging.log_dir, "csv")
        if merged.get("checkpoint"):
            merged.checkpoint.save_dir = os.path.join(
                merged.logging.log_dir, "ckpts"
            )
    if merged.get("checkpoint"):
        merged.checkpoint.filter_keys = OmegaConf.create({"enabled": False})
    return merged
