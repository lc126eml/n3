# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
# Plan: This file should provide a trainer class, which provides
# 1. The init of DDP                                            Done
# 2. The init of optimizers, tb, timers, and so on               Done
# 3. A basic training framework (especially for finetuning)
#       self._train_epoch_                                     Done
#       self._process_batch_                                  Done
#       self._step_                                           Done
# 4. The training loop: more utils to be added
'''
import contextlib
import atexit

import copy
import functools
import gc
import json
import logging
import math
import os
import sys
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import random
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import subprocess
import torch.distributed as dist
import torch.nn as nn
import torchvision


import fvcore
from einops import rearrange
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from train_utils.csv_writer import CsvLogger

from datetime import timedelta

from safetensors.torch import load_file
from train_utils.priority_lock import PriorityLock


def _setup_project_root_from_file(start_file: str) -> None:
    env_project_root = os.environ.get("PROJECT_ROOT")
    if env_project_root:
        project_root = Path(env_project_root).resolve()
    else:
        p = Path(start_file).resolve()
        project_root = None
        for parent in [p.parent, *p.parents]:
            if (parent / ".project-root").exists():
                project_root = parent
                break
        if project_root is None:
            project_root = p.parent.parent
        os.environ.setdefault("PROJECT_ROOT", str(project_root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_setup_project_root_from_file(__file__)
# 
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules, unfreeze
from train_utils.optimizer import construct_optimizers
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.checkpoint import DDPCheckpointSaver

from eval_utils.eval_wrapper import eval_batch
from eval_utils.align_utils.align_camera import align_camera_and_points_batch_ext
from eval_utils.align_utils.umeyama_alignment import align_pred_to_gt_torch_batch, align_extrinsics_torch, align_pred_to_gt_torch_batch_roma, align_c2w_poses_points_torch, align_rotation_only_torch, align_c2w_poses_points_rotation_only
from eval_utils.normalize_utils.normalize_pc import normalize_depth_cam_extrinsics
from eval_utils.align_utils.depth_median_scaling import median_scale_depth_torch, median_scale_depth_torch_batch
from eval_utils.normalize_utils.normalize_pc import normalize_pointcloud_vggt, normalize_pr_pointcloud, normalize_pointcloud_invariant, calculate_depth_scale
from eval_utils.transform_utils import global_points_from_cam, cam_points_from_depth
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from dust3r.utils.camera import center_c2w_poses_batch, get_pred_world_to_gt_world_transforms

def get_amp_type(amp_dtype: str):
    assert amp_dtype in ["bfloat16", "float16"], f"Invalid Amp type: {amp_dtype}"
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float16


class Trainer:
    """
    Trainer supporting the DDP training strategies.
    """

    EPSILON = 1e-8
    _RESUME_CONFIG_SKIP_KEYS_HARDCODED = (
        "checkpoint.resume_checkpoint_path",
        "checkpoint.resume_config_skip_keys",
        "logging.run_folder_name",
    )

    def __init__(self, cfg: DictConfig):
        # --- Acquire a file lock to ensure exclusive GPU usage ---
        lock_path = "/tmp/gpu.lock"
        lock_priority = int(os.environ.get("GPU_LOCK_PRIORITY", "10"))
        self.gpu_lock = PriorityLock(lock_dir=lock_path, priority=lock_priority)
        print(f"Attempting to acquire lock on '{lock_path}' (priority={lock_priority})...")
        self.gpu_lock.acquire()
        print("Lock acquired. It is safe to proceed.")
        atexit.register(self.gpu_lock.release)

        self._scalar_log_keys_cache = {}
        self._setup_timers()
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)

        self._resume_ckpt_path = None
        self._resume_checkpoint = None
        self._resume_checkpoint_amp = None
        self._trainer_config_snapshot = None
        self.data_module = None

        cfg = self._merge_resume_config(cfg)
        cfg = self._resolve_conf_logit_max(cfg)
        self.accum_steps = cfg.get("accum_steps", 1)
        self.accumulation_mode = cfg.optim.get("accumulation_mode", "chunk_within_batch")
        if self.accumulation_mode == "across_batches" and self._resume_checkpoint is None:
            cfg.logging.log_freq = cfg.logging.log_freq * self.accum_steps
        self.cfg = cfg

        self._setup_env_variables(cfg.get("env_variables"))

        self.data_conf = cfg.data
        self.model_conf = cfg.model
        self.loss_conf = cfg.loss
        self.logging_conf = cfg.logging
        self.checkpoint_conf = cfg.checkpoint
        self.postprocess_conf = cfg.postprocess

        # hyperparameters
        self.log_per_optimizer_step = cfg.optim.get("log_per_optimizer_step", False)
        self.max_epochs = cfg.max_epochs
        self.mode = cfg.get("mode", "train")
        self.val_epoch_freq = cfg.get("val_epoch_freq", 1)
        self.limit_train_batches = cfg.get("limit_train_batches")
        self.limit_val_batches = cfg.get("limit_val_batches")
        self.optim_conf = cfg.optim
        self.compile_conf = cfg.get("compile")
        self.env_variables = cfg.get("env_variables")
        self.device_conf = cfg.get("device", "cuda")
        self.cuda_conf = cfg.get("cuda")

        self.where = 0.0
        self.seed_value = cfg.get("seed_value", 123)
        self.total_run_time_hr = cfg.get("total_run_time_hr")
        self.resume_bs = cfg.get("resume_bs", False)

        log_dir = self.logging_conf.log_dir
        exp_name = cfg.get("exp_name")
        suffix_parts = []
        align_conf = self.postprocess_conf.get("train", {}).get("align", {})
        for key in [
            "to_first_cam",
            "pr_align_cam",
            "pred_center",
            "center_world",
            "pts_align_to_gt",
            "pts_align_to_gt_rot",
            "gt_align_to_pts",
            "depth_align_to_gt",
        ]:
            if align_conf.get(key, {}).get("enabled"):
                suffix_parts.append(key)

        train_augs = self.data_conf.data_module.get("train_config", {}).get("augs", {})
        random_crop_prob_schedule = train_augs.get("random_crop_prob_schedule")
        # has_aug_schedule = any(
        #     key in train_augs for key in ["random_crop_prob_schedule", "prot_schedule", "pcrop_schedule"]
        # )
        # if has_aug_schedule:
        #     self._base_train_augs = self._extract_train_augs(self.data_conf)
        #     self._last_train_augs = None
        # else:
        #     self._base_train_augs = None

        # self._setup_dataloaders()
        # import shutil
        # shutil.rmtree(os.environ.get("PROJECT_ROOT"))
        # sys.exit(0)
        self._base_train_augs = None
        if random_crop_prob_schedule is not None:
            self._base_train_augs = self._extract_train_augs(self.data_conf)
            self._last_train_augs = None

            start = float(random_crop_prob_schedule.get("start", 0.0))
            end = float(random_crop_prob_schedule.get("end", start))
            start_epoch = int(random_crop_prob_schedule.get("start_epoch", 0))
            end_epoch = int(random_crop_prob_schedule.get("end_epoch", start_epoch))
            suffix_parts.append(f"rcropPr{start:.2f}-{end:.2f}e{start_epoch}-{end_epoch}")
        else:
            random_crop_prob = train_augs.get("random_crop_prob", train_augs.get("random_crop"))
            if isinstance(random_crop_prob, bool):
                random_crop_prob = 1.0 if random_crop_prob else 0.0
            if random_crop_prob is not None:
                suffix_parts.append(f"rcropP{float(random_crop_prob):.2f}")

        lr_schedulers = getattr(self.optim_conf.options, "lr", [])
        try:
            base_lr = lr_schedulers[0].scheduler.schedulers[0].end_value
        except Exception:
            base_lr = None
        if base_lr is not None:
            suffix_parts.append(f"lr{int(base_lr * 1e5)}")
        suffix_parts.append(f"e{self.max_epochs}")
        if exp_name and suffix_parts:
            suffix = "_".join(suffix_parts)
            new_exp_name = f"{exp_name}/{suffix}"
            if not (isinstance(log_dir, str) and new_exp_name in log_dir):
                if isinstance(log_dir, str) and exp_name in log_dir:
                    log_dir = log_dir.replace(exp_name, new_exp_name, 1)
                else:
                    run_folder_name = self.logging_conf.get("run_folder_name")
                    if run_folder_name:
                        log_dir = os.path.join(log_dir, new_exp_name, run_folder_name)
                    else:
                        log_dir = os.path.join(log_dir, new_exp_name)
            self.logging_conf.log_dir = log_dir
        

        self.checkpoint_conf.save_dir = os.path.join(log_dir, "ckpts")
        
        safe_makedirs(self.logging_conf.log_dir)
        print(self.logging_conf.log_dir)
        self._write_run_metadata()

        self._setup_device(self.device_conf)
        self._setup_cuda_backend(self.cuda_conf)
        torch.set_float32_matmul_precision("high")
        if self.compile_conf and self.compile_conf.get("enabled"):
            try:
                import torch._dynamo as dynamo
                dynamo.config.suppress_errors = True
                dynamo.config.capture_scalar_outputs = True
            except Exception as exc:
                logging.warning(f"Failed to set torch._dynamo configs: {exc}")
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=0,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(self.seed_value, self.max_epochs, 0)
        amp_conf = getattr(self.optim_conf, "amp", None)
        if amp_conf is not None and bool(amp_conf.enabled):
            self.amp_type = get_amp_type(amp_conf.amp_dtype)
        else:
            self.amp_type = torch.float32

        self._setup_components()  # Except Optimizer everything is setup here.
        self._setup_dataloaders()

        self.model.to(self.device)
        if self.scaler:
            copy_data_to_device(self.scaler, self.device)

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        if self.mode != "val":
            self.optims = construct_optimizers(
                self.model,
                self.optim_conf,
            )

        self.csv_logger = None
        if self.logging_conf.get("csv_writer") and self.logging_conf.csv_writer.get("enabled"):
            csv_conf = self.logging_conf.csv_writer
            csv_path = os.path.join(csv_conf.path, csv_conf.filename)
            
            self.csv_logger = CsvLogger(csv_path)


        ################################
        # If you want to force to resume from a specific checkpoint, you can do so by setting the resume_checkpoint_path in the config
        if self._resume_ckpt_path is None:
            self._resume_ckpt_path = self._resolve_resume_checkpoint_path()
        if self._resume_ckpt_path is not None:
            if self._resume_checkpoint is None:
                self._resume_checkpoint = self._load_checkpoint_file(self._resume_ckpt_path)
            if self._resume_checkpoint_amp is None and self._resume_checkpoint:
                resume_cfg = self._resume_checkpoint.get("trainer_config")
                if resume_cfg:
                    self._resume_checkpoint_amp = resume_cfg.get("optim", {}).get("amp", None)
            self._load_resuming_checkpoint(self._resume_ckpt_path, checkpoint=self._resume_checkpoint)

        # Save the full config for reproducibility (after applying resume overrides).
        if self.mode != "val":
            conf_to_save = OmegaConf.create(self.cfg)
            self._trainer_config_snapshot = conf_to_save
            config_path = os.path.join(self.logging_conf.log_dir, "trainer_config.yaml")
            with g_pathmgr.open(config_path, "w") as f:
                f.write(OmegaConf.to_yaml(conf_to_save))
            print(f"Saved trainer config to {config_path}")

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _write_run_metadata(self) -> None:
        git_hash = None
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(__file__),
                stderr=subprocess.DEVNULL,
            ).decode("utf-8").strip()
        except Exception:
            git_hash = None

        metadata = {
            "git_commit": git_hash,
            "cmdline": " ".join(sys.argv),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_path = os.path.join(self.logging_conf.log_dir, "run_metadata.json")
        try:
            with g_pathmgr.open(meta_path, "w") as f:
                f.write(json.dumps(metadata, indent=2))
        except Exception as exc:
            logging.warning(f"Failed to write run metadata: {exc}")

    def _resolve_resume_checkpoint_path(self) -> Optional[str]:
        if not self.checkpoint_conf:
            return None
        if self.checkpoint_conf.get("resume_checkpoint_path") is not None:
            return self.checkpoint_conf.resume_checkpoint_path
        return None

    def _normalize_resume_skip_keys(self, cfg: DictConfig) -> List[str]:
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
        for key_path in [*self._RESUME_CONFIG_SKIP_KEYS_HARDCODED, *raw_skip_keys]:
            if not isinstance(key_path, str) or not key_path:
                logging.warning(f"Ignoring invalid resume_config_skip_keys entry: {key_path!r}")
                continue
            if key_path in seen_skip_keys:
                continue
            seen_skip_keys.add(key_path)
            resume_skip_keys.append(key_path)
        return resume_skip_keys

    def _merge_resume_config(self, cfg: DictConfig) -> DictConfig:
        if not cfg.get("checkpoint") or not cfg.checkpoint.get("resume_checkpoint_path"):
            return cfg
        self._resume_ckpt_path = cfg.checkpoint.resume_checkpoint_path
        self._resume_checkpoint = self._load_checkpoint_file(self._resume_ckpt_path)
        if not self._resume_checkpoint or not isinstance(self._resume_checkpoint, dict):
            raise ValueError(f"Checkpoint could not be loaded: {self._resume_ckpt_path}")
        resume_cfg = self._resume_checkpoint.get("trainer_config")
        if resume_cfg is None:
            raise ValueError("Checkpoint does not contain trainer_config; cannot resume with minimal config.")
        self._resume_checkpoint_amp = resume_cfg.get("optim", {}).get("amp", None)
        base_cfg = OmegaConf.create(resume_cfg)
        merged = OmegaConf.merge(cfg, base_cfg)
        resume_skip_keys = self._normalize_resume_skip_keys(cfg)
        _missing = object()
        for key_path in resume_skip_keys:
            try:
                value = OmegaConf.select(cfg, key_path, default=_missing)
                if value is _missing:
                    continue
                OmegaConf.update(merged, key_path, copy.deepcopy(value), merge=False)
            except Exception as exc:
                logging.warning(f"Failed to apply resume_config_skip_keys override for '{key_path}': {exc}")
        base_log_dir = base_cfg.get("logging", {}).get("log_dir")
        new_run_folder = merged.get("logging", {}).get("run_folder_name")
        if base_log_dir and new_run_folder:
            base_parent = os.path.dirname(os.path.normpath(base_log_dir))
            merged.logging.log_dir = os.path.join(base_parent, str(new_run_folder))
            if merged.get("logging", {}).get("tensorboard_writer"):
                merged.logging.tensorboard_writer.path = os.path.join(
                    merged.logging.log_dir, "tensorboard"
                )
            if merged.get("logging", {}).get("csv"):
                merged.logging.csv.path = os.path.join(merged.logging.log_dir, "csv")
            if merged.get("checkpoint"):
                merged.checkpoint.save_dir = os.path.join(
                    merged.logging.log_dir, "ckpts"
                )
        if merged.get("checkpoint"):
            merged.checkpoint.filter_keys = OmegaConf.create({"enabled": False})
        return merged

    def _load_checkpoint_file(self, ckpt_path: str) -> Optional[Dict[str, Any]]:
        if not ckpt_path:
            return None
        if ckpt_path.endswith(".safetensors"):
            try:
                return load_file(ckpt_path)
            except Exception as exc:
                logging.error(f"Error loading safetensors file: {exc}")
                return None
        with g_pathmgr.open(ckpt_path, "rb") as f:
            return torch.load(f, map_location="cpu", weights_only=False)

    def _maybe_to_container(self, cfg: Any) -> Any:
        if cfg is None:
            return None
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
        return copy.deepcopy(cfg)

    def _merge_loss_conf_from_module(self, loss_conf: Any) -> Any:
        if not self.loss:
            return self._maybe_to_container(loss_conf)
        merged = self._maybe_to_container(loss_conf)
        if merged is None:
            merged = {}
        for key in (
            "camera",
            "angle_pose",
            "depth",
            "point",
            "track",
            "switch",
            "regulize_scale",
            "vggt",
        ):
            if hasattr(self.loss, key):
                merged[key] = self._maybe_to_container(getattr(self.loss, key))
        return merged

    def _build_trainer_config_snapshot(self) -> Dict[str, Any]:
        return {
            "data": self._maybe_to_container(self.data_conf),
            "model": self._maybe_to_container(self.model_conf),
            "logging": self._maybe_to_container(self.logging_conf),
            "checkpoint": self._maybe_to_container(self.checkpoint_conf),
            "max_epochs": self.max_epochs,
            "mode": self.mode,
            "device": self.device_conf,
            "seed_value": self.seed_value,
            "val_epoch_freq": self.val_epoch_freq,
            "cuda": self._maybe_to_container(self.cuda_conf),
            "limit_train_batches": self.limit_train_batches,
            "limit_val_batches": self.limit_val_batches,
            "optim": self._maybe_to_container(self.optim_conf),
            "loss": self._merge_loss_conf_from_module(self.loss_conf),
            "env_variables": self._maybe_to_container(self.env_variables),
            "accum_steps": self.accum_steps,
            "postprocess": self._maybe_to_container(self.postprocess_conf),
            "compile": self._maybe_to_container(self.compile_conf),
        }

    def _apply_resume_config(self, resume_cfg: DictConfig, current_overrides: Optional[Dict[str, Any]] = None) -> None:
        # Deprecated: resume config is merged before init.
        return


    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters


    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        print(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_cuda_backend(self, cuda_conf) -> None:
        self.rank = 0
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

    @staticmethod
    def _update_ckpt_keys_revised(ckpt, heads_to_keep=None, heads_to_discard=None, default_keep=True):
        """
        Helper function to selectively keep, discard, and rename keys from a checkpoint's state_dict.
        """
        if heads_to_keep is None:
            heads_to_keep = []
        if heads_to_discard is None:
            heads_to_discard = []

        new_ckpt = {}

        for key, value in ckpt.items():
            discard = False
            for head in heads_to_discard:
                if key.startswith(head):
                    discard = True
                    break
            if discard:
                continue

            processed = False
            for old_prefix, new_prefix in heads_to_keep:
                if old_prefix == "*":
                    new_key = f"{new_prefix}{key}"
                    new_ckpt[new_key] = value
                    processed = True
                    break
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    new_ckpt[new_key] = value
                    processed = True
                    break  # Process with the first matching rule

            if processed:
                continue

            if default_keep:
                new_ckpt[key] = value

        return new_ckpt

    def _load_resuming_checkpoint(self, ckpt_path: str, checkpoint: Optional[Dict[str, Any]] = None):
        # This method seems fine for single GPU as it loads to CPU first.
        logging.info(f"Resuming training from {ckpt_path}")
        if checkpoint is None:
            checkpoint = self._load_checkpoint_file(ckpt_path)
        if checkpoint is None:
            logging.warning("Checkpoint could not be loaded; skipping resume.")
            return
            
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        
        if self.checkpoint_conf.get("filter_keys") and self.checkpoint_conf.filter_keys.get("enabled"):
            if "trainer_config" in checkpoint:
                logging.info("Checkpoint filter_keys is enabled but resume detected; skipping key filtering.")
            else:
                filter_conf = self.checkpoint_conf.filter_keys
                logging.info("Filtering checkpoint keys before loading.")
                model_state_dict = self._update_ckpt_keys_revised(
                    model_state_dict,
                    heads_to_keep=filter_conf.get("heads_to_keep"),
                    heads_to_discard=filter_conf.get("heads_to_discard"),
                    default_keep=filter_conf.get("default_keep", True)
                )

        missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=self.checkpoint_conf.strict)
        
        if missing_keys:
            logging.warning(f"Missing keys when loading model state dict: {missing_keys}")
        else:
            logging.info(f"No missing keys when loading model state dict")
            
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading model state dict: {unexpected_keys}")
        else:
            logging.info(f"No unexpected keys when loading model state dict")
        logging.info(f"Loading the optimizer state dict")
        if "optimizer" in checkpoint and self.optims:
            opt_state = checkpoint["optimizer"]
            if isinstance(opt_state, list):
                if len(opt_state) != len(self.optims):
                    logging.warning(
                        f"Optimizer state count ({len(opt_state)}) does not match current optimizers "
                        f"({len(self.optims)}); restoring the first {min(len(opt_state), len(self.optims))} only."
                    )
                for optim, state in zip(self.optims, opt_state):
                    optim.optimizer.load_state_dict(state)
            else:
                if len(self.optims) == 1:
                    self.optims[0].optimizer.load_state_dict(opt_state)
                else:
                    logging.warning("Optimizer state is not a list but multiple optimizers exist; skipping restore.")

        loaded_epoch = None
        if "epoch" in checkpoint:
            loaded_epoch = checkpoint["epoch"]
        elif "prev_epoch" in checkpoint:
            loaded_epoch = checkpoint["prev_epoch"]
        if loaded_epoch is not None:
            if self.mode == "val":
                self.epoch = loaded_epoch
            else:
                self.epoch = loaded_epoch + 1

        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        if "scaler" in checkpoint and self.optim_conf.amp.enabled:
            if self._resume_checkpoint_amp is None:
                self.scaler.load_state_dict(checkpoint["scaler"])
            else:
                saved_amp = self._resume_checkpoint_amp
                saved_enabled = saved_amp.get("enabled") if isinstance(saved_amp, dict) else None
                saved_dtype = saved_amp.get("amp_dtype") if isinstance(saved_amp, dict) else None
                current_enabled = self.optim_conf.amp.enabled
                current_dtype = self.optim_conf.amp.amp_dtype
                if saved_enabled == current_enabled and saved_dtype == current_dtype:
                    self.scaler.load_state_dict(checkpoint["scaler"])
                else:
                    logging.warning(
                        "Skipping scaler restore due to AMP config mismatch: "
                        f"saved enabled={saved_enabled}, dtype={saved_dtype}; "
                        f"current enabled={current_enabled}, dtype={current_dtype}."
                    )

        if checkpoint.get("rng_state"):
            rng_state = checkpoint["rng_state"]
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"])
            if "cuda" in rng_state and rng_state["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "python" in rng_state:
                random.setstate(rng_state["python"])
        if checkpoint.get("train_dataset_checkpoint_state") is not None:
            self._restore_train_dataset_checkpoint_state(checkpoint["train_dataset_checkpoint_state"])


    def _setup_device(self, device):
        self.local_rank = 0
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _resolve_conf_logit_max(self, cfg: DictConfig) -> DictConfig:
        if cfg is None or "model" not in cfg or "conf_logit_max" not in cfg.model:
            return cfg
        current = cfg.model.get("conf_logit_max")
        if current is not None:
            return cfg
        dtype = torch.float32
        if cfg.optim.amp.enabled:
            amp_dtype = cfg.optim.amp.amp_dtype
            if amp_dtype == "float16":
                dtype = torch.float16
            elif amp_dtype == "bfloat16":
                dtype = torch.bfloat16
        max_logit = math.log(torch.finfo(dtype).max) - 1.0
        cfg.model.conf_logit_max = float(max_logit)
        return cfg


    def _setup_components(self):
        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}
        self.meters = None

        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names}"
            )

        model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
        model_summary(self.model, log_file=model_summary_path)
        logging.info(f"Model summary saved to {model_summary_path}")

        if self.compile_conf and self.compile_conf.get("enabled"):
            if not hasattr(torch, "compile"):
                logging.warning("torch.compile requested but not available; skipping.")
            else:
                compile_kwargs = dict(self.compile_conf)
                compile_kwargs.pop("enabled", None)
                try:
                    self.model = torch.compile(self.model, **compile_kwargs)
                    logging.info(f"torch.compile enabled with options: {compile_kwargs}")
                except Exception as exc:
                    logging.warning(f"torch.compile failed; continuing without compile. Error: {exc}")

        # TODO: Remind myself to finish this
        # Clean the dirty loss and build a single object
        self.loss = instantiate(self.loss_conf, _recursive_=False)


        # GradScaler is only needed for fp16 AMP (not bf16/float32).
        scaler_enabled = (
            torch.cuda.is_available()
            and self.optim_conf.amp.enabled
            and str(self.optim_conf.amp.amp_dtype) == "float16"
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)

        logging.info("Successfully initialized all training components: model, loss function, optimizer, and etc.")



    def _setup_dataloaders(self):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Instantiate the data module from the config
        data_module = instantiate(self.data_conf.data_module, _recursive_=False)
        data_module.seed = self.seed_value
        self.data_module = data_module

        if self.mode in ["train", "val"]:
            # Get the validation dataloader from the data module
            self.val_loader = data_module.val_dataloader()
            self.test_loader = data_module.test_dataloader()

        if self.mode in ["train"]:
            # Get the training dataloader from the data module
            self.train_loader = data_module.train_dataloader()


    def _move_to_device(self):
        print(
            f"Moving components to device {self.device}."
        )
        self.model.to(self.device)

        if self.loss:
            copy_data_to_device(self.loss, self.device)
        if self.scaler:
            copy_data_to_device(self.scaler, self.device)
        for meter in self._get_meters().values():
            meter.set_sync_device(self.device)

        print(
            f"Done moving components to device {self.device}."
        )

    def save_checkpoint(self, epoch, checkpoint_names=None):        
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        # if checkpoint_names is None:
        checkpoint_names = ["checkpoint"]
        if not (self.checkpoint_conf.save_freq > 0 and int(epoch + 1) % self.checkpoint_conf.save_freq == 0):
            return
            # if (
            #     self.checkpoint_conf.save_freq > 0
            #     and int(epoch) % self.checkpoint_conf.save_freq == 0
            #     and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            # ):
            #     checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "epoch": epoch,
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
            "trainer_config": OmegaConf.to_container(self.cfg, resolve=True),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        train_dataset_checkpoint_state = self._get_train_dataset_checkpoint_state()
        if train_dataset_checkpoint_state is not None:
            checkpoint_content["train_dataset_checkpoint_state"] = train_dataset_checkpoint_state
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=0,
            epoch=epoch,
        )

        saver.save_checkpoint(
            model=self.model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )



    def _get_train_dataset_checkpoint_state(self):
        state = {}

        data_module = getattr(self, "data_module", None)
        if data_module is not None and hasattr(data_module, "state_dict"):
            try:
                data_module_state = data_module.state_dict()
                if data_module_state is not None:
                    state["data_module"] = data_module_state
            except Exception as exc:
                logging.warning(f"Failed to collect data_module checkpoint state: {exc}")

        train_loader = getattr(self, "train_loader", None)
        if train_loader is None:
            return state or None

        loader_generator = getattr(train_loader, "generator", None)
        if isinstance(loader_generator, torch.Generator):
            try:
                state["loader_generator_state"] = loader_generator.get_state()
            except Exception as exc:
                logging.warning(f"Failed to collect train loader generator state: {exc}")

        for key, obj in (
            ("dataset", getattr(train_loader, "dataset", None)),
            ("sampler", getattr(train_loader, "sampler", None)),
            ("batch_sampler", getattr(train_loader, "batch_sampler", None)),
        ):
            if obj is None:
                continue
            if hasattr(obj, "state_dict"):
                try:
                    obj_state = obj.state_dict()
                    if obj_state is not None:
                        state[key] = obj_state
                except Exception as exc:
                    logging.warning(f"Failed to collect train loader {key} state: {exc}")
            obj_generator = getattr(obj, "generator", None)
            if isinstance(obj_generator, torch.Generator):
                try:
                    state[f"{key}_generator_state"] = obj_generator.get_state()
                except Exception as exc:
                    logging.warning(f"Failed to collect train loader {key} generator state: {exc}")

        return state or None

    def _restore_train_dataset_checkpoint_state(self, checkpoint_state):
        if not checkpoint_state:
            return

        data_module = getattr(self, "data_module", None)
        if data_module is not None and "data_module" in checkpoint_state and hasattr(data_module, "load_state_dict"):
            try:
                data_module.load_state_dict(checkpoint_state["data_module"])
            except Exception as exc:
                logging.warning(f"Failed to restore data_module checkpoint state: {exc}")

        train_loader = getattr(self, "train_loader", None)
        if train_loader is None:
            return

        loader_generator_state = checkpoint_state.get("loader_generator_state")
        loader_generator = getattr(train_loader, "generator", None)
        if loader_generator_state is not None and isinstance(loader_generator, torch.Generator):
            try:
                loader_generator.set_state(loader_generator_state)
            except Exception as exc:
                logging.warning(f"Failed to restore train loader generator state: {exc}")

        for key, obj in (
            ("dataset", getattr(train_loader, "dataset", None)),
            ("sampler", getattr(train_loader, "sampler", None)),
            ("batch_sampler", getattr(train_loader, "batch_sampler", None)),
        ):
            if obj is None:
                continue

            obj_state = checkpoint_state.get(key)
            if obj_state is not None and hasattr(obj, "load_state_dict"):
                try:
                    obj.load_state_dict(obj_state)
                except Exception as exc:
                    logging.warning(f"Failed to restore train loader {key} state: {exc}")

            obj_generator_state = checkpoint_state.get(f"{key}_generator_state")
            obj_generator = getattr(obj, "generator", None)
            if obj_generator_state is not None and isinstance(obj_generator, torch.Generator):
                try:
                    obj_generator.set_state(obj_generator_state)
                except Exception as exc:
                    logging.warning(f"Failed to restore train loader {key} generator state: {exc}")


    def _get_scalar_log_keys(self, phase):
        if phase in self._scalar_log_keys_cache:
            return self._scalar_log_keys_cache[phase]
        if self.logging_conf.scalar_keys_to_log is not None:
            keys = self.logging_conf.scalar_keys_to_log[phase].keys_to_log
            pruned = self._prune_scalar_log_keys(phase, keys)
            self._scalar_log_keys_cache[phase] = pruned
            return pruned
        self._scalar_log_keys_cache[phase] = []
        return []

    def _get_loss_conf_with_warmup(self):
        loss_conf = OmegaConf.create(OmegaConf.to_container(self.loss_conf, resolve=False))
        warmup_configs = getattr(self.optim_conf, "warmup_configs", None)
        if not warmup_configs:
            return loss_conf
        for conf in warmup_configs:
            attr = conf.get("attr")
            if not attr or not attr.startswith("loss."):
                continue
            rel_path = attr[len("loss.") :]
            try:
                OmegaConf.update(loss_conf, rel_path, conf.get("value"), merge=True)
            except Exception:
                continue
        return loss_conf

    def _prune_scalar_log_keys(self, phase, keys):
        if phase != "train":
            return keys

        loss_conf = self._get_loss_conf_with_warmup()
        camera_conf = loss_conf.get("camera")
        angle_conf = loss_conf.get("angle_pose")
        depth_conf = loss_conf.get("depth")
        point_conf = loss_conf.get("point")
        scale_conf = loss_conf.get("regulize_scale")

        camera_enabled = camera_conf is not None and camera_conf.get("weight", 1.0) != 0
        angle_enabled = angle_conf is not None and angle_conf.get("weight", 1.0) != 0
        depth_enabled = depth_conf is not None and depth_conf.get("weight", 1.0) != 0
        point_enabled = point_conf is not None and point_conf.get("weight", 1.0) != 0
        scale_enabled = scale_conf is not None and (
            scale_conf.get("enabled", False)
            or scale_conf.get("scale_weight", 0) != 0
        )

        kept = []
        for key in keys:
            if key == "loss_objective":
                kept.append(key)
                continue

            if key in {"loss_camera", "loss_T", "loss_R", "loss_FL"}:
                if not camera_enabled:
                    continue
                if key == "loss_T" and camera_conf.get("weight_trans", 0) == 0:
                    continue
                if key == "loss_R" and camera_conf.get("weight_rot", 0) == 0:
                    continue
                if key == "loss_FL" and camera_conf.get("weight_focal", 0) == 0:
                    continue
                kept.append(key)
                continue

            if key in {"loss_Rel_T", "loss_Rel_R", "loss_Pose_T", "loss_Pose_R"}:
                if not angle_enabled:
                    continue
                compute_rel = angle_conf.get("compute_relative", False)
                compute_abs = angle_conf.get("compute_absolute", False)
                wt = angle_conf.get("weight_trans", 0)
                wr = angle_conf.get("weight_rot", 0)
                if key in {"loss_Rel_T", "loss_Rel_R"} and not compute_rel:
                    continue
                if key in {"loss_Pose_T", "loss_Pose_R"} and not compute_abs:
                    continue
                if key in {"loss_Rel_T", "loss_Pose_T"} and wt == 0:
                    continue
                if key in {"loss_Rel_R", "loss_Pose_R"} and wr == 0:
                    continue
                kept.append(key)
                continue

            if key in {"loss_conf_depth", "loss_reg_depth", "loss_grad_depth", "loss_silog_depth", "loss_silog_conf_depth"}:
                if not depth_enabled:
                    continue
                if key == "loss_conf_depth" and depth_conf.get("conf_weight", 1.0) == 0:
                    continue
                if key == "loss_reg_depth" and depth_conf.get("reg_weight", 1.0) == 0:
                    continue
                if key == "loss_grad_depth":
                    if depth_conf.get("grad_weight", 1.0) == 0:
                        continue
                    grad_fn = depth_conf.get("gradient_loss_fn")
                    if not grad_fn:
                        continue
                if key == "loss_silog_depth" and depth_conf.get("silog_weight", 0.0) == 0:
                    continue
                if key == "loss_silog_conf_depth" and depth_conf.get("silog_conf_weight", 0.0) == 0:
                    continue
                kept.append(key)
                continue

            if "loss_conf_point" in key or "loss_reg_point" in key or "loss_grad_point" in key:
                if not point_enabled:
                    continue
                if "loss_conf_point" in key and point_conf.get("conf_weight", 1.0) == 0:
                    continue
                if "loss_reg_point" in key and point_conf.get("reg_weight", 1.0) == 0:
                    continue
                if "loss_grad_point" in key:
                    if point_conf.get("grad_weight", 1.0) == 0:
                        continue
                    grad_fn = point_conf.get("gradient_loss_fn")
                    if not grad_fn:
                        continue
                if "cam_from_depth" in key and not self.postprocess_conf.train.transform.get("cam_from_depth"):
                    continue
                if "global_from_cam" in key and not self.postprocess_conf.train.transform.get("global_from_cam"):
                    continue
                if "global_from_depth" in key and not self.postprocess_conf.train.transform.get("global_from_depth"):
                    continue
                if "pts3d_cam" in key and not self.model_conf.get("enable_cam_point"):
                    continue
                kept.append(key)
                continue

            if key in {"loss_scale", "loss_trans"}:
                if not scale_enabled:
                    continue
                kept.append(key)
                continue

            kept.append(key)

        return kept



    def _init_model_initializer(self):
        return instantiate(self.checkpoint_conf.model_weight_initializer)

    def _call_model_initializer(self, model_weight_initializer):
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.model = model_weight_initializer(model=self.model)

    def is_intermediate_val_epoch(self, epoch):
        return epoch % self.val_epoch_freq == 0 and epoch < self.max_epochs - 1


    def run(self):
        mode = self.mode
        assert mode in [
            "train",
            "val",
        ]
        if mode == "train":
            self.run_train()
            if self.cfg.get('test', False):
                self.run_val(val_loader=self.test_loader)
        elif mode == "val":
            self.run_val(val_loader=self.test_loader)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        print(f"log_dir: {self.logging_conf.log_dir}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _log_epoch_metrics_to_csv(self, phase, metrics):
        """Logs epoch metrics to a CSV file if enabled."""
        if not self.csv_logger:
            return

        data_dict = {'epoch': self.epoch, 'phase': phase}
        for k, v in metrics.items():
            if hasattr(v, "average"):
                data_dict[k] = v.average
            elif torch.is_tensor(v):
                data_dict[k] = float(v.item())
            else:
                data_dict[k] = v
        
        self.csv_logger.log(data_dict, val=(phase == 'val'))
    
    def end_warmup(self):
        if self.epoch == self.optim_conf.warmup_epochs:
            unfreeze(self.model, True)
        if not hasattr(self.optim_conf, "warmup_configs"):
            return
        for warmup_conf in self.optim_conf.warmup_configs:
            if self.epoch == warmup_conf.epoch:                
                parts = warmup_conf.attr.split('.')
                obj = self  # Start from the 'self' object

                try:
                    # Navigate down to the parent object
                    # e.g., for "loss.angle_pose.relative_weight", loop goes to 'loss', then 'angle_pose'
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    
                    # 'obj' is now the parent (e.g., self.loss.angle_pose)
                    # 'parts[-1]' is the final attribute name (e.g., 'relative_weight')
                    attr_name = parts[-1]
                    value = warmup_conf.value

                    # Set the attribute on the parent object
                    setattr(obj, attr_name, value)
                    
                    # Optional: log the change
                    print(f"Epoch {self.epoch}: Applied warmup config. Set {warmup_conf.attr} = {value}")

                except AttributeError as e:
                    print(f"Warning: Could not apply warmup config for {warmup_conf.attr}. Attribute not found. Error: {e}")

    def _extract_train_augs(self, data_conf):
        train_augs = (
            data_conf.data_module.get("train_config", {})
            .get("augs")
        )
        if train_augs is None:
            return None
        if OmegaConf.is_config(train_augs):
            return OmegaConf.to_container(train_augs, resolve=True)
        return copy.deepcopy(train_augs)

    def _iter_datasets_for_augs(self, dataset):
        if dataset is None:
            return
        if hasattr(dataset, "datasets"):
            for child in dataset.datasets:
                yield from self._iter_datasets_for_augs(child)
            return
        if hasattr(dataset, "dataset"):
            yield from self._iter_datasets_for_augs(dataset.dataset)
            return
        yield dataset

    def _ramp_value(self, spec, epoch):
        if not isinstance(spec, dict):
            return spec
        start = float(spec.get("start", 0.0))
        end = float(spec.get("end", start))
        start_epoch = int(spec.get("start_epoch", 0))
        end_epoch = int(spec.get("end_epoch", start_epoch))
        if end_epoch <= start_epoch:
            return end if epoch >= end_epoch else start
        if epoch <= start_epoch:
            return start
        if epoch >= end_epoch:
            return end
        t = (epoch - start_epoch) / float(end_epoch - start_epoch)
        return start + t * (end - start)

    def _apply_train_aug_schedule(self, epoch):
        if not self._base_train_augs or self.train_loader is None:
            return

        augs = copy.deepcopy(self._base_train_augs)

        if augs.get("random_crop_prob_schedule", None) is not None:
            augs["random_crop_prob"] = self._ramp_value(
                augs["random_crop_prob_schedule"], epoch
            )
        else:
            return

        # if "prot_schedule" in augs:
        #     augs["prot"] = self._ramp_value(augs["prot_schedule"], epoch)
        # else:
            # return

        # if "pcrop_schedule" in augs:
        #     augs["pcrop"] = self._ramp_value(augs["pcrop_schedule"], epoch)
        # else:
            # return

        if augs == self._last_train_augs:
            return

        for ds in self._iter_datasets_for_augs(self.train_loader.dataset):
            if hasattr(ds, "set_augs"):
                ds.set_augs(augs)
        self._last_train_augs = augs

    def run_train(self):
        last_train_epoch_duration_sec = 0.0
        last_val_epoch_duration_sec = 0.0
        limit_sec = None
        if self.total_run_time_hr is not None:
            try:
                limit_sec = float(self.total_run_time_hr) * 3600.0
            except (TypeError, ValueError):
                logging.warning(f"Ignoring invalid total_run_time_hr={self.total_run_time_hr!r}")
                limit_sec = None
        while self.epoch < self.max_epochs:
            self.end_warmup()
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, 0)
            self._apply_train_aug_schedule(self.epoch)

            train_epoch_start_time = time.time()
            ok = self.train_epoch(self.train_loader)
            last_train_epoch_duration_sec = time.time() - train_epoch_start_time
            if ok is False:
                logging.error("Stopping training due to non-finite loss.")
                break
            
            # Save checkpoint before validating
            self.save_checkpoint(self.epoch)
            ran_val = False
            if (self.epoch + 1) % self.optim_conf.val_freq == 0:
                val_epoch_start_time = time.time()
                self.run_val(val_loader=self.val_loader, epoch=self.epoch)
                last_val_epoch_duration_sec = time.time() - val_epoch_start_time
                ran_val = True

            # gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            self.epoch += 1

            if limit_sec is not None and limit_sec > 0:
                elapsed_sec = time.time() - self.start_time
                if elapsed_sec + last_train_epoch_duration_sec + last_val_epoch_duration_sec > limit_sec:
                    logging.info(
                        "Stopping before next epoch due to total_run_time_hr budget: "
                        f"elapsed={elapsed_sec/3600.0:.2f}h, "
                        f"limit={limit_sec/3600.0:.2f}h."
                    )
                    break
            if self.epoch == self.cfg.get("break_at", -1):
                break
        self.epoch -= 1

    @torch.no_grad()
    def _dump_model_stats_for_tests(self):
        # Done on all ranks because of FSDP and also for debugging DDP
        logging.info("Dumping stats of the trained model")
        stats = {
            "epoch": self.epoch,
            "rank": 0,
            "model": sum(p.sum() for p in self.model.parameters()).item(),
        }
        with g_pathmgr.open(
            os.path.join(
                self.logging_conf.log_dir,
                "unit_tests_model_stats.json",
            ),
            "a",
        ) as f:
            f.write(json.dumps(stats) + "\n")

    def run_val(self, val_loader=None, epoch=0, is_fresh_epoch=True):
        if not val_loader:
            return

        # The concept of a "fresh epoch" is not directly available with CombinedLoader
        outs = self.val_epoch(val_loader, is_fresh_epoch=is_fresh_epoch)
        outs_json = self._to_jsonable(outs)
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     torch.cuda.reset_peak_memory_stats()

        self.tb_writer.log_dict(outs_json, epoch)  # Logged only on rank 0

        # Log metrics to CSV
        self._log_epoch_metrics_to_csv(
            "val",
            {k: v for k, v in outs_json.items() if isinstance(v, (int, float))},
        )


        with g_pathmgr.open(
            os.path.join(self.logging_conf.log_dir, "val_stats.json"),
            "a",
        ) as f:
            f.write(json.dumps(outs_json) + "\n")

    def val_epoch(self, val_loader, is_fresh_epoch: bool):
        curr_phases = ['val']
        curr_models = [self.model]
        phase = curr_phases[0]

        for model in curr_models:
            model.eval()
            if hasattr(model, "on_validation_epoch_start"):
                model.on_validation_epoch_start()

        all_metrics = {}

        for dl_idx, current_val_loader in enumerate(val_loader):
            batch_time = AverageMeter("Batch Time", self.device, ":.4f")
            data_time = AverageMeter("Data Time", self.device, ":.4f")
            
            iters_per_epoch = len(current_val_loader)

            loss_names = ["objective"] + self._get_scalar_log_keys(phase)
            loss_names = [f"{phase}_{name}" for name in loss_names]
            
            loss_meters = {
                name: AverageMeter(name, self.device, ":.4f") for name in loss_names
            }

            progress = ProgressMeter(
                iters_per_epoch,
                [batch_time, data_time,
                self.time_elapsed_meter,
                *loss_meters.values(),],
                self._get_meters(curr_phases),
                prefix=f"Val Epoch: [{self.epoch}]",
            )

            end = time.time()

            limit_val_batches = (
                iters_per_epoch
                if self.limit_val_batches is None
                else self.limit_val_batches
            )
            dataset_name = None

            for data_iter, batch in enumerate(current_val_loader):
                # if data_iter > limit_val_batches:
                #     break

                data_time.update(time.time() - end)
                if dataset_name is None:
                    dataset_name = batch['dataset'][0][0]
                    progress.prefix = f"Val Epoch: [{self.epoch}] ({dataset_name})"

                #     with torch.amp.autocast(device_type='cuda', enabled=False):
                #         batch = self._process_batch(batch)
                batch = copy_data_to_device(batch, self.device)
                
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type='cuda',
                        enabled=self.optim_conf.amp.enabled,
                        dtype=self.amp_type,
                    ):
                        for phase, model in zip(curr_phases, curr_models):
                            self._val_step(
                                batch,
                                model,
                                phase,
                                loss_meters,
                            )

                batch_time.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

            # Collect metrics for this dataloader
            for k, v in loss_meters.items():
                metric_name = k.split('/')[-1].replace(f"{phase}_", "")
                all_metrics[f"{phase}_{dataset_name}_{metric_name}"] = v.avg
            
            self._reset_meters(curr_phases)

        for model in curr_models:
            if hasattr(model, "on_validation_epoch_end"):
                model.on_validation_epoch_end()

        for phase in curr_phases:
            all_metrics.update(self._get_trainer_state(phase))

        logging.info(f"Meters: {all_metrics}")
        return all_metrics

    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    @staticmethod
    def _to_jsonable(metrics: Mapping) -> Dict[str, Any]:
        out = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                out[key] = float(value.item())
            else:
                out[key] = value
        return out

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        peak_mem = AverageMeter("Peak Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                peak_mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = int(len(train_loader))
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        if self.accumulation_mode not in {"chunk_within_batch", "across_batches"}:
            raise ValueError(f"Unknown accumulation_mode: {self.accumulation_mode}")

        if self.accumulation_mode == "across_batches":
            for optim in self.optims:
                optim.zero_grad(set_to_none=True)

        for data_iter, batch in enumerate(train_loader):
            if data_iter >= limit_train_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            #     with torch.cuda.amp.autocast(enabled=False):
            #         batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            batch_size = batch["img"].shape[0]
            if batch_size <= 0:
                logging.warning("Skipping empty batch during training.")
                continue

            if self.accumulation_mode == "chunk_within_batch":
                accum_steps = self.accum_steps
                if accum_steps > batch_size:
                    logging.warning(
                        f"accum_steps ({accum_steps}) > batch_size ({batch_size}); "
                        f"clamping accum_steps to batch_size for this batch. image size: {batch['img'].shape}"
                    )
                    accum_steps = batch_size

                if accum_steps == 1:
                    chunked_batches = [batch]
                else:
                    chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

                ok = self._run_steps_on_batch_chunks(
                    chunked_batches, phase, loss_meters
                )
                if ok is False:
                    return False

                # compute gradient and do SGD step
                # assert data_iter <= limit_train_batches  # allow for off by one errors
                exact_epoch = self.epoch + float(data_iter) / limit_train_batches
                self.where = float(exact_epoch) / self.max_epochs
                
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    for optim in self.optims:
                        optim.step_schedulers(self.where)
                else:
                    logging.warning(
                        f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                    )
                        
                # Log schedulers
                if self.steps[phase] % self.logging_conf.log_freq == 0:
                    for i, optim in enumerate(self.optims):
                        for j, param_group in enumerate(optim.optimizer.param_groups):
                            for option in optim.schedulers[j]:
                                optim_prefix = (
                                    f"{i}_"
                                    if len(self.optims) > 1
                                    else (
                                        "" + f"{j}_"
                                        if len(optim.optimizer.param_groups) > 1
                                        else ""
                                    )
                                )
                                self.tb_writer.log(
                                    os.path.join("Optim", f"{optim_prefix}", option),
                                    param_group[option],
                                    self.steps[phase],
                                )
                    self.tb_writer.log(
                        os.path.join("Optim", "where"),
                        self.where,
                        self.steps[phase],
                    )
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                # Clipping gradients and detecting diverging gradients
                if self.gradient_clipper is not None:
                    for optim in self.optims:
                        self.scaler.unscale_(optim.optimizer)

                    grad_norm_dict = self.gradient_clipper(model=self.model)

                    for key, grad_norm in grad_norm_dict.items():
                        loss_meters[f"Grad/{key}"].update(grad_norm)

                # Optimizer step
                for optim in self.optims:
                    self.scaler.step(optim.optimizer)
                self.scaler.update()
            else:
                accum_steps = self.accum_steps
                should_step = ((data_iter + 1) % accum_steps == 0) or (data_iter + 1 == limit_train_batches)
                log_enabled = (not self.log_per_optimizer_step) or should_step
                with torch.amp.autocast(
                    device_type='cuda',
                    enabled=self.optim_conf.amp.enabled,
                    dtype=self.amp_type,
                ):
                    loss_dict = self._step(
                        batch,
                        self.model,
                        phase,
                        loss_meters,
                        log_enabled=log_enabled,
                        log_step=self.steps[phase],
                    )

                loss = loss_dict["objective"]
                loss_key = f"{phase}_loss_objective"
                loss_value = loss.detach()
                if not torch.isfinite(loss_value).all():
                    loss_value_item = loss_value.item()
                    error_msg = f"Loss is {loss_value_item}, attempting to stop training"
                    logging.error(error_msg)
                    return False

                loss = loss / accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss_value, batch_size)

                if should_step:
                    opt_iters = math.ceil(limit_train_batches / accum_steps)
                    step_idx = math.ceil((data_iter + 1) / accum_steps)
                    exact_epoch = self.epoch + float(step_idx) / opt_iters
                    self.where = float(exact_epoch) / self.max_epochs

                    assert self.where <= 1 + self.EPSILON
                    if self.where < 1.0:
                        for optim in self.optims:
                            optim.step_schedulers(self.where)
                    else:
                        logging.warning(
                            f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                        )

                    if self.steps[phase] % self.logging_conf.log_freq == 0:
                        for i, optim in enumerate(self.optims):
                            for j, param_group in enumerate(optim.optimizer.param_groups):
                                for option in optim.schedulers[j]:
                                    optim_prefix = (
                                        f"{i}_"
                                        if len(self.optims) > 1
                                        else (
                                            "" + f"{j}_"
                                            if len(optim.optimizer.param_groups) > 1
                                            else ""
                                        )
                                    )
                                    self.tb_writer.log(
                                        os.path.join("Optim", f"{optim_prefix}", option),
                                        param_group[option],
                                        self.steps[phase],
                                    )
                        self.tb_writer.log(
                            os.path.join("Optim", "where"),
                            self.where,
                            self.steps[phase],
                        )
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()

                    if self.gradient_clipper is not None:
                        for optim in self.optims:
                            self.scaler.unscale_(optim.optimizer)

                        grad_norm_dict = self.gradient_clipper(model=self.model)

                        for key, grad_norm in grad_norm_dict.items():
                            loss_meters[f"Grad/{key}"].update(grad_norm)

                    for optim in self.optims:
                        self.scaler.step(optim.optimizer)
                    self.scaler.update()
                    for optim in self.optims:
                        optim.zero_grad(set_to_none=True)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            if data_iter % self.logging_conf.log_freq == 0:
                peak_mem.update(torch.cuda.max_memory_allocated() / (1024 ** 3))
                progress.display(data_iter)

        # Log metrics to CSV
        self._log_epoch_metrics_to_csv("train", loss_meters)

        return True



    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        for i, chunked_batch in enumerate(chunked_batches):
            with torch.amp.autocast(
                device_type='cuda',
                enabled=self.optim_conf.amp.enabled, dtype=self.amp_type
            ):
                log_enabled = (not self.log_per_optimizer_step) or (i == accum_steps - 1)
                loss_dict = self._step(
                    chunked_batch,
                    self.model,
                    phase,
                    loss_meters,
                    log_enabled=log_enabled,
                    log_step=self.steps[phase],
                )


            loss = loss_dict["objective"]
            loss_key = f"{phase}_loss_objective"
            batch_size = chunked_batch["img"].shape[0]
            loss_value = loss.detach()

            if not torch.isfinite(loss_value).all():
                loss_value_item = loss_value.item()
                error_msg = f"Loss is {loss_value_item}, attempting to stop training"
                logging.error(error_msg)
                return False

            loss /= accum_steps
            self.scaler.scale(loss).backward()
            loss_meters[loss_key].update(loss_value, batch_size)

        return True



    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()



    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        tensor_keys = [
            "img", "depthmap", "camera_pose", "intrinsics", 
            "pts3d_cam", "pts3d", "valid_mask", 
        ]        
        string_keys = ["seq_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] + batch[key]
        
        return batch


    # def _process_batch(self, batch: Mapping):      
    #     # if self.data_conf.train.common_config.repeat_batch:
    #     #     batch = self._apply_batch_repetition(batch)
        
    #     # Normalize camera extrinsics and points. The function returns new tensors.
    #     normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
    #         normalize_camera_extrinsics_and_points_batch(
    #             extrinsics=batch["camera_pose"],
    #             cam_points=batch["pts3d_cam"],
    #             world_points=batch["pts3d"],
    #             depths=batch["depthmap"],
    #             point_masks=batch["valid_mask"],
    #         )

    #     # Replace the original values in the batch with the normalized ones.
    #     batch["camera_pose"] = normalized_extrinsics
    #     batch["pts3d_cam"] = normalized_cam_points
    #     batch["pts3d"] = normalized_world_points
    #     batch["depthmap"] = normalized_depths

        # return batch
    def _postprocess(self, batch, pred, pp_conf, pred_data_keys):
        data_keys = self.data_conf.data_keys 
        if pp_conf.get('to_extri'):
           # The image tensor `batch['img']` has a shape of (B, S, C, H, W).
           # We extract the last two dimensions (Height, Width) for the image size.
           image_size_hw = batch['img'].shape[-2:]
           pred[pred_data_keys.extrinsics], pred[pred_data_keys.intrinsics]= pose_encoding_to_extri_intri(pred["pose_enc"], image_size_hw=image_size_hw, build_intrinsics=True)

        if pp_conf.align.get('to_first_cam', {}).get('enabled'):
            with torch.no_grad():
                if pp_conf.align.to_first_cam.get("points"):
                    batch[data_keys.extrinsics], batch[data_keys.world_points] = align_camera_and_points_batch_ext(batch[data_keys.extrinsics], batch[data_keys.world_points])
                else:
                    batch[data_keys.extrinsics], _ = align_camera_and_points_batch_ext(batch[data_keys.extrinsics])
    
        if pp_conf.align.get('pr_align_cam', {}).get('enabled') and pred[pred_data_keys.extrinsics] is not None:
            pred['pose_aligned'] = True
            if pp_conf.align.pr_align_cam.points:
                pred[pred_data_keys.extrinsics], pred[pred_data_keys.world_points] = align_camera_and_points_batch_ext(pred[pred_data_keys.extrinsics], pred[pred_data_keys.world_points])                    
            else:
                pred[pred_data_keys.extrinsics], _ = align_camera_and_points_batch_ext(poses=pred[pred_data_keys.extrinsics])

        if pp_conf.transform.get('global_from_cam') and pred[pred_data_keys.extrinsics] is not None:
            # Detach pts3d_cam to prevent exploding gradients.
            # The gradient for the camera pose is proportional to the magnitude of the points.
            # Detaching breaks this unstable feedback loop.
            pred[pred_data_keys.global_from_cam] = global_points_from_cam(pred[pred_data_keys.extrinsics], batch[data_keys.pts3d_cam])
            # pred[pred_data_keys.global_from_cam] = global_points_from_cam(pred[pred_data_keys.extrinsics], pred[pred_data_keys.pts3d_cam])
            # pred[pred_data_keys.global_from_cam] = global_points_from_cam(pred[pred_data_keys.extrinsics], pred[pred_data_keys.pts3d_cam].detach())

        if pp_conf.transform.get('global_from_cam_detach_pose') and pred[pred_data_keys.extrinsics] is not None:
            pred[pred_data_keys.global_from_cam_detach_pose] = global_points_from_cam(batch[data_keys.extrinsics], pred[pred_data_keys.pts3d_cam])

        if (pp_conf.transform.get('cam_from_depth') or pp_conf.transform.get('global_from_depth')) and pred[pred_data_keys.intrinsics] is not None:
            pred[pred_data_keys.cam_from_depth] = cam_points_from_depth(pred[pred_data_keys.intrinsics], pred[pred_data_keys.depths].detach())

            if pp_conf.transform.global_from_depth  and pred[pred_data_keys.extrinsics] is not None:
                pred[pred_data_keys.global_from_depth] = global_points_from_cam(pred[pred_data_keys.extrinsics], pred[pred_data_keys.cam_from_depth].detach())

        if pp_conf.align.get('center_world', {}).get('enabled') and pred[pred_data_keys.extrinsics] is not None:
            center_world_key = pred_data_keys.get("global_to_center_world", "global_to_center_world")
            with torch.no_grad():
                mean_pose_in_old_world, old_world_to_mean_pose, _ = center_c2w_poses_batch(c2w_poses=batch[data_keys.extrinsics], return_poses=True)
                batch[data_keys.extrinsics], batch[data_keys.world_points] = align_camera_and_points_batch_ext(batch[data_keys.extrinsics], batch[data_keys.world_points], mean_pose_in_old_world)
                # pred[pred_data_keys.extrinsics], pred[center_world_key] = align_camera_and_points_batch_ext(pred[pred_data_keys.extrinsics], pred[pred_data_keys.world_points], mean_pose_in_old_world.detach())

        if pp_conf.normalize.get('gt_pts'):
            batch[data_keys.world_points], gt_avg_scale = normalize_pointcloud_vggt(batch[data_keys.world_points], batch[data_keys.valid_mask])
            # print(f"Normalize with scale {avg_scale}")

            batch[data_keys.depths], batch[data_keys.pts3d_cam], batch[data_keys.extrinsics], _ = normalize_depth_cam_extrinsics(gt_avg_scale, batch[data_keys.depths], cam_points=batch[data_keys.pts3d_cam], extrinsics=batch[data_keys.extrinsics])

        pts_align_conf = pp_conf.normalize.get("gt_pts_invariant", {})
        if pts_align_conf.get("enabled"):
            if pts_align_conf.get("translate"):
                batch[data_keys.world_points], batch[data_keys.extrinsics], centroid, inv_avg_scale = normalize_pointcloud_invariant(batch[data_keys.world_points], batch[data_keys.valid_mask], c2w_poses=batch[data_keys.extrinsics], return_pts=True)
                batch[data_keys.depths], batch[data_keys.pts3d_cam], _, _ = normalize_depth_cam_extrinsics(inv_scale=inv_avg_scale, depths=batch[data_keys.depths], cam_points=batch[data_keys.pts3d_cam])
            else:
                centroid, inv_avg_scale = normalize_pointcloud_invariant(batch[data_keys.world_points], batch[data_keys.valid_mask], return_pts=False)
                batch[data_keys.depths], batch[data_keys.pts3d_cam], batch[data_keys.extrinsics], batch[data_keys.world_points] = normalize_depth_cam_extrinsics(inv_scale=inv_avg_scale, depths=batch[data_keys.depths], cam_points=batch[data_keys.pts3d_cam], extrinsics=batch[data_keys.extrinsics], global_points3d=batch[data_keys.world_points])
                                    
        if pp_conf.normalize.get('gt_depth'):
            gt_avg_scale = calculate_depth_scale(batch[data_keys.depths], batch[data_keys.valid_mask], eps=1e-3, mode='mean')           

            batch[data_keys.depths], batch[data_keys.pts3d_cam], batch[data_keys.extrinsics], batch[data_keys.world_points] = normalize_depth_cam_extrinsics(norm_factor=gt_avg_scale, depths=batch[data_keys.depths], cam_points=batch[data_keys.pts3d_cam], extrinsics=batch[data_keys.extrinsics], global_points3d=batch[data_keys.world_points])

        if pp_conf.normalize.pr_pts.get('enabled'):
            if pp_conf.normalize.pr_pts.metric:
                pred[pred_data_keys.world_points], norm_factor_pr = normalize_pr_pointcloud(pred[pred_data_keys.world_points], gt_avg_scale, batch[data_keys.valid_mask], not_metric_mask=None)
            else:
                pred[pred_data_keys.world_points], norm_factor_pr = normalize_pointcloud_vggt(pred[pred_data_keys.world_points], batch[data_keys.valid_mask])
                pred['scale'] = 1.0 / norm_factor_pr

            pred["depth"], _, pred[pred_data_keys.extrinsics], _ = normalize_depth_cam_extrinsics(norm_factor_pr, pred["depth"], None, pred[pred_data_keys.extrinsics])
            pred["pose_trans_aligned"] = True

        pts_align_conf = pp_conf.normalize.get("pr_pts_invariant", {})
        if pts_align_conf.get("enabled"):
            if pts_align_conf.get("translate"):
                pred[pred_data_keys.world_points], pred[pred_data_keys.extrinsics], centroid, inv_avg_scale = normalize_pointcloud_invariant(pred[pred_data_keys.world_points], batch[data_keys.valid_mask], c2w_poses=pred[pred_data_keys.extrinsics], return_pts=True)
            else:
                centroid, inv_avg_scale = normalize_pointcloud_invariant(pred[pred_data_keys.world_points], batch[data_keys.valid_mask], return_pts=False)
                _, _, pred[pred_data_keys.extrinsics], pred[pred_data_keys.world_points] = normalize_depth_cam_extrinsics(inv_scale=inv_avg_scale, extrinsics=pred[pred_data_keys.extrinsics], global_points3d=pred[pred_data_keys.world_points])
            pred["pose_trans_aligned"] = True
            pred['scale'] = inv_avg_scale
            pred['translation'] = centroid
        
        pts_align_conf = pp_conf.normalize.get("pred_center", {})
        if pts_align_conf.get('enabled') and pred[pred_data_keys.extrinsics] is not None:
            if not pts_align_conf.get('pr_to_gt'):
                with torch.no_grad():                
                    gt_to_pred_transform = get_pred_world_to_gt_world_transforms(pred[pred_data_keys.extrinsics], batch[data_keys.extrinsics])
                    mean_pose_in_old_world, old_world_to_mean_pose, _ = center_c2w_poses_batch(c2w_poses=gt_to_pred_transform, return_poses=False)
                    batch[data_keys.extrinsics], batch[data_keys.world_points] = align_camera_and_points_batch_ext(batch[data_keys.extrinsics], batch[data_keys.world_points], mean_pose_in_old_world)
            else:
                aligned_to_center_key = pred_data_keys.get("global_aligned_to_center", "global_aligned_to_center")
                # with torch.no_grad():
                pred_to_gt_transform = get_pred_world_to_gt_world_transforms(batch[data_keys.extrinsics], pred[pred_data_keys.extrinsics])
                mean_pose_in_old_world, old_world_to_mean_pose, _ = center_c2w_poses_batch(c2w_poses=pred_to_gt_transform, return_poses=False)
                pred[pred_data_keys.extrinsics], pred[aligned_to_center_key] = align_camera_and_points_batch_ext(pred[pred_data_keys.extrinsics], pred[pred_data_keys.world_points], mean_pose_in_old_world)
                pred['pose_aligned'] = True

        pts_align_conf = pp_conf.align.get("gt_align_to_pts", {})
        if pts_align_conf.get("enabled"):
            if pred_data_keys.world_points in pred:
                with torch.no_grad():                
                    _, transform_params = align_pred_to_gt_torch_batch_roma(batch[data_keys.world_points], pred[pred_data_keys.world_points],  batch[data_keys.valid_mask], pred["world_points_conf"], conf_percentage=pts_align_conf.conf_percentage, with_scale=False, return_points=False)
                     
                    if pts_align_conf.align_pose:
                        batch[data_keys.extrinsics], batch[data_keys.world_points] = align_c2w_poses_points_torch(c2w_poses=batch[data_keys.extrinsics], transform_params=transform_params, points3D=batch[data_keys.world_points], with_scale=False)
                    else:
                        _, batch[data_keys.world_points] = align_c2w_poses_points_torch(transform_params=transform_params, points3D=batch[data_keys.world_points], with_scale=False)

        pts_align_conf = pp_conf.align.get("pts_align_to_gt", {})
        if pts_align_conf.get("enabled"):
            # aligned_global_from_cam_key = pred_data_keys.get("aligned_global_from_cam", "aligned_global_from_cam")
            # if pred_data_keys.global_from_cam in pred:
            #     pred[aligned_global_from_cam_key], transform_params = align_pred_to_gt_torch_batch(pred[pred_data_keys.global_from_cam], batch[data_keys.world_points], batch[data_keys.valid_mask], pred["cam_points_conf"], conf_percentage=pp_conf.align.pts_align_to_gt.conf_percentage, with_scale=pp_conf.align.pts_align_to_gt.with_scale)

            # aligned_global_from_depth_key = pred_data_keys.get("aligned_global_from_depth", "aligned_global_from_depth")
            # if pred_data_keys.global_from_depth in pred:
            #     pred[aligned_global_from_depth_key], transform_params = align_pred_to_gt_torch_batch(pred[pred_data_keys.global_from_depth], batch[data_keys.world_points], batch[data_keys.valid_mask], pred["depth_conf"], conf_percentage=pp_conf.align.pts_align_to_gt.conf_percentage, with_scale=pp_conf.align.pts_align_to_gt.with_scale)

            aligned_world_points_key = pred_data_keys.get("aligned_world_points", "aligned_world_points")
            if pred_data_keys.world_points in pred:                
                with_scale = pts_align_conf.with_scale
                _, transform_params = align_pred_to_gt_torch_batch_roma(pred[pred_data_keys.world_points], batch[data_keys.world_points], batch[data_keys.valid_mask], pred["world_points_conf"], conf_percentage=pts_align_conf.conf_percentage, with_scale=with_scale, return_points=False)
                # 
                if 'translation' not in pred:
                    if pts_align_conf.with_scale:
                        pred['scale'] = transform_params['scale']
                    pred['translation'] = transform_params['translation']
                if pts_align_conf.with_scale:
                    pred["pose_trans_aligned"] = True
                # params_detached = {k: v.detach() for k, v in transform_params.items() if v is not None}
                params_detached = transform_params
                if pts_align_conf.align_pose:
                    pred['pose_aligned'] = True
                    pred[pred_data_keys.extrinsics], pred[aligned_world_points_key] = align_c2w_poses_points_torch(c2w_poses=pred[pred_data_keys.extrinsics], transform_params=params_detached, points3D=pred[pred_data_keys.world_points], with_scale=with_scale)
                else:
                    _, pred[aligned_world_points_key] = align_c2w_poses_points_torch(transform_params=params_detached, points3D=pred[pred_data_keys.world_points], with_scale=with_scale)

                if pts_align_conf.get("normalize_pose"):
                    _, _, pred[pred_data_keys.extrinsics], _ = normalize_depth_cam_extrinsics(extrinsics=pred[pred_data_keys.extrinsics],  norm_factor=(1/transform_params['scale']))
                if pts_align_conf.get("normalize_depth"):
                    pred["depth"], _, _, _ = normalize_depth_cam_extrinsics(norm_factor=(1/transform_params['scale']), depths=pred["depth"])

        pts_align_conf = pp_conf.align.get("pts_align_to_gt_rot", {})
        if pts_align_conf.get("enabled"):
            aligned_world_points_key = pred_data_keys.get("aligned_world_points", "aligned_world_points")
            if pred_data_keys.world_points in pred:                
                R_opt = align_rotation_only_torch(pred[pred_data_keys.world_points], batch[data_keys.world_points], batch[data_keys.valid_mask], )#pred["world_points_conf"], conf_percentage=pts_align_conf.conf_percentage
                if pts_align_conf.align_pose:
                    pred['pose_aligned'] = True
                    pred[pred_data_keys.extrinsics], pred[aligned_world_points_key] = align_c2w_poses_points_rotation_only(R_opt, c2w_poses=pred[pred_data_keys.extrinsics], points3D=pred[pred_data_keys.world_points])
                else:
                    _, pred[aligned_world_points_key] = align_c2w_poses_points_rotation_only(R_opt, points3D=pred[pred_data_keys.world_points])

        if pp_conf.align.get('depth_align_to_gt', {}).get('enabled'):
            pred["depth"], batch_median_pred, batch_median_gt = median_scale_depth_torch_batch(pred["depth"], batch[data_keys.depths], batch[data_keys.valid_mask], pred["depth_conf"], conf_percentage=pp_conf.align.depth_align_to_gt.conf_percentage)
            if pp_conf.align.depth_align_to_gt.pose:
                _, _, pred[pred_data_keys.extrinsics], _ = normalize_depth_cam_extrinsics(extrinsics=pred[pred_data_keys.extrinsics],  norm_factor=(batch_median_pred/batch_median_gt))


    def _step(
        self,
        batch,
        model: nn.Module,
        phase: str,
        loss_meters: dict[str, AverageMeter],
        log_enabled: bool = True,
        log_step: int | None = None,
    ):
        # Forward run of the model
        y_hat = model(images = batch["img"])

        self._postprocess(batch, y_hat, self.postprocess_conf.train, self.postprocess_conf.data_keys)

        # Compute the loss
        loss_dict = self.loss(y_hat, batch, data_keys=self.data_conf.data_keys, pred_data_keys=self.postprocess_conf.data_keys)
        
        # concatenate y_hat, loss_dict and batch for visualizations
        y_hat_batch = {**y_hat, **loss_dict, **batch}

        if log_enabled:
            step = self.steps[phase] if log_step is None else log_step
            self._update_and_log_scalars(y_hat_batch, phase, step, loss_meters)
            self._log_tb_visuals(y_hat_batch, phase, step)
            self.steps[phase] += 1

        return loss_dict


    def _val_step(
        self,
        batch,
        model: nn.Module,
        phase: str,
        loss_meters: dict[str, AverageMeter],
    ):
        # Forward run of the model
        y_hat = model(images = batch["img"])

        self._postprocess(batch, y_hat, self.postprocess_conf.val, self.postprocess_conf.data_keys)

        y_hat_batch = eval_batch(y_hat=y_hat, batch=batch, metrics_conf=self.postprocess_conf.val.metrics, data_keys=self.data_conf.data_keys, pred_data_keys=self.postprocess_conf.data_keys)

        # Compute the loss
        # loss_dict = self.loss(y_hat, batch)
        
        # concatenate y_hat, loss_dict and batch for visualizations
        # y_hat_batch = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(y_hat_batch, phase, self.steps[phase], loss_meters, batch_size=batch["img"].shape[0])
        self._log_tb_visuals(y_hat_batch, phase, self.steps[phase])
        self.steps[phase] += 1


    def _update_and_log_scalars(
        self,
        batch: Mapping,
        phase: str,
        step: int,        
        loss_meters: dict[str, AverageMeter],
        batch_size: int = 1,
    ) -> None:
        keys_to_log = self._get_scalar_log_keys(phase)
        if "camera_pose" in batch:
            batch_size = batch["camera_pose"].shape[0]
        for key in keys_to_log:
            if key in batch:
                if torch.is_tensor(batch[key]):
                    value = batch[key].detach()
                else:
                    value = batch[key]
                loss_meters[f"{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0:
                    value_item = value.item() if torch.is_tensor(value) else value
                    self.tb_writer.log(f"Values/{phase}/{key}", value_item, step)


    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )



def chunk_batch_for_accum_steps(batch, accum_steps: int):
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]


def is_sequence_of_primitives(data):
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )


def get_chunk_from_data(data, chunk_id, num_chunks):
    """
    Recursively splits all the tensors inside the passed data object into num_chunks.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
