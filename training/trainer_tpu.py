import copy
import gc
import json
import logging
import os
import random
import resource
import time
from contextlib import nullcontext
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import DictConfig, OmegaConf

from training.trainer import Trainer, get_amp_type
from train_utils.csv_writer import CsvLogger
from train_utils.freeze import freeze_modules
from train_utils.general import (
    AverageMeter,
    DurationMeter,
    ProgressMeter,
    copy_data_to_device,
    safe_makedirs,
    set_seeds,
)
from train_utils.logging import setup_logging
from train_utils.optimizer import construct_optimizers

_TPU_DATA_MODULE_TARGET = (
    "training.datasets.base.standalone_multiview_datamodule_tpu.StandaloneMultiViewDataModuleTPU"
)
_PREFORK_MODEL = None
_PREFORK_TRAIN_DATASET = None


class _NullScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        del optimizer

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict) -> None:
        del state_dict


class _TPUTrainMeter:
    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        del val
        del n
        return

    def update_host(self, val, n=1):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        value = float(val)
        self.val = value
        self.sum += value * float(n)
        self.count += float(n)
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(name=self.name, val=float(self.val), avg=float(self.avg))


class TrainerTPU(Trainer):
    def _apply_tpu_config_overrides(self, cfg: DictConfig) -> DictConfig:
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)
        if OmegaConf.select(cfg, "data.data_module") is not None:
            OmegaConf.update(cfg, "data.data_module._target_", _TPU_DATA_MODULE_TARGET, merge=False, force_add=True)
        OmegaConf.update(cfg, "accum_steps", 1, merge=False, force_add=True)
        if OmegaConf.select(cfg, "optim.amp") is not None:
            OmegaConf.update(cfg, "optim.amp.enabled", True, merge=False, force_add=True)
            OmegaConf.update(cfg, "optim.amp.amp_dtype", "bfloat16", merge=False, force_add=True)
        return cfg

    def _merge_resume_config(self, cfg: DictConfig) -> DictConfig:
        merged = super()._merge_resume_config(cfg)
        return self._apply_tpu_config_overrides(merged)

    def __init__(self, cfg: DictConfig):
        self._scalar_log_keys_cache = {}
        self._setup_timers()
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)

        self._resume_ckpt_path = None
        self._resume_checkpoint = None
        self._resume_checkpoint_amp = None
        self._trainer_config_snapshot = None
        self.metrics_history = {"train": [], "val": []}
        self.data_module = None
        global _PREFORK_MODEL, _PREFORK_TRAIN_DATASET
        self._prefork_model = _PREFORK_MODEL
        self._prefork_train_dataset = _PREFORK_TRAIN_DATASET
        _PREFORK_MODEL = None
        _PREFORK_TRAIN_DATASET = None

        cfg = self._merge_resume_config(cfg)
        cfg = self._resolve_conf_logit_max(cfg)
        self.accum_steps = 1
        self.cfg = cfg

        self.env_variables = cfg.get("env_variables")
        self._apply_env_variables(self.env_variables)
        self._import_xla()
        self._setup_tpu_runtime()
        self._log_tpu_runtime_environment()

        self.data_conf = cfg.data
        self.model_conf = cfg.model
        self.loss_conf = cfg.loss
        self.logging_conf = cfg.logging
        self.checkpoint_conf = cfg.checkpoint
        self.postprocess_conf = cfg.postprocess

        self.log_per_optimizer_step = cfg.optim.get("log_per_optimizer_step", False)
        self.max_epochs = cfg.max_epochs
        self.mode = cfg.get("mode", "train")
        self.val_epoch_freq = cfg.get("val_epoch_freq", 1)
        self.limit_train_batches = cfg.get("limit_train_batches")
        self.limit_val_batches = cfg.get("limit_val_batches")
        self.optim_conf = cfg.optim
        self.compile_conf = cfg.get("compile")
        if self.compile_conf and self.compile_conf.get("enabled"):
            self.compile_conf.enabled = False
        self.device_conf = "xla"
        self.cuda_conf = cfg.get("cuda")
        self.where = 0.0
        self.seed_value = cfg.get("seed_value", 123)
        self.total_run_time_hr = cfg.get("total_run_time_hr")
        self.resume_bs = cfg.get("resume_bs", False)
        train_augs = self.data_conf.data_module.get("train_config", {}).get("augs", {})
        random_crop_prob_schedule = train_augs.get("random_crop_prob_schedule")
        self._base_train_augs = None
        if random_crop_prob_schedule is not None:
            self._base_train_augs = self._extract_train_augs(self.data_conf)
            self._last_train_augs = copy.deepcopy(self._base_train_augs)
        self.memory_debug_rank = int(os.environ.get("TPU_MEMORY_DEBUG_RANK", cfg.get("tpu_memory_debug_rank", 0)))
        self.memory_debug_interval = int(
            os.environ.get(
                "TPU_MEMORY_DEBUG_INTERVAL",
                cfg.get("tpu_memory_debug_interval", 0),
            )
        )
        self.check_loss_finite = str(
            os.environ.get("TPU_CHECK_LOSS_FINITE", cfg.get("tpu_check_loss_finite", False))
        ).lower() in {"1", "true", "yes", "on"}
        self.mp_loader_prefetch_size = int(
            os.environ.get("TPU_MP_LOADER_PREFETCH_SIZE", cfg.get("tpu_mp_loader_prefetch_size", 1))
        )
        self.mp_device_prefetch_size = int(
            os.environ.get("TPU_MP_DEVICE_PREFETCH_SIZE", cfg.get("tpu_mp_device_prefetch_size", 1))
        )
        self.debug_cpu_first_batch = str(
            os.environ.get("TPU_DEBUG_CPU_FIRST_BATCH", cfg.get("tpu_debug_cpu_first_batch", True))
        ).lower() in {"1", "true", "yes", "on"}
        self.debug_first_train_step = str(
            os.environ.get("TPU_DEBUG_FIRST_TRAIN_STEP", cfg.get("tpu_debug_first_train_step", True))
        ).lower() in {"1", "true", "yes", "on"}
        self.debug_skip_first_optimizer_step = str(
            os.environ.get(
                "TPU_DEBUG_SKIP_FIRST_OPTIMIZER_STEP",
                cfg.get("tpu_debug_skip_first_optimizer_step", False),
            )
        ).lower() in {"1", "true", "yes", "on"}
        self.xla_metrics_debug = str(
            os.environ.get("TPU_XLA_METRICS_DEBUG", cfg.get("tpu_xla_metrics_debug", False))
        ).lower() in {"1", "true", "yes", "on"}
        self._debug_step_iter: Optional[int] = None
        self._debug_step_iter_limit: Optional[int] = None

        log_dir = self.logging_conf.log_dir
        self.checkpoint_conf.save_dir = os.path.join(log_dir, "ckpts")
        if self.is_master:
            safe_makedirs(self.logging_conf.log_dir)
            self._write_run_metadata()

        self._setup_device("xla")
        self._setup_cuda_backend(self.cuda_conf)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(self.seed_value, self.max_epochs, self.rank)

        amp_conf = getattr(self.optim_conf, "amp", None)
        if amp_conf is not None and bool(amp_conf.enabled):
            self.amp_type = get_amp_type(amp_conf.amp_dtype)
        else:
            self.amp_type = torch.float32
        self.autocast_device_type = "xla"

        self._log_memory_snapshot("before_runtime_objects")
        self._setup_runtime_objects()
        self._log_memory_snapshot("after_runtime_objects")

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        if self.mode != "val":
            self.optims = construct_optimizers(
                self.model,
                self.optim_conf,
            )
        else:
            self.optims = []

        self.csv_logger = None
        if self.logging_conf.get("csv_writer") and self.logging_conf.csv_writer.get("enabled") and self.is_master:
            csv_conf = self.logging_conf.csv_writer
            csv_path = os.path.join(csv_conf.path, csv_conf.filename)
            self.csv_logger = CsvLogger(csv_path)

        if self._resume_ckpt_path is not None:
            if self._resume_checkpoint is None:
                self._resume_checkpoint = self._load_checkpoint_file(self._resume_ckpt_path)
            if self._resume_checkpoint_amp is None and self._resume_checkpoint:
                resume_cfg = self._resume_checkpoint.get("trainer_config")
                if resume_cfg:
                    self._resume_checkpoint_amp = resume_cfg.get("optim", {}).get("amp", None)
            self._load_resuming_checkpoint(self._resume_ckpt_path, checkpoint=self._resume_checkpoint)

        if self.mode != "val" and self.is_master:
            conf_to_save = OmegaConf.create(self.cfg)
            self._trainer_config_snapshot = conf_to_save
            config_path = os.path.join(self.logging_conf.log_dir, "trainer_config.yaml")
            with g_pathmgr.open(config_path, "w") as file_obj:
                file_obj.write(OmegaConf.to_yaml(conf_to_save))

    def _import_xla(self) -> None:
        import torch_xla as txla
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.runtime as xr

        self.txla = txla
        self.xm = xm
        self.xla_metrics = met
        self.pl = pl
        self.xr = xr

    def _setup_tpu_runtime(self) -> None:
        self.world_size = int(self.xr.world_size()) if hasattr(self.xr, "world_size") else 1
        self.rank = int(self.xr.global_ordinal()) if hasattr(self.xr, "global_ordinal") else 0
        self.local_rank = int(self.xr.local_ordinal()) if hasattr(self.xr, "local_ordinal") else self.rank
        self.is_master = bool(self.xm.is_master_ordinal())

    def _master_print(self, msg: str) -> None:
        if self.is_master:
            self.xm.master_print(msg, flush=True)

    @staticmethod
    def _host_memory_snapshot_mb() -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    if key not in {"VmRSS", "VmHWM", "VmSize", "VmSwap"}:
                        continue
                    parts = value.strip().split()
                    if not parts:
                        continue
                    out[key] = float(parts[0]) / 1024.0
        except Exception:
            pass
        try:
            ru_maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if ru_maxrss_kb:
                out.setdefault("ru_maxrss", float(ru_maxrss_kb) / 1024.0)
        except Exception:
            pass
        return out

    @staticmethod
    def _system_memory_snapshot_mb() -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    if key not in {"MemAvailable", "MemFree", "SwapFree"}:
                        continue
                    parts = value.strip().split()
                    if not parts:
                        continue
                    out[key] = float(parts[0]) / 1024.0
        except Exception:
            pass

        def _read_cgroup_value(path: str) -> Optional[float]:
            try:
                with open(path, "r", encoding="utf-8") as file_obj:
                    raw = file_obj.read().strip()
            except Exception:
                return None
            if not raw or raw == "max":
                return None
            try:
                return float(raw) / (1024.0 ** 2)
            except Exception:
                return None

        cgroup_current = _read_cgroup_value("/sys/fs/cgroup/memory.current")
        if cgroup_current is not None:
            out["cgroup_current"] = cgroup_current
        cgroup_max = _read_cgroup_value("/sys/fs/cgroup/memory.max")
        if cgroup_max is not None:
            out["cgroup_max"] = cgroup_max
        return out

    def _xla_memory_snapshot_mb(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        device = getattr(self, "device", None)
        if device is None:
            return out
        get_mem_info = getattr(self.xm, "get_memory_info", None)
        if get_mem_info is None:
            return out
        try:
            mem_info = get_mem_info(device)
        except Exception:
            return out
        if not isinstance(mem_info, Mapping):
            return out
        for key, value in mem_info.items():
            try:
                out[str(key)] = float(value) / (1024.0 ** 2)
            except Exception:
                continue
        return out

    @staticmethod
    def _describe_dataset_state(dataset) -> str:
        if dataset is None:
            return "dataset=None"
        pieces = [f"type={type(dataset).__name__}"]
        try:
            pieces.append(f"len={len(dataset)}")
        except Exception:
            pass
        for attr in ("images", "scenes", "scene_img_list", "datasets"):
            if hasattr(dataset, attr):
                try:
                    pieces.append(f"{attr}={len(getattr(dataset, attr))}")
                except Exception:
                    pieces.append(f"{attr}=<?>")
        return " ".join(pieces)

    def _should_log_memory_snapshot(self) -> bool:
        debug_rank = int(self.memory_debug_rank)
        return debug_rank < 0 or int(getattr(self, "rank", 0)) == debug_rank

    def _should_log_train_iter_detail(self, data_iter: int, limit_train_batches: int) -> bool:
        if data_iter == 0 and self.debug_first_train_step:
            return True
        if self.memory_debug_interval <= 0:
            return False
        return (
            data_iter == 0
            or data_iter % self.memory_debug_interval == 0
            or data_iter + 1 == limit_train_batches
        )

    def _should_display_progress(self, data_iter: int, limit_batches: int) -> bool:
        log_freq = int(self.logging_conf.log_freq)
        if log_freq <= 0:
            return False
        return (data_iter + 1) % log_freq == 0 or data_iter + 1 == limit_batches

    def _should_update_train_host_metrics(self, data_iter: int, limit_batches: int) -> bool:
        return self.is_master and self._should_display_progress(data_iter, limit_batches)

    @staticmethod
    def _describe_batch_state(batch: Any) -> str:
        if not isinstance(batch, Mapping):
            return f"batch_type={type(batch).__name__}"
        parts = []
        images = batch.get("img")
        if hasattr(images, "shape"):
            try:
                parts.append(f"img_shape={tuple(images.shape)}")
            except Exception:
                pass
        dataset_name = batch.get("dataset")
        if isinstance(dataset_name, list) and dataset_name:
            try:
                first_dataset = dataset_name[0]
                if isinstance(first_dataset, list) and first_dataset:
                    first_dataset = first_dataset[0]
                parts.append(f"dataset={first_dataset}")
            except Exception:
                pass
        return " ".join(parts)

    @staticmethod
    def _summarize_mapping_keys(data: Any, limit: int = 8) -> str:
        if not isinstance(data, Mapping):
            return f"type={type(data).__name__}"
        keys = list(data.keys())
        shown = ",".join(str(key) for key in keys[:limit])
        if len(keys) > limit:
            shown += ",..."
        return f"keys=[{shown}]"

    def _log_memory_snapshot(self, stage: str, *, extra: Optional[str] = None) -> None:
        if not self._should_log_memory_snapshot():
            return
        host_mem = self._host_memory_snapshot_mb()
        system_mem = self._system_memory_snapshot_mb()
        xla_mem = self._xla_memory_snapshot_mb()
        parts = [
            f"[trainer_tpu] mem stage={stage}",
            f"rank={getattr(self, 'rank', '?')}",
            f"pid={os.getpid()}",
        ]
        if extra:
            parts.append(extra)
        for key in ("VmRSS", "VmHWM", "VmSize", "VmSwap", "ru_maxrss"):
            if key in host_mem:
                parts.append(f"{key}={host_mem[key]:.1f}MB")
        for key in ("MemAvailable", "MemFree", "SwapFree", "cgroup_current", "cgroup_max"):
            if key in system_mem:
                parts.append(f"{key}={system_mem[key]:.1f}MB")
        for key, value in sorted(xla_mem.items()):
            parts.append(f"xla_{key}={value:.1f}MB")
        print(" ".join(parts), flush=True)

    def _log_xla_metrics_snapshot(self, stage: str, *, extra: Optional[str] = None) -> None:
        if not self.xla_metrics_debug or not self._should_log_memory_snapshot():
            return
        metrics = getattr(self, "xla_metrics", None)
        if metrics is None:
            return
        try:
            report = metrics.short_metrics_report()
        except Exception as exc:
            print(
                f"[trainer_tpu] xla_metrics stage={stage} rank={getattr(self, 'rank', '?')} "
                f"pid={os.getpid()} error={type(exc).__name__}:{exc}",
                flush=True,
            )
            return

        interesting = (
            "Compile",
            "Execute",
            "Transfer",
            "Sync",
            "MarkStep",
            "CreateCompileHandles",
            "Uncached",
            "Cached",
            "aten::",
            "xla::",
        )
        lines = []
        for raw_line in str(report).splitlines():
            line = raw_line.strip()
            if line and any(token in line for token in interesting):
                lines.append(line)
        if not lines:
            lines = [line.strip() for line in str(report).splitlines() if line.strip()][:20]

        header = [
            f"[trainer_tpu] xla_metrics stage={stage}",
            f"rank={getattr(self, 'rank', '?')}",
            f"pid={os.getpid()}",
        ]
        if extra:
            header.append(extra)
        print(" ".join(header), flush=True)
        for line in lines[:80]:
            print(f"[trainer_tpu] xla_metrics {line}", flush=True)

    def _debug_step_boundary(
        self,
        stage: str,
        *,
        phase: str,
        batch=None,
        y_hat=None,
        loss_dict=None,
    ) -> None:
        if phase != "train":
            return
        data_iter = self._debug_step_iter
        limit_train_batches = self._debug_step_iter_limit
        if data_iter is None or limit_train_batches is None:
            return
        if not self._should_log_train_iter_detail(data_iter, limit_train_batches):
            return
        extras = [f"epoch={self.epoch}", f"iter={data_iter}/{limit_train_batches}"]
        if stage == "before_model":
            extras.append(self._describe_batch_state(batch))
        elif stage in {"after_model", "after_postprocess", "after_eval"} and y_hat is not None:
            extras.append(self._summarize_mapping_keys(y_hat))
        elif stage == "after_loss" and loss_dict is not None:
            extras.append(self._summarize_mapping_keys(loss_dict))
        extra = " ".join(extras)
        self._log_memory_snapshot(f"train_step_{stage}", extra=extra)
        self._log_xla_metrics_snapshot(f"train_step_{stage}", extra=extra)

    def _step(
        self,
        batch,
        model,
        phase: str,
        loss_meters: Dict[str, AverageMeter],
        log_enabled: bool = True,
        log_step: Optional[int] = None,
    ):
        self._debug_step_boundary("before_model", phase=phase, batch=batch)
        y_hat = model(images=batch["img"])
        self._debug_step_boundary("after_model", phase=phase, batch=batch, y_hat=y_hat)

        self._postprocess(batch, y_hat, self.postprocess_conf.train, self.postprocess_conf.data_keys)
        self._debug_step_boundary("after_postprocess", phase=phase, batch=batch, y_hat=y_hat)

        loss_dict = self.loss(
            y_hat,
            batch,
            data_keys=self.data_conf.data_keys,
            pred_data_keys=self.postprocess_conf.data_keys,
        )
        self._debug_step_boundary("after_loss", phase=phase, batch=batch, y_hat=y_hat, loss_dict=loss_dict)

        if log_enabled:
            step = self.steps[phase] if log_step is None else log_step
            scalar_batch = {
                key: loss_dict[key]
                for key in self._get_scalar_log_keys(phase)
                if key in loss_dict
            }
            self._update_and_log_scalars(scalar_batch, phase, step, loss_meters, batch_size=batch["img"].shape[0])
            if self.logging_conf.log_visuals:
                self._log_tb_visuals({**y_hat, **loss_dict, **batch}, phase, step)
            self.steps[phase] += 1
            self._debug_step_boundary("after_log", phase=phase, batch=batch, y_hat=y_hat, loss_dict=loss_dict)

        return loss_dict

    def _update_and_log_scalars(
        self,
        batch: Mapping,
        phase: str,
        step: int,
        loss_meters: Dict[str, AverageMeter],
        batch_size: int = 1,
    ) -> None:
        keys_to_log = self._get_scalar_log_keys(phase)
        if "camera_pose" in batch:
            batch_size = batch["camera_pose"].shape[0]
        should_log_tb = (
            self.tb_writer is not None
            and self.logging_conf.log_freq > 0
            and (step + 1) % self.logging_conf.log_freq == 0
        )
        should_update_host_meter = (
            phase != "train"
            or (
                self.is_master
                and self.logging_conf.log_freq > 0
                and (step + 1) % self.logging_conf.log_freq == 0
            )
        )
        for key in keys_to_log:
            if key not in batch:
                continue
            value = batch[key].detach() if torch.is_tensor(batch[key]) else batch[key]
            if f"{phase}_{key}" in loss_meters:
                if should_update_host_meter and hasattr(loss_meters[f"{phase}_{key}"], "update_host"):
                    value_item = value.item() if torch.is_tensor(value) else value
                    loss_meters[f"{phase}_{key}"].update_host(value_item, batch_size)
                elif phase != "train" and should_update_host_meter:
                    value_item = value.item() if torch.is_tensor(value) else value
                    loss_meters[f"{phase}_{key}"].update(value_item, batch_size)
                else:
                    loss_meters[f"{phase}_{key}"].update(value, batch_size)
            if should_log_tb:
                value_item = value.item() if torch.is_tensor(value) else value
                self.tb_writer.log(f"Values/{phase}/{key}", value_item, step)

    def _apply_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _log_tpu_runtime_environment(self) -> None:
        if self.is_master:
            print(
                f"TPU runtime: rank={self.rank} local_rank={self.local_rank} world_size={self.world_size}",
                flush=True,
            )
            print(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}", flush=True)

    def _setup_device(self, device):
        del device
        self.device = self.xm.xla_device()

    def _setup_cuda_backend(self, cuda_conf) -> None:
        del cuda_conf
        return

    def _setup_runtime_objects(self) -> None:
        if self.world_size <= 1:
            self._setup_runtime_objects_for_rank()
            return

        # Model/dataset construction can briefly allocate large CPU buffers
        # (for example, Hugging Face from_pretrained state dicts). Serializing
        # startup avoids 8 TPU workers peaking at the same time on Kaggle.
        for setup_rank in range(self.world_size):
            if self.rank == setup_rank:
                self._setup_runtime_objects_for_rank()
                self._mark_step()
                gc.collect()
            self.xm.rendezvous(f"runtime_objects_rank_{setup_rank}_ready")

    def _setup_runtime_objects_for_rank(self) -> None:
        self._master_print(f"Setting up TPU runtime objects for rank {self.rank}/{self.world_size}")
        self._log_memory_snapshot("before_setup_components")
        self._setup_components()
        self._log_memory_snapshot("after_setup_components")
        gc.collect()
        self._log_memory_snapshot("after_setup_components_gc")
        self._setup_dataloaders()
        self._log_memory_snapshot(
            "after_setup_dataloaders",
            extra=self._describe_dataset_state(getattr(self.train_loader_cpu, "dataset", None)),
        )
        gc.collect()
        self._log_memory_snapshot("after_setup_dataloaders_gc")

        self._log_memory_snapshot("before_model_to_device")
        self.model.to(self.device)
        if self.loss:
            self.loss.to(self.device)
        self._mark_step()
        self._log_memory_snapshot("after_model_to_device")
        gc.collect()
        self._log_memory_snapshot("after_model_to_device_gc")

    def _setup_components(self):
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}
        self.meters = None
        self.tb_writer = None
        tb_conf = self.logging_conf.get("tensorboard_writer")
        if tb_conf and self.is_master:
            self.tb_writer = instantiate(tb_conf, _recursive_=False)
        self._log_memory_snapshot("before_model_instantiate")
        if self._prefork_model is not None:
            self.model = self._prefork_model
            self._prefork_model = None
        else:
            self.model = instantiate(self.model_conf, _recursive_=False)
        self._log_memory_snapshot("after_model_instantiate")
        self._native_frozen_param_names = {
            name for name, param in self.model.named_parameters() if not param.requires_grad
        }
        if getattr(self.optim_conf, "frozen_module_names", None):
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
        self._log_memory_snapshot("before_loss_instantiate")
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self._log_memory_snapshot("after_loss_instantiate")
        self.scaler = _NullScaler()
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self._log_memory_snapshot("after_gradient_clipper_instantiate")

    def _setup_dataloaders(self):
        self.train_loader = None
        self.train_loader_cpu = None
        self.val_loader = None
        self.val_loader_cpu = None
        self.test_loader = None
        self.test_loader_cpu = None

        data_module = instantiate(self.data_conf.data_module, _recursive_=False)
        self._log_memory_snapshot("after_datamodule_instantiate")
        data_module.seed = self.seed_value
        data_module.world_size = self.world_size
        data_module.rank = self.rank
        if self._prefork_train_dataset is not None:
            data_module.prefork_train_dataset = self._prefork_train_dataset
            self._prefork_train_dataset = None
        if hasattr(data_module, "accum_steps"):
            data_module.accum_steps = int(self.accum_steps)
        if hasattr(data_module, "train_config") and isinstance(data_module.train_config, Mapping):
            data_module.train_config["accum_steps"] = int(self.accum_steps)
        self.data_module = data_module

        if self.mode in ["train"]:
            self._log_memory_snapshot("before_train_dataloader_build")
            self.train_loader_cpu = data_module.train_dataloader()
            self._log_memory_snapshot(
                "after_train_dataloader_build",
                extra=self._describe_dataset_state(getattr(self.train_loader_cpu, "dataset", None)),
            )
            self.train_loader = self._make_mp_device_loader(self.train_loader_cpu)
            self._log_memory_snapshot("after_train_mp_device_loader")
        elif self.mode == "val":
            self._ensure_test_loader()

    def _wrap_cpu_loaders(self, loaders):
        return [self._make_mp_device_loader(loader) for loader in loaders]

    def _make_mp_device_loader(self, loader):
        kwargs = {
            "loader_prefetch_size": self.mp_loader_prefetch_size,
            "device_prefetch_size": self.mp_device_prefetch_size,
        }
        try:
            return self.pl.MpDeviceLoader(loader, self.device, **kwargs)
        except TypeError:
            if self.is_master:
                logging.warning("MpDeviceLoader prefetch kwargs unsupported; using torch_xla defaults.")
            return self.pl.MpDeviceLoader(loader, self.device)

    def _ensure_val_loader(self):
        if self.val_loader is None:
            self.val_loader_cpu = self.data_module.val_dataloader()
            self.val_loader = self._wrap_cpu_loaders(self.val_loader_cpu)
        return self.val_loader

    def _ensure_test_loader(self):
        if self.test_loader is None:
            self.test_loader_cpu = self.data_module.test_dataloader()
            self.test_loader = self._wrap_cpu_loaders(self.test_loader_cpu)
        return self.test_loader

    def _release_val_loader(self) -> None:
        self.val_loader = None
        self.val_loader_cpu = None
        gc.collect()

    def _set_eval_loaders_epoch(self, loaders, epoch: int) -> None:
        if not loaders:
            return
        for loader in loaders:
            self._set_loader_epoch(loader, epoch)

    def save_checkpoint(self, epoch, checkpoint_names=None):
        checkpoint_folder = self.checkpoint_conf.save_dir
        if self.is_master:
            safe_makedirs(checkpoint_folder)
        self.xm.rendezvous("save_checkpoint_dir_ready")
        checkpoint_names = checkpoint_names or ["checkpoint"]
        if not (
            self.checkpoint_conf.save_freq > 0
            and int(epoch + 1) % self.checkpoint_conf.save_freq == 0
        ):
            return

        if not self.is_master:
            self.xm.rendezvous(f"save_checkpoint_{epoch}_done")
            return

        self._log_memory_snapshot("before_checkpoint_build", extra=f"epoch={epoch}")
        checkpoint_content = {
            "epoch": epoch,
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
            "trainer_config": OmegaConf.to_container(self.cfg, resolve=True),
            "metrics_history": self.metrics_history,
            "rng_state": {
                "torch": torch.get_rng_state(),
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

        checkpoint_content["model"] = self.model.state_dict()
        self._log_memory_snapshot(
            "after_checkpoint_build",
            extra=f"epoch={epoch} checkpoint_keys={sorted(checkpoint_content.keys())}",
        )
        for ckpt_name in checkpoint_names:
            checkpoint_path = os.path.join(checkpoint_folder, f"{ckpt_name}.pt")
            self.xm.save(checkpoint_content, checkpoint_path, master_only=True)
        self._log_memory_snapshot("after_checkpoint_save", extra=f"epoch={epoch}")
        self.xm.rendezvous(f"save_checkpoint_{epoch}_done")

    def run(self):
        mode = self.mode
        assert mode in ["train", "val"]
        if mode == "train":
            self.run_train()
            if self.cfg.get("test", False):
                self.run_val(val_loader=self._ensure_test_loader())
        elif mode == "val":
            self.run_val(val_loader=self._ensure_test_loader())
        self._mark_step()
        gc.collect()

    def run_train(self):
        last_train_epoch_duration_sec = 0.0
        last_val_epoch_duration_sec = 0.0
        limit_sec = None
        if self.total_run_time_hr is not None:
            try:
                limit_sec = float(self.total_run_time_hr) * 3600.0
            except (TypeError, ValueError):
                limit_sec = None
        while self.epoch < self.max_epochs:
            self.end_warmup()
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.rank)
            self._apply_train_aug_schedule(self.epoch)
            self._set_loader_epoch(self.train_loader_cpu, self.epoch)
            self._log_memory_snapshot("epoch_start", extra=f"epoch={self.epoch}")

            train_epoch_start_time = time.time()
            ok = self.train_epoch(self.train_loader)
            last_train_epoch_duration_sec = time.time() - train_epoch_start_time
            self._log_memory_snapshot(
                "after_train_epoch",
                extra=f"epoch={self.epoch} duration_sec={last_train_epoch_duration_sec:.1f}",
            )
            if ok is False:
                break

            self._log_memory_snapshot("before_save_checkpoint", extra=f"epoch={self.epoch}")
            self.save_checkpoint(self.epoch)
            self._log_memory_snapshot("after_save_checkpoint", extra=f"epoch={self.epoch}")
            if (self.epoch + 1) % self.optim_conf.val_freq == 0:
                val_epoch_start_time = time.time()
                self._log_memory_snapshot("before_ensure_val_loader", extra=f"epoch={self.epoch}")
                val_loader = self._ensure_val_loader()
                self._log_memory_snapshot(
                    "after_ensure_val_loader",
                    extra=f"epoch={self.epoch} num_val_loaders={len(val_loader) if val_loader else 0}",
                )
                self._set_eval_loaders_epoch(self.val_loader_cpu, self.epoch)
                self._log_memory_snapshot("before_val_epoch", extra=f"epoch={self.epoch}")
                self.run_val(val_loader=val_loader, epoch=self.epoch)
                self._log_memory_snapshot("after_val_epoch", extra=f"epoch={self.epoch}")
                self._release_val_loader()
                self._log_memory_snapshot("after_release_val_loader", extra=f"epoch={self.epoch}")
                last_val_epoch_duration_sec = time.time() - val_epoch_start_time

            self.epoch += 1
            if limit_sec is not None and limit_sec > 0:
                elapsed_sec = time.time() - self.start_time
                if elapsed_sec + last_train_epoch_duration_sec + last_val_epoch_duration_sec > limit_sec:
                    break
            if self.epoch == self.cfg.get("break_at", -1):
                break
        self.epoch -= 1

    def run_val(self, val_loader=None, epoch=0, is_fresh_epoch=True):
        if not val_loader:
            return
        outs = self.val_epoch(val_loader, is_fresh_epoch=is_fresh_epoch)
        outs_json = self._to_jsonable(outs)
        if self.tb_writer is not None:
            self.tb_writer.log_dict(outs_json, epoch)
        if self.is_master:
            self._log_epoch_metrics_to_csv(
                "val",
                {k: v for k, v in outs_json.items() if isinstance(v, (int, float))},
            )
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as file_obj:
                file_obj.write(json.dumps(outs_json) + "\n")
        self._mark_step()

    def train_epoch(self, train_loader):
        self._log_memory_snapshot("train_epoch_entry", extra=f"epoch={self.epoch}")
        batch_time = AverageMeter("Batch Time", None, ":.4f")
        data_time = AverageMeter("Data Time", None, ":.4f")
        self.model.train()
        self._log_memory_snapshot("train_epoch_after_model_train", extra=f"epoch={self.epoch}")
        phase = "train"
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: _TPUTrainMeter(name, ":.4f") for name in loss_names
        }
        for config in self.gradient_clipper.configs:
            param_names = ",".join(config["module_names"])
            loss_meters[f"Grad/{param_names}"] = _TPUTrainMeter(
                f"Grad/{param_names}", ":.4f"
            )
        self._log_memory_snapshot("train_epoch_after_meter_setup", extra=f"epoch={self.epoch}")

        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[batch_time, data_time, self.time_elapsed_meter],
            real_meters={},
            prefix=f"Train Epoch: [{self.epoch}]",
        )
        self._log_memory_snapshot("train_epoch_after_progress_setup", extra=f"epoch={self.epoch}")
        end = time.time()

        if self.gradient_clipper is not None:
            self.gradient_clipper.setup_clipping(self.model)
        self._log_memory_snapshot("train_epoch_after_clip_setup", extra=f"epoch={self.epoch}")

        iters_per_epoch = int(len(train_loader))
        limit_train_batches = (
            iters_per_epoch if self.limit_train_batches is None else self.limit_train_batches
        )
        self._log_memory_snapshot(
            "train_epoch_after_len",
            extra=f"epoch={self.epoch} iters_per_epoch={iters_per_epoch} limit={limit_train_batches}",
        )

        for optim in self.optims:
            optim.zero_grad(set_to_none=True)
        self._log_memory_snapshot("train_epoch_after_zero_grad", extra=f"epoch={self.epoch}")
        self.where = min(float(self.epoch) / self.max_epochs, 1.0)
        if self.where < 1.0:
            self._log_memory_snapshot(
                "train_epoch_before_scheduler_step",
                extra=f"epoch={self.epoch} where={self.where:.6f}",
            )
            for optim in self.optims:
                optim.step_schedulers(self.where)
            self._log_memory_snapshot(
                "train_epoch_after_scheduler_step",
                extra=f"epoch={self.epoch} where={self.where:.6f}",
            )

        self._log_memory_snapshot("train_iter_create_start", extra=f"epoch={self.epoch}")
        train_iter = iter(train_loader)
        self._log_memory_snapshot("train_iter_create_end", extra=f"epoch={self.epoch}")

        if self.debug_cpu_first_batch and getattr(self, "train_loader_cpu", None) is not None:
            self._log_memory_snapshot("train_cpu_first_next_start", extra=f"epoch={self.epoch}")
            cpu_iter = iter(self.train_loader_cpu)
            try:
                cpu_batch = next(cpu_iter)
                self._log_memory_snapshot(
                    "train_cpu_first_next_end",
                    extra=f"epoch={self.epoch} {self._describe_batch_state(cpu_batch)}",
                )
            finally:
                del cpu_iter
                if "cpu_batch" in locals():
                    del cpu_batch

        for data_iter in range(limit_train_batches):
            log_fetch_boundary = data_iter == 0 or self._should_log_train_iter_detail(data_iter, limit_train_batches)
            if log_fetch_boundary:
                self._log_memory_snapshot(
                    "train_iter_next_start",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
                self._log_xla_metrics_snapshot(
                    "train_iter_next_start",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            if log_fetch_boundary:
                self._log_memory_snapshot(
                    "train_iter_next_end",
                    extra=(
                        f"epoch={self.epoch} iter={data_iter}/{limit_train_batches} "
                        f"{self._describe_batch_state(batch)}"
                    ),
                )
                self._log_xla_metrics_snapshot(
                    "train_iter_next_end",
                    extra=(
                        f"epoch={self.epoch} iter={data_iter}/{limit_train_batches} "
                        f"{self._describe_batch_state(batch)}"
                    ),
                )
            data_time.update(time.time() - end)
            batch_size = batch["img"].shape[0]
            if batch_size <= 0:
                continue
            self._debug_step_iter = data_iter
            self._debug_step_iter_limit = limit_train_batches
            if self._should_log_train_iter_detail(data_iter, limit_train_batches):
                self._log_memory_snapshot(
                    "train_iter_start",
                    extra=(
                        f"epoch={self.epoch} iter={data_iter}/{limit_train_batches} "
                        f"{self._describe_batch_state(batch)}"
                    ),
                )

            if self._should_log_train_iter_detail(data_iter, limit_train_batches):
                self._log_memory_snapshot(
                    "train_iter_before_step",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
            with self._autocast_context():
                loss_dict = self._step(
                    batch,
                    self.model,
                    phase,
                    loss_meters,
                    log_enabled=True,
                    log_step=self.steps[phase],
                )
            if self._should_log_train_iter_detail(data_iter, limit_train_batches):
                self._log_memory_snapshot(
                    "train_iter_after_step",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
            loss = loss_dict["objective"]
            loss_value = loss.detach()
            if self.check_loss_finite and not torch.isfinite(loss_value).all():
                return False
            if self._should_log_train_iter_detail(data_iter, limit_train_batches):
                self._log_memory_snapshot(
                    "train_iter_before_backward",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
            self.scaler.scale(loss).backward()
            if self._should_log_train_iter_detail(data_iter, limit_train_batches):
                self._log_memory_snapshot(
                    "train_iter_after_backward",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
            log_iter_detail = self._should_log_train_iter_detail(data_iter, limit_train_batches)
            update_host_metrics = self._should_update_train_host_metrics(data_iter, limit_train_batches)
            if log_iter_detail and update_host_metrics:
                self._log_memory_snapshot(
                    "train_iter_before_objective_item",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
            if update_host_metrics and hasattr(loss_meters[f"{phase}_loss_objective"], "update_host"):
                loss_meters[f"{phase}_loss_objective"].update_host(loss_value.item(), batch_size)
            else:
                loss_meters[f"{phase}_loss_objective"].update(loss_value, batch_size)
            if log_iter_detail and update_host_metrics:
                self._log_memory_snapshot(
                    "train_iter_after_objective_item",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )

            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            if self.gradient_clipper is not None:
                if log_iter_detail:
                    self._log_memory_snapshot(
                        "train_iter_before_grad_clip",
                        extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                    )
                grad_norm_dict = self.gradient_clipper(model=self.model)
                if log_iter_detail:
                    self._log_memory_snapshot(
                        "train_iter_after_grad_clip",
                        extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                    )
                for key, grad_norm in grad_norm_dict.items():
                    if log_iter_detail and update_host_metrics:
                        self._log_memory_snapshot(
                            "train_iter_before_grad_norm_item",
                            extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches} key={key}",
                        )
                    if update_host_metrics and hasattr(loss_meters[f"Grad/{key}"], "update_host"):
                        loss_meters[f"Grad/{key}"].update_host(grad_norm.item())
                    else:
                        loss_meters[f"Grad/{key}"].update(grad_norm)
                    if log_iter_detail and update_host_metrics:
                        self._log_memory_snapshot(
                            "train_iter_after_grad_norm_item",
                            extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches} key={key}",
                        )
            if log_iter_detail:
                self._log_memory_snapshot(
                    "train_iter_before_optimizer_step",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
                self._log_xla_metrics_snapshot(
                    "train_iter_before_optimizer_step",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
            skip_optimizer_step = self.debug_skip_first_optimizer_step and data_iter == 0
            if skip_optimizer_step:
                self._log_memory_snapshot(
                    "train_iter_skip_optimizer_step",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
            for optim in self.optims:
                if not skip_optimizer_step:
                    if log_iter_detail:
                        self._log_memory_snapshot(
                            "train_iter_before_xm_optimizer_step",
                            extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                        )
                        self._log_xla_metrics_snapshot(
                            "train_iter_before_xm_optimizer_step",
                            extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                        )
                    self._optimizer_step(optim.optimizer)
                    if log_iter_detail:
                        self._log_memory_snapshot(
                            "train_iter_after_xm_optimizer_step",
                            extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                        )
                        self._log_xla_metrics_snapshot(
                            "train_iter_after_xm_optimizer_step",
                            extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                        )
                if log_iter_detail:
                    self._log_memory_snapshot(
                        "train_iter_before_zero_grad",
                        extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                    )
                optim.zero_grad(set_to_none=True)
                if log_iter_detail:
                    self._log_memory_snapshot(
                        "train_iter_after_zero_grad",
                        extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                    )
            if self.gradient_clipper is not None:
                if log_iter_detail:
                    self._log_memory_snapshot(
                        "train_iter_before_grad_norm_delete",
                        extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                    )
                del grad_norm_dict
                if log_iter_detail:
                    self._log_memory_snapshot(
                        "train_iter_after_grad_norm_delete",
                        extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                    )
            if log_iter_detail:
                self._log_memory_snapshot(
                    "train_iter_before_loss_delete",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
            del loss, loss_dict, loss_value
            if log_iter_detail:
                self._log_memory_snapshot(
                    "train_iter_after_loss_delete",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
            if log_iter_detail:
                self._log_memory_snapshot(
                    "train_iter_after_optimizer_step",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
                self._log_xla_metrics_snapshot(
                    "train_iter_after_optimizer_step",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )

            if log_iter_detail:
                self._log_memory_snapshot(
                    "train_iter_end",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches} where={self.where:.6f}",
                )
                self._log_xla_metrics_snapshot(
                    "train_iter_end",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches} where={self.where:.6f}",
                )
            batch_time.update(time.time() - end)
            if log_iter_detail:
                self._log_memory_snapshot(
                    "train_iter_after_batch_time_update",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
            end = time.time()
            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)
            if log_iter_detail:
                self._log_memory_snapshot(
                    "train_iter_after_elapsed_update",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
            if self._should_display_progress(data_iter, limit_train_batches) and self.is_master:
                if log_iter_detail:
                    self._log_memory_snapshot(
                        "train_iter_before_progress_display",
                        extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                    )
                progress.display(data_iter)
                if log_iter_detail:
                    self._log_memory_snapshot(
                        "train_iter_after_progress_display",
                        extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                    )
            del batch
            if log_iter_detail:
                self._log_memory_snapshot(
                    "train_iter_after_batch_delete",
                    extra=f"epoch={self.epoch} iter={data_iter}/{limit_train_batches}",
                )
                if data_iter + 1 < limit_train_batches:
                    self._log_memory_snapshot(
                        "train_iter_before_next_loop",
                        extra=f"epoch={self.epoch} next_iter={data_iter + 1}/{limit_train_batches}",
                    )
                    self._log_xla_metrics_snapshot(
                        "train_iter_before_next_loop",
                        extra=f"epoch={self.epoch} next_iter={data_iter + 1}/{limit_train_batches}",
                    )
        self._debug_step_iter = None
        self._debug_step_iter_limit = None

        if self.is_master:
            self._log_epoch_metrics_to_csv("train", loss_meters)
        self._mark_step()
        return True

    def val_epoch(self, val_loader, is_fresh_epoch: bool):
        del is_fresh_epoch
        curr_phases = ["val"]
        curr_models = [self.model]
        phase = curr_phases[0]

        for model in curr_models:
            model.eval()
            if hasattr(model, "on_validation_epoch_start"):
                model.on_validation_epoch_start()

        all_metrics = {}
        for current_val_loader in val_loader:
            batch_time = AverageMeter("Batch Time", None, ":.4f")
            data_time = AverageMeter("Data Time", None, ":.4f")
            iters_per_epoch = len(current_val_loader)
            loss_names = ["objective"] + self._get_scalar_log_keys(phase)
            loss_names = [f"{phase}_{name}" for name in loss_names]
            loss_meters = {
                name: AverageMeter(name, None, ":.4f") for name in loss_names
            }
            progress = ProgressMeter(
                iters_per_epoch,
                [batch_time, data_time, self.time_elapsed_meter],
                {},
                prefix=f"Val Epoch: [{self.epoch}]",
            )
            end = time.time()
            dataset_name = None
            limit_val_batches = (
                iters_per_epoch if self.limit_val_batches is None else self.limit_val_batches
            )
            for data_iter, batch in enumerate(current_val_loader):
                if data_iter >= limit_val_batches:
                    break
                data_time.update(time.time() - end)
                if dataset_name is None:
                    dataset_name = batch["dataset"][0][0]
                    progress.prefix = f"Val Epoch: [{self.epoch}] ({dataset_name})"
                if self._should_log_train_iter_detail(data_iter, limit_val_batches):
                    self._log_memory_snapshot(
                        "val_iter_start",
                        extra=(
                            f"epoch={self.epoch} iter={data_iter}/{limit_val_batches} "
                            f"{self._describe_batch_state(batch)}"
                        ),
                    )
                # Validation postprocess/eval mutates prediction dictionaries and
                # tensor views in-place. inference_mode tensors reject version
                # counter updates for those writes, so use no_grad instead.
                with torch.no_grad():
                    with self._autocast_context():
                        for phase_name, model in zip(curr_phases, curr_models):
                            self._val_step(
                                batch,
                                model,
                                phase_name,
                                loss_meters,
                            )
                if self._should_log_train_iter_detail(data_iter, limit_val_batches):
                    self._log_memory_snapshot(
                        "val_iter_end",
                        extra=f"epoch={self.epoch} iter={data_iter}/{limit_val_batches}",
                    )
                batch_time.update(time.time() - end)
                end = time.time()
                self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)
                if self._should_display_progress(data_iter, limit_val_batches) and self.is_master:
                    progress.display(data_iter)
                del batch

            self._synchronize_loss_meters(loss_meters)
            for key, value in loss_meters.items():
                metric_name = key.split("/")[-1].replace(f"{phase}_", "")
                all_metrics[f"{phase}_{dataset_name}_{metric_name}"] = value.avg

        for model in curr_models:
            if hasattr(model, "on_validation_epoch_end"):
                model.on_validation_epoch_end()
        for phase_name in curr_phases:
            all_metrics.update(self._get_trainer_state(phase_name))
        self._mark_step()
        return all_metrics

    def _autocast_context(self):
        if not self.optim_conf.amp.enabled:
            return nullcontext()
        return torch.autocast(device_type=self.autocast_device_type, dtype=self.amp_type)

    def _mark_step(self) -> None:
        self.xm.mark_step()

    def _optimizer_step(self, optimizer) -> None:
        self.xm.optimizer_step(optimizer, barrier=False)
        self._mark_step()

    def _set_loader_epoch(self, loader, epoch: int) -> None:
        if loader is None:
            return
        if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)

    def _synchronize_loss_meters(self, loss_meters: Dict[str, Any]) -> None:
        for meter in loss_meters.values():
            if not hasattr(meter, "sum") or not hasattr(meter, "count"):
                continue
            if torch.is_tensor(meter.sum):
                sum_tensor = meter.sum.detach().to(self.device)
            else:
                sum_tensor = torch.tensor(float(meter.sum), device=self.device)
            if torch.is_tensor(meter.count):
                count_tensor = meter.count.detach().to(self.device)
            else:
                count_tensor = torch.tensor(float(meter.count), device=self.device)
            if self.world_size > 1:
                reduced = self.xm.all_reduce(self.xm.REDUCE_SUM, [sum_tensor, count_tensor])
                if reduced is not None:
                    sum_tensor, count_tensor = reduced
            avg_tensor = sum_tensor / count_tensor.clamp_min(1.0)
            meter.sum = float(sum_tensor.item())
            meter.count = float(count_tensor.item())
            meter.avg = float(avg_tensor.item())
            meter.val = float(meter.val.item()) if torch.is_tensor(meter.val) else float(meter.val)

    def _get_train_dataset_checkpoint_state(self):
        original_loader = getattr(self, "train_loader", None)
        try:
            self.train_loader = getattr(self, "train_loader_cpu", None)
            return super()._get_train_dataset_checkpoint_state()
        finally:
            self.train_loader = original_loader

    def _restore_train_dataset_checkpoint_state(self, checkpoint_state):
        original_loader = getattr(self, "train_loader", None)
        try:
            self.train_loader = getattr(self, "train_loader_cpu", None)
            return super()._restore_train_dataset_checkpoint_state(checkpoint_state)
        finally:
            self.train_loader = original_loader

    def _apply_train_aug_schedule(self, epoch):
        original_loader = getattr(self, "train_loader", None)
        try:
            self.train_loader = getattr(self, "train_loader_cpu", None)
            return super()._apply_train_aug_schedule(epoch)
        finally:
            self.train_loader = original_loader
