import argparse
import math
import os
import resource
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from packaging import version

_IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"))
_N3_GITHUB_ZIP_URL = os.environ.get(
    "N3R_GITHUB_ZIP_URL", "https://github.com/lc126eml/n3/archive/refs/heads/master.zip"
)

# Managed by kaggle/process_kaggle.py. Dot-path overrides applied after Hydra compose.
# BEGIN_KAGGLE_RUNTIME_OVERRIDES
KAGGLE_RUNTIME_CONFIG_NAME = 'default_kaggle_tpu'
KAGGLE_RUNTIME_CONFIG_OVERRIDES = {'accum_steps': 1,
 'break_at': 130,
 'checkpoint.resume_checkpoint_path': None,
 'checkpoint.resume_config_skip_keys': [],
 'data.data_module.train_config.batch_size': 64,
 'data.data_module.train_config.debug_enumerate_batches': False,
 'loss.switch.gt_align_to_pts': False,
 'loss.switch.pts_align_to_center': False,
 'loss.switch.pts_align_to_gt': False,
 'loss.switch.pts_center_world': False,
 'max_epochs': 130,
 'mode': 'train',
 'model.enable_depth': False,
 'optim.options.lr.0.scheduler.schedulers.0.end_value': 0.0003,
 'optim.patch_embed_lr_ratio': 0.01,
 'optim.val_freq': 1,
 'optim.warmup_batch_cost_discount': 0.55,
 'optim.warmup_epochs': 30,
 'postprocess.train.align.center_world.align_pose': True,
 'postprocess.train.align.center_world.enabled': False,
 'postprocess.train.align.gt_align_to_pts.align_pose': True,
 'postprocess.train.align.gt_align_to_pts.conf_percentage': 80,
 'postprocess.train.align.gt_align_to_pts.enabled': False,
 'postprocess.train.align.pred_center.align_pose': True,
 'postprocess.train.align.pred_center.enabled': False,
 'postprocess.train.align.pred_center.pr_to_gt': False,
 'postprocess.train.align.pts_align_to_gt.align_pose': True,
 'postprocess.train.align.pts_align_to_gt.conf_percentage': 80,
 'postprocess.train.align.pts_align_to_gt.enabled': False,
 'postprocess.train.align.pts_align_to_gt.normalize_depth': False,
 'postprocess.train.align.pts_align_to_gt.normalize_pose': False,
 'postprocess.train.align.pts_align_to_gt.with_scale': True,
 'postprocess.train.align.to_first_cam.enabled': True,
 'postprocess.train.align.to_first_cam.points': True,
 'postprocess.train.normalize.gt_depth': False,
 'postprocess.train.normalize.gt_pts': True,
 'postprocess.train.normalize.gt_pts_invariant.enabled': False,
 'postprocess.train.normalize.gt_pts_invariant.translate': True,
 'postprocess.train.normalize.pr_pts.enabled': False,
 'postprocess.train.normalize.pr_pts.metric': False,
 'postprocess.train.normalize.pr_pts_invariant.enabled': False,
 'postprocess.train.normalize.pr_pts_invariant.translate': False,
 'resume_bs': True,
 'seed_value': 42,
 'total_run_time_hr': 11.6}
# END_KAGGLE_RUNTIME_OVERRIDES

_TPU_TRAINER_TARGET = "training.trainer_tpu.TrainerTPU"
_TPU_DATA_MODULE_TARGET = (
    "training.datasets.base.standalone_multiview_datamodule_tpu.StandaloneMultiViewDataModuleTPU"
)
_PREFORK_MODEL = None
_PREFORK_TRAIN_DATASET = None
_WORKER_DEBUG_INDEX = int(os.environ.get("TPU_WORKER_DEBUG_INDEX", "0"))


def _resolve_config_name(cli_config_name: str | None) -> str:
    if cli_config_name:
        return cli_config_name
    if isinstance(KAGGLE_RUNTIME_CONFIG_NAME, str) and KAGGLE_RUNTIME_CONFIG_NAME.strip():
        return KAGGLE_RUNTIME_CONFIG_NAME.strip()
    return "default_kaggle"


def _host_memory_snapshot_mb() -> dict[str, float]:
    out: dict[str, float] = {}
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


def _format_memory_snapshot(prefix: str) -> str:
    mem = _host_memory_snapshot_mb()
    if not mem:
        return f"[launch_tpu] mem stage={prefix} unavailable"
    ordered_keys = ("VmRSS", "VmHWM", "VmSize", "VmSwap", "ru_maxrss")
    parts = [f"{key}={mem[key]:.1f}MB" for key in ordered_keys if key in mem]
    return f"[launch_tpu] mem stage={prefix} pid={os.getpid()} " + " ".join(parts)


def _log_memory_snapshot(prefix: str) -> None:
    print(_format_memory_snapshot(prefix), flush=True)


def _apply_kaggle_runtime_overrides(cfg) -> None:
    overrides = KAGGLE_RUNTIME_CONFIG_OVERRIDES
    if not isinstance(overrides, dict) or not overrides:
        return
    from omegaconf import OmegaConf

    applied = []
    for key, value in overrides.items():
        if not isinstance(key, str) or not key:
            continue
        OmegaConf.update(cfg, key, value, merge=False)
        applied.append(key)
    if applied:
        print(f"[launch_tpu] Applied runtime config overrides: {applied}", flush=True)


def _apply_kaggle_tpu_policy(cfg) -> None:
    from omegaconf import OmegaConf

    OmegaConf.update(cfg, "data.data_module._target_", _TPU_DATA_MODULE_TARGET, merge=False, force_add=True)

    optim_conf = cfg.get("optim")
    if optim_conf is not None and getattr(optim_conf, "amp", None):
        optim_conf.amp.enabled = True
        optim_conf.amp.amp_dtype = "bfloat16"

    if _IS_KAGGLE:
        user_keys = {
            "data.data_module.train_config.num_workers",
            "data.data_module.validation_config.num_workers",
            "data.data_module.test_config.num_workers",
            "data.data_module.persistent_workers",
            "data.data_module.prefetch_factor",
        }
        defaults = {
            "data.data_module.train_config.num_workers": 0,
            "data.data_module.validation_config.num_workers": 0,
            "data.data_module.test_config.num_workers": 0,
            "data.data_module.persistent_workers": False,
            "data.data_module.prefetch_factor": None,
        }
        for key in user_keys:
            current_value = OmegaConf.select(cfg, key, default=None)
            if current_value is None and key in defaults:
                OmegaConf.update(cfg, key, defaults[key], merge=False, force_add=True)


def _find_repo_root(start_file: str) -> Path | None:
    def _candidate_roots() -> list[Path]:
        out: list[Path] = []
        seen = set()

        def _add(path: Path | None) -> None:
            if path is None:
                return
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            key = str(resolved)
            if key in seen:
                return
            seen.add(key)
            out.append(resolved)

        env_repo = os.environ.get("N3R_REPO_ROOT")
        if env_repo:
            _add(Path(env_repo))

        _add(Path("/kaggle/working/n3"))
        _add(Path.cwd())

        start = Path(start_file).resolve()
        _add(start.parent)
        for parent in start.parents:
            _add(parent)
        for parent in Path.cwd().resolve().parents:
            _add(parent)

        kaggle_working = Path("/kaggle/working")
        if kaggle_working.is_dir():
            for child in kaggle_working.iterdir():
                if child.is_dir():
                    _add(child)

        return out

    for cand in _candidate_roots():
        if _looks_like_n3_repo(cand):
            return cand
    return None


def _setup_project_root() -> Path:
    project_root = _find_repo_root(__file__)
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("PROJECT_ROOT", str(project_root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    training_root = project_root / "training"
    if training_root.is_dir() and str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))
    return project_root


def _looks_like_n3_repo(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "training").is_dir()
        and (path / "dust3r").is_dir()
    )


def _download_with_retries(url: str, dst: Path, retries: int = 3, timeout: int = 30) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(dst) + ".part"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/zip",
    }

    def _cleanup_tmp() -> None:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    def _finalize() -> None:
        if not zipfile.is_zipfile(tmp_path):
            raise RuntimeError("Downloaded file is not a zip.")
        os.replace(tmp_path, str(dst))

    def _try_curl() -> bool:
        curl = shutil.which("curl")
        if not curl:
            print("[launch_tpu] download: curl not found", flush=True)
            return False
        print("[launch_tpu] download: trying curl", flush=True)
        cmd = [
            curl,
            "-L",
            "--retry",
            "3",
            "--retry-all-errors",
            "--max-redirs",
            "20",
            "--connect-timeout",
            str(timeout),
            "-H",
            f"User-Agent: {headers['User-Agent']}",
            "-H",
            f"Accept: {headers['Accept']}",
            "-o",
            tmp_path,
            url,
        ]
        subprocess.check_call(cmd)
        _finalize()
        return True

    def _try_requests() -> bool:
        try:
            import requests  # type: ignore
        except Exception:
            print("[launch_tpu] download: requests not available", flush=True)
            return False
        print("[launch_tpu] download: trying requests", flush=True)
        resp = requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        with open(tmp_path, "wb") as file_obj:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)
        _finalize()
        return True

    def _try_urllib() -> bool:
        print("[launch_tpu] download: trying urllib", flush=True)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(tmp_path, "wb") as file_obj:
                shutil.copyfileobj(resp, file_obj)
        _finalize()
        return True

    methods = [_try_curl, _try_requests, _try_urllib]
    last_exc = None
    for attempt in range(1, retries + 1):
        for method in methods:
            try:
                if method():
                    print("[launch_tpu] download: ok", flush=True)
                    return
            except Exception as exc:
                last_exc = exc
                print(f"[launch_tpu] download: method failed ({exc})", flush=True)
                _cleanup_tmp()
        if attempt < retries:
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download zip from {url}. Last error: {last_exc}")


def _ensure_n3_repo_on_kaggle(current_root: Path) -> Path:
    if not _IS_KAGGLE:
        return current_root
    if _looks_like_n3_repo(current_root):
        return current_root

    env_repo = os.environ.get("N3R_REPO_ROOT")
    if env_repo and _looks_like_n3_repo(Path(env_repo)):
        repo_root = Path(env_repo)
        os.environ["PROJECT_ROOT"] = str(repo_root)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        training_root = repo_root / "training"
        if training_root.is_dir() and str(training_root) not in sys.path:
            sys.path.insert(0, str(training_root))
        return repo_root

    work_root = Path("/kaggle/working")
    target_root = work_root / "n3"
    if _looks_like_n3_repo(target_root):
        repo_root = target_root
    else:
        zip_path = work_root / "n3_repo.zip"
        if zip_path.exists() and not zipfile.is_zipfile(zip_path):
            zip_path.unlink()
        if not zip_path.exists():
            print(f"[launch_tpu] Downloading n3 repo from {_N3_GITHUB_ZIP_URL}", flush=True)
            try:
                _download_with_retries(_N3_GITHUB_ZIP_URL, zip_path)
            except Exception:
                if "master.zip" in _N3_GITHUB_ZIP_URL:
                    alt_url = _N3_GITHUB_ZIP_URL.replace("master.zip", "main.zip")
                elif "main.zip" in _N3_GITHUB_ZIP_URL:
                    alt_url = _N3_GITHUB_ZIP_URL.replace("main.zip", "master.zip")
                else:
                    raise
                print(f"[launch_tpu] Retrying with fallback URL: {alt_url}", flush=True)
                _download_with_retries(alt_url, zip_path)

        tmp_extract = work_root / "n3_extract_tmp"
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract)
        tmp_extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_extract)

        extracted_roots = [path for path in tmp_extract.iterdir() if path.is_dir()]
        if not extracted_roots:
            raise RuntimeError("Failed to extract n3 repo zip on Kaggle")
        extracted_root = extracted_roots[0]
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.move(str(extracted_root), str(target_root))
        shutil.rmtree(tmp_extract, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
        repo_root = target_root

    if not _looks_like_n3_repo(repo_root):
        raise RuntimeError(f"Downloaded repo does not look like n3: {repo_root}")

    os.environ["PROJECT_ROOT"] = str(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    training_root = repo_root / "training"
    if training_root.is_dir() and str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))
    os.environ.setdefault("N3R_REPO_ROOT", str(repo_root))
    return repo_root


def install_libs(lib_names: list[str]) -> None:
    libs = [str(item).strip() for item in lib_names if str(item).strip()]
    if not libs:
        return
    print(f"[launch_tpu] Installing Python libs: {libs}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *libs])


def _default_install_libs() -> list[str]:
    raw = os.environ.get("N3R_INSTALL_LIBS")
    if raw is not None:
        raw = raw.replace(",", " ")
        return [item for item in (s.strip() for s in raw.split()) if item]
    if _IS_KAGGLE:
        return [
            "hydra-core",
            "fvcore",
            "iopath",
            "einops",
            "safetensors",
            "wcmatch",
            "roma",
            "huggingface-hub>=1.0.0,<2.0",
        ]
    return []


def check_torch_version_above(target: str = "2.6.0") -> bool:
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", "torch"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines():
            if line.startswith("Version:"):
                installed_ver = line.split()[1]
                return version.parse(installed_ver) > version.parse(target)
    except Exception:
        return False
    return False


def _preflight_tpu_runtime() -> None:
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    if not _IS_KAGGLE:
        return

    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("TPU_PROCESS_COUNT", None)

    if os.environ.get("TPU_FIX_TF_DONE") != "1":
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tensorflow-cpu"])
            os.environ["TPU_FIX_TF_DONE"] = "1"
        except Exception as exc:
            print(f"[launch_tpu] WARNING: tensorflow-cpu setup failed ({exc}); continuing.", flush=True)

    if os.environ.get("TPU_ENV_DUMPED") != "1":
        tpu_env = {k: v for k, v in os.environ.items() if k.startswith("TPU") or k.startswith("XLA")}
        print(f"[launch_tpu] TPU_ENV: {tpu_env}", flush=True)
        os.environ["TPU_ENV_DUMPED"] = "1"


def _spawn_tpu(cfg) -> None:
    from omegaconf import OmegaConf

    _preflight_tpu_runtime()
    try:
        import torch_xla.distributed.xla_multiprocessing as xmp
    except Exception as exc:
        raise RuntimeError(f"TPU spawn failed: {exc}") from exc

    cfg_container = OmegaConf.to_container(cfg, resolve=False)
    start_method = os.environ.get("TPU_START_METHOD") or ("fork" if _IS_KAGGLE else "spawn")
    resume_path = OmegaConf.select(cfg, "checkpoint.resume_checkpoint_path", default=None)
    _log_memory_snapshot("spawn_start")
    if (
        start_method == "fork"
        and not resume_path
        and os.environ.get("TPU_PREFORK_MODEL", "1") != "0"
    ):
        global _PREFORK_MODEL
        from hydra.utils import instantiate
        import torch

        if OmegaConf.select(cfg, "model.conf_logit_max", default=None) is None:
            dtype = torch.float32
            if OmegaConf.select(cfg, "optim.amp.enabled", default=False):
                amp_dtype = OmegaConf.select(cfg, "optim.amp.amp_dtype", default=None)
                if amp_dtype == "float16":
                    dtype = torch.float16
                elif amp_dtype == "bfloat16":
                    dtype = torch.bfloat16
            OmegaConf.update(
                cfg,
                "model.conf_logit_max",
                float(math.log(torch.finfo(dtype).max) - 1.0),
                merge=False,
            )

        _log_memory_snapshot("before_prefork_model")
        print("[launch_tpu] prebuilding CPU model before fork", flush=True)
        _PREFORK_MODEL = instantiate(cfg.model, _recursive_=False)
        _log_memory_snapshot("after_prefork_model")
        print("[launch_tpu] prebuilt CPU model before fork", flush=True)
    if start_method == "fork" and os.environ.get("TPU_PREFORK_DATASET", "1") != "0":
        global _PREFORK_TRAIN_DATASET
        from training import datasets_tpu

        train_config = OmegaConf.to_container(
            cfg.data.data_module.train_config,
            resolve=True,
        )
        train_datasets_concat = " + ".join(train_config["datasets"])
        _log_memory_snapshot("before_prefork_dataset")
        print("[launch_tpu] prebuilding train dataset metadata before fork", flush=True)
        _PREFORK_TRAIN_DATASET = eval(train_datasets_concat, datasets_tpu.__dict__)
        _log_memory_snapshot("after_prefork_dataset")
        print("[launch_tpu] prebuilt train dataset metadata before fork", flush=True)
    _log_memory_snapshot("before_xmp_spawn")
    print(f"[launch_tpu] xmp.spawn nprocs=None start_method={start_method}", flush=True)
    xmp.spawn(_tpu_worker, args=(cfg_container,), nprocs=None, start_method=start_method)


def _tpu_worker(index: int, cfg_container) -> None:
    os.environ["RUN_TPU_WORKER"] = "1"
    from omegaconf import OmegaConf

    if index == _WORKER_DEBUG_INDEX:
        _log_memory_snapshot("worker_entry")
    cfg = OmegaConf.create(cfg_container)
    import training.trainer_tpu as trainer_tpu_mod

    trainer_tpu_mod._PREFORK_MODEL = _PREFORK_MODEL
    trainer_tpu_mod._PREFORK_TRAIN_DATASET = _PREFORK_TRAIN_DATASET
    if index == _WORKER_DEBUG_INDEX:
        _log_memory_snapshot("before_trainer_init")
    trainer = trainer_tpu_mod.TrainerTPU(cfg)
    if index == _WORKER_DEBUG_INDEX:
        _log_memory_snapshot("after_trainer_init")
    trainer.run()


def main() -> None:
    project_root = _setup_project_root()
    project_root = _ensure_n3_repo_on_kaggle(project_root)

    # Keep torch/torch_xla compatibility intact on TPU. Avoid the GPU launcher behavior
    # that may reinstall torch independently from the matching torch_xla build.
    if check_torch_version_above("2.6.0"):
        print("[launch_tpu] torch>2.6.0 detected; leaving torch unchanged to preserve torch_xla compatibility.", flush=True)

    install_libs(_default_install_libs())

    from hydra import compose, initialize_config_dir

    parser = argparse.ArgumentParser(description="Run n3 training on Kaggle TPU (Hydra config entrypoint).")
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Name of the config file to use (without .yaml extension).",
    )
    args = parser.parse_args()
    config_name = _resolve_config_name(args.config_name)

    config_dir = (project_root / "training" / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=[])

    if _IS_KAGGLE:
        _apply_kaggle_runtime_overrides(cfg)
    _apply_kaggle_tpu_policy(cfg)

    print(f"[launch_tpu] project_root={project_root}", flush=True)
    if KAGGLE_RUNTIME_CONFIG_NAME:
        print(f"[launch_tpu] Kaggle runtime config_name: {KAGGLE_RUNTIME_CONFIG_NAME}", flush=True)
    if KAGGLE_RUNTIME_CONFIG_OVERRIDES:
        print(f"[launch_tpu] Kaggle runtime overrides: {KAGGLE_RUNTIME_CONFIG_OVERRIDES}", flush=True)
    print(f"[launch_tpu] trainer_target={_TPU_TRAINER_TARGET}", flush=True)
    print(f"[launch_tpu] data_module_target={cfg.data.data_module._target_}", flush=True)
    print("[launch_tpu] device=xla", flush=True)

    _spawn_tpu(cfg)


if __name__ == "__main__":
    main()
