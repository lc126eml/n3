import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

_IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"))
_N3_GITHUB_ZIP_URL = os.environ.get(
    "N3R_GITHUB_ZIP_URL", "https://github.com/lc126eml/n3/archive/refs/heads/kaggle.zip"
)

def _find_repo_root(start_file: str) -> Path | None:
    def _candidate_roots() -> list[Path]:
        out: list[Path] = []
        seen = set()

        def _add(p: Path | None) -> None:
            if p is None:
                return
            try:
                rp = p.resolve()
            except Exception:
                rp = p
            key = str(rp)
            if key in seen:
                return
            seen.add(key)
            out.append(rp)

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
        and (path / ".project-root").exists()
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
            print("[launch] download: curl not found", flush=True)
            return False
        print("[launch] download: trying curl", flush=True)
        import subprocess

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
            print("[launch] download: requests not available", flush=True)
            return False
        print("[launch] download: trying requests", flush=True)
        resp = requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        _finalize()
        return True

    def _try_urllib() -> bool:
        print("[launch] download: trying urllib", flush=True)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(resp, f)
        _finalize()
        return True

    methods = [_try_curl, _try_requests, _try_urllib]
    last_exc = None
    for attempt in range(1, retries + 1):
        for method in methods:
            try:
                if method():
                    print("[launch] download: ok", flush=True)
                    return
            except Exception as exc:
                last_exc = exc
                print(f"[launch] download: method failed ({exc})", flush=True)
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
            print(f"[launch] Downloading n3 repo from {_N3_GITHUB_ZIP_URL}")
            try:
                _download_with_retries(_N3_GITHUB_ZIP_URL, zip_path)
            except Exception:
                if "master.zip" in _N3_GITHUB_ZIP_URL:
                    alt_url = _N3_GITHUB_ZIP_URL.replace("master.zip", "main.zip")
                elif "main.zip" in _N3_GITHUB_ZIP_URL:
                    alt_url = _N3_GITHUB_ZIP_URL.replace("main.zip", "master.zip")
                else:
                    raise
                print(f"[launch] Retrying with fallback URL: {alt_url}")
                _download_with_retries(alt_url, zip_path)

        tmp_extract = work_root / "n3_extract_tmp"
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract)
        tmp_extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_extract)

        extracted_roots = [p for p in tmp_extract.iterdir() if p.is_dir()]
        if not extracted_roots:
            raise RuntimeError("Failed to extract n3 repo zip on Kaggle")
        extracted_root = extracted_roots[0]
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.move(str(extracted_root), str(target_root))
        shutil.rmtree(tmp_extract, ignore_errors=True)
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


def _guess_kaggle_data_root() -> str | None:
    env_root = os.environ.get("N3R_DATA_ROOT")
    if env_root:
        return env_root

    candidates = []
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.is_dir():
        return None

    for d in kaggle_input.rglob("*"):
        if not d.is_dir():
            continue
        names = {p.name for p in d.iterdir() if p.is_dir()}
        score = 0
        for expected in ["hypersim_processed", "blendedmvs_processed", "scannetpp_processed"]:
            if expected in names:
                score += 1
        if score > 0:
            candidates.append((score, str(d)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def _default_kaggle_overrides(user_overrides: list[str]) -> list[str]:
    if not _IS_KAGGLE:
        return []

    overrides = []
    user_keys = {o.split("=", 1)[0] for o in user_overrides if "=" in o}

    if "logging.log_dir" not in user_keys:
        overrides.append("logging.log_dir=/kaggle/working/logs/${exp_name}/${logging.run_folder_name}")

    data_root = _guess_kaggle_data_root()
    if data_root and "data.data_root" not in user_keys:
        overrides.append(f"data.data_root={data_root}")

    # Kaggle usually works better with fewer workers.
    for key, value in [
        ("data.data_module.train_config.num_workers", "2"),
        ("data.data_module.validation_config.num_workers", "2"),
        ("data.data_module.test_config.num_workers", "2"),
    ]:
        if key not in user_keys:
            overrides.append(f"{key}={value}")

    return overrides


def install_libs(lib_names: list[str]) -> None:
    libs = [str(x).strip() for x in lib_names if str(x).strip()]
    if not libs:
        return
    print(f"[launch] Installing Python libs: {libs}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *libs])


def _default_install_libs() -> list[str]:
    raw = os.environ.get("N3R_INSTALL_LIBS")
    if raw is not None:
        raw = raw.replace(",", " ")
        return [x for x in (s.strip() for s in raw.split()) if x]
    if _IS_KAGGLE:
        return ["hydra-core", "fvcore", "iopath", "einops", "safetensors", "wcmatch", "roma"]
    return []


def main() -> None:
    lib_names = _default_install_libs()
    install_libs(lib_names)
    project_root = _setup_project_root()
    project_root = _ensure_n3_repo_on_kaggle(project_root)

    from hydra import compose, initialize_config_dir

    parser = argparse.ArgumentParser(description="Run n3 training (Hydra config entrypoint).")
    parser.add_argument(
        "--config_name",
        type=str,
        default="default",
        help="Name of the config file to use (without .yaml extension).",
    )
    parser.add_argument(
        "--no_kaggle_defaults",
        action="store_true",
        help="Disable automatic Kaggle-friendly Hydra overrides.",
    )
    args, unknown = parser.parse_known_args()

    overrides = list(unknown)
    if not args.no_kaggle_defaults:
        overrides = _default_kaggle_overrides(overrides) + overrides

    config_dir = (project_root / "training" / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    if _IS_KAGGLE:
        print(f"[launch] Kaggle detected, project_root={project_root}")
        if overrides:
            print(f"[launch] Applied overrides: {overrides}")

    from training.trainer import Trainer

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()

# examples:
#   python launch.py --config_name default
#   python launch.py --config_name default data.data_root=/kaggle/input/my3d
