import argparse
import copy
import logging
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import process_kaggle  # noqa: E402
from lock_utils import file_lock  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _load_yaml(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _now():
    return datetime.now()


def _parse_time(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _format_time(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _owner_from_kernel_id(kernel_id: str):
    if "/" not in kernel_id:
        return None
    return kernel_id.split("/", 1)[0]


def _parse_status(output: str):
    text = (output or "").lower()
    if "running" in text:
        return "running"
    if any(k in text for k in ["complete", "finished", "success"]):
        return "finished"
    if any(k in text for k in ["error", "failed"]):
        return "error"
    return "unknown"


def _kernel_status(kernel_id: str):
    owner_id = _owner_from_kernel_id(kernel_id)
    if not owner_id:
        return "error", "missing owner in kernel id"
    token = process_kaggle._load_token(owner_id)
    if not token:
        return "error", f"missing token for owner {owner_id}"

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = ["kaggle", "kernels", "status", kernel_id]
    result = subprocess.run(cmd, cwd=BASE_DIR, env=env, check=False, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return "error", output.strip()
    return _parse_status(output), output.strip()


def _push_kernel(cfg: dict, dry: bool = False):
    cfg_path = BASE_DIR / "config.yaml"
    _write_yaml(cfg_path, cfg)
    cmd = ["python", str(BASE_DIR / "process_kaggle.py"), "--run", "--concise"]
    if dry:
        logging.info("Dry run: %s", " ".join(cmd))
        return True, ""
    result = subprocess.run(cmd, cwd=BASE_DIR, check=False, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def _update_left_time(node, notebook, quota_hours):
    before = float(node.get("left_time", quota_hours))
    start_time = _parse_time(notebook.get("start_time"))
    if start_time is None:
        return before
    elapsed = (_now() - start_time).total_seconds() / 3600.0
    after = before - elapsed
    node["left_time"] = after
    return after


def _move_to_available_node(nodes, current_node, notebook, available_ids, quota_hours):
    for node in nodes:
        if node is current_node:
            continue
        if float(node.get("left_time", 0)) <= 0:
            continue
        if len(node.get("notebooks") or []) < 2:
            node.setdefault("notebooks", []).append(notebook)
            current_node["notebooks"].remove(notebook)
            return node

    if not available_ids:
        return None

    new_id = available_ids.pop(0)
    target = {"id": new_id, "left_time": quota_hours, "notebooks": [notebook]}
    nodes.append(target)
    current_node["notebooks"].remove(notebook)
    return target


def _build_resumed_cfg(base_cfg: dict, *, target_id: str, resumed_from: str, run_id: int):
    cfg = copy.deepcopy(base_cfg)
    cfg["id"] = target_id
    cfg["run_id"] = run_id
    cfg["kernel_sources"] = [resumed_from]
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Auto-resume n3r Kaggle kernels")
    parser.add_argument("--config", default=str(BASE_DIR / "config_kernel.yaml"))
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--quota-hours", type=float, default=30.0)
    parser.add_argument("--no-lock", action="store_false", dest="lock", default=True)
    args = parser.parse_args()

    verbose = not args.quiet
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (BASE_DIR / config_path).resolve()

    kcfg = _load_yaml(config_path)
    if not kcfg:
        raise ValueError(f"Missing or empty config kernel file: {config_path}")

    sleep_time_hr = float(kcfg.get("sleep_time_hr", 0))
    if sleep_time_hr > 0:
        time.sleep(sleep_time_hr * 3600)

    while True:
        lock_ctx = file_lock(config_path) if args.lock else nullcontext()
        with lock_ctx:
            kcfg = _load_yaml(config_path)
            base_cfg = _load_yaml(BASE_DIR / "config.yaml")
            running_nodes = kcfg.get("running_nodes") or []
            available_ids = list(kcfg.get("available_ids") or [])
            exhausted_ids = list(kcfg.get("exhausted_ids") or [])
            finished_notebooks = list(kcfg.get("finished_notebooks") or [])
            error_notebooks = list(kcfg.get("error_notebooks") or [])
            poll_interval_minutes = float(kcfg.get("poll_interval_minutes", 10))
            changed = False

            for node in list(running_nodes):
                node.setdefault("left_time", args.quota_hours)
                notebooks = node.get("notebooks") or []
                for notebook in list(notebooks):
                    kernel_id = notebook.get("kernel_id")
                    if not kernel_id:
                        continue

                    if args.dry:
                        status, detail = "unknown", ""
                        if verbose:
                            logging.info("Dry run status check: %s", kernel_id)
                    else:
                        status, detail = _kernel_status(kernel_id)

                    if verbose:
                        logging.info("%s -> %s", kernel_id, status)

                    if status == "running":
                        continue
                    if status == "error":
                        rec = dict(notebook)
                        rec["error_detail"] = detail
                        error_notebooks.append(rec)
                        notebooks.remove(notebook)
                        changed = True
                        continue
                    if status not in ("finished", "unknown"):
                        continue

                    old_run_id = int(notebook.get("run_id", 0))
                    total_runs = int(notebook.get("total_runs", 1))
                    next_run_id = old_run_id + 1
                    resumed_from = kernel_id

                    history = list(notebook.get("history_ids") or [])
                    history.append(kernel_id)
                    notebook["history_ids"] = history
                    notebook["resumed_from"] = resumed_from

                    left_time = _update_left_time(node, notebook, args.quota_hours)

                    if next_run_id >= total_runs:
                        fin = dict(notebook)
                        fin["finished_time"] = _format_time(_now())
                        finished_notebooks.append(fin)
                        notebooks.remove(notebook)
                        changed = True
                        continue

                    target_node = node
                    if left_time <= 0:
                        target_node = _move_to_available_node(
                            running_nodes, node, notebook, available_ids, args.quota_hours
                        )
                        if target_node is None:
                            logging.warning("No available ids to resume %s", kernel_id)
                            continue
                        changed = True

                    cfg = _build_resumed_cfg(
                        base_cfg,
                        target_id=target_node["id"],
                        resumed_from=resumed_from,
                        run_id=next_run_id,
                    )
                    new_kernel_id = process_kaggle._build_kernel_id(cfg)
                    notebook["run_id"] = next_run_id
                    notebook["kernel_id"] = new_kernel_id
                    notebook["start_time"] = _format_time(_now())

                    ok, output = _push_kernel(cfg, dry=args.dry)
                    if not ok:
                        logging.error("Push failed for %s: %s", new_kernel_id, output)
                        rec = dict(notebook)
                        rec["error_detail"] = output
                        error_notebooks.append(rec)
                        if notebook in (target_node.get("notebooks") or []):
                            target_node["notebooks"].remove(notebook)
                        changed = True
                        continue
                    changed = True

                node["notebooks"] = notebooks

            for node in list(running_nodes):
                if not (node.get("notebooks") or []) and float(node.get("left_time", 0)) <= 0:
                    running_nodes.remove(node)
                    nid = node.get("id")
                    if nid and nid not in exhausted_ids:
                        exhausted_ids.append(nid)
                    changed = True

            kcfg["running_nodes"] = running_nodes
            kcfg["available_ids"] = available_ids
            kcfg["exhausted_ids"] = exhausted_ids
            kcfg["finished_notebooks"] = finished_notebooks
            kcfg["error_notebooks"] = error_notebooks
            if changed:
                _write_yaml(config_path, kcfg)

        logging.info("sleeping for %.1f minutes...", poll_interval_minutes)
        time.sleep(poll_interval_minutes * 60)


if __name__ == "__main__":
    main()
