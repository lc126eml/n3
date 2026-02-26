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


USAGE = """\
Expected config_kernel.yaml (GPU) fields:
- sleep_time_hr: float
- poll_interval_minutes: float
- available_ids: [owner_id, ...]
- exhausted_ids: [owner_id, ...]
- running_nodes:
  - id: owner_id
    left_time: float
    notebooks:
      - kernel_id: owner/slug
        total_runs: int
        run_id: int
        start_time: ISO8601 string or unix timestamp
        resumed_from: kernel_id or null
        history_ids: [kernel_id, ...]
- finished_notebooks: [notebook, ...]
- error_notebooks: [notebook, ...]  # legacy / currently not written back

config_kernel_tpu.yaml uses the same field names as config_kernel.yaml.
"""


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


def _now_naive():
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
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _owner_from_kernel_id(kernel_id: str):
    if "/" not in kernel_id:
        return None
    return kernel_id.split("/", 1)[0]


def _parse_status(output: str):
    text = (output or "").lower()
    if "running" in text:
        return "running"
    if "complete" in text or "finished" in text or "success" in text:
        return "finished"
    if "error" in text or "failed" in text:
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


def _push_kernel(cfg: dict):
    cfg_path = BASE_DIR / "config.yaml"
    original_text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else None
    try:
        _write_yaml(cfg_path, cfg)
        result = subprocess.run(
            [
                sys.executable,
                str(BASE_DIR / "process_kaggle.py"),
                "--run",
                "--concise",
                "--push-output-only",
            ],
            cwd=BASE_DIR,
            check=False,
            capture_output=True,
            text=True,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, output.strip()
    finally:
        try:
            if original_text is None:
                cfg_path.unlink(missing_ok=True)
            else:
                cfg_path.write_text(original_text, encoding="utf-8")
        except Exception as exc:
            logging.warning("Failed to restore %s after push: %s", cfg_path, exc)


def _move_finished(notebook, finished_notebooks):
    finished = dict(notebook)
    for key in ("run_id", "resumed_from"):
        finished.pop(key, None)
    finished_notebooks.append(finished)


def _quota_hours(is_tpu: bool, override: float | None):
    if override is not None:
        return float(override)
    return 20.0 if is_tpu else 30.0


def _update_left_time(node, notebook, now, is_tpu: bool, quota_hours_override: float | None, verbose: bool):
    start_time = _parse_time(notebook.get("start_time"))
    before_left = float(node.get("left_time", _quota_hours(is_tpu, quota_hours_override)))
    if start_time is None:
        if verbose:
            logging.warning(
                "Missing start_time for %s; skipping left_time update",
                notebook.get("kernel_id"),
            )
        return before_left, before_left, 0.0
    elapsed_hr = (now - start_time).total_seconds() / 3600.0
    left_time = before_left - elapsed_hr
    node["left_time"] = left_time
    if verbose:
        logging.info(
            "Updated left_time for %s: %.2f -> %.2f (elapsed %.2f h)",
            node.get("id"),
            before_left,
            left_time,
            elapsed_hr,
        )
    return before_left, left_time, elapsed_hr


def _move_to_new_node(node, notebook, running_nodes, available_ids, exhausted_ids, is_tpu: bool, quota_hours_override: float | None):
    node_notebooks = node.get("notebooks") or []
    notebook_limit = 1 if is_tpu else 2
    move_now = _now_naive()
    default_quota = _quota_hours(is_tpu, quota_hours_override)
    for candidate in running_nodes:
        if candidate is node:
            continue
        if process_kaggle._effective_left_time_for_selection(
            candidate.get("left_time"),
            candidate.get("notebooks"),
            default_left_time=default_quota,
            now=move_now,
        ) <= 0:
            continue
        candidate_notebooks = candidate.get("notebooks") or []
        if len(candidate_notebooks) < notebook_limit:
            candidate_notebooks.append(notebook)
            candidate["notebooks"] = candidate_notebooks
            if notebook in node_notebooks:
                node_notebooks.remove(notebook)
                node["notebooks"] = node_notebooks
            return candidate

    if not available_ids:
        logging.warning("No available ids left to resume.")
        return None
    if isinstance(available_ids, set):
        new_id = available_ids.pop()
    else:
        new_id = available_ids.pop(0)
    target_node = {"id": new_id, "notebooks": []}
    target_node["left_time"] = _quota_hours(is_tpu, quota_hours_override)
    target_node["notebooks"].append(notebook)
    running_nodes.append(target_node)
    if notebook in node_notebooks:
        node_notebooks.remove(notebook)
        node["notebooks"] = node_notebooks
    return target_node


def _remove_simple_override(cfg: dict, key: str) -> None:
    simple = cfg.get("simple")
    if simple is None:
        return
    if isinstance(simple, dict):
        simple.pop(key, None)
        return
    if isinstance(simple, list):
        cfg["simple"] = [item for item in simple if not (isinstance(item, dict) and key in item)]


def _build_resumed_cfg(
    base_cfg: dict,
    *,
    target_id: str,
    resumed_from: str,
    run_id: int,
    is_tpu: bool,
):
    cfg = copy.deepcopy(base_cfg)
    cfg["id"] = target_id
    cfg["run_id"] = run_id
    cfg.pop("slug", None)
    _remove_simple_override(cfg, "checkpoint.resume_checkpoint_path")
    resume_checkpoint_path = process_kaggle._kernel_checkpoint_path(resumed_from)
    if resume_checkpoint_path is not None:
        cfg["resume_checkpoint_path"] = resume_checkpoint_path
    else:
        cfg.pop("resume_checkpoint_path", None)
    cfg["kernel_sources"] = [resumed_from]
    cfg["resume_full_ckpt"] = True
    cfg["resume_source"] = "kernel"
    cfg["resume_infer"] = True
    cfg["enable_tpu"] = bool(is_tpu)
    cfg["enable_gpu"] = not bool(is_tpu)
    # Keep auto_resume's predicted kernel_id aligned with process_kaggle.py.
    process_kaggle._apply_resume_infer(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Auto resume Kaggle kernels.", epilog=USAGE)
    parser.add_argument(
        "--config",
        default=None,
        help="Override config path (defaults to config_kernel.yaml for GPU or config_kernel_tpu.yaml for TPU).",
    )
    parser.add_argument("--dry", action="store_true", help="Do not run kaggle commands; only report them.")
    parser.add_argument("--quiet", action="store_true", help="Suppress poll status details.")
    parser.add_argument(
        "--quota-hours",
        type=float,
        default=None,
        help="Override per-node quota hours (default: 30 GPU / 20 TPU).",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--tpu", action="store_true", help="Process TPU running nodes.")
    mode_group.add_argument("--gpu", action="store_true", help="Process GPU running nodes.")
    parser.add_argument("--no-lock", action="store_false", dest="lock", default=True, help="Disable config locking.")
    args = parser.parse_args()
    verbose = not args.quiet

    is_tpu = bool(args.tpu)
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (BASE_DIR / config_path).resolve()
    else:
        config_path = BASE_DIR / ("config_kernel_tpu.yaml" if is_tpu else "config_kernel.yaml")

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
            poll_interval_minutes = float(kcfg.get("poll_interval_minutes", 10))

            changed = False
            running_nodes = kcfg.get("running_nodes") or []
            available_ids = set(kcfg.get("available_ids") or [])
            exhausted_ids = kcfg.get("exhausted_ids") or []
            finished_notebooks = kcfg.get("finished_notebooks") or []
            error_notebooks = kcfg.get("error_notebooks") or []

            def _process_nodes(nodes, available, exhausted):
                nonlocal changed
                for node in list(nodes):
                    notebooks = node.get("notebooks") or []
                    if "left_time" not in node:
                        node["left_time"] = _quota_hours(is_tpu, args.quota_hours)
                    for notebook in list(notebooks):
                        kernel_id = notebook.get("kernel_id")
                        if not kernel_id:
                            continue

                        if args.dry:
                            if verbose:
                                logging.info("Dry run: kaggle kernels status %s", kernel_id)
                            status, detail = "unknown", ""
                        else:
                            status, detail = _kernel_status(kernel_id)

                        if verbose:
                            logging.info("%s: %s", kernel_id, status)

                        if status == "running":
                            continue
                        if status == "error":
                            logging.error(detail)
                            continue
                        if status != "finished":
                            continue

                        try:
                            slug = str(kernel_id).split("/", 1)[1].strip()
                            _, old_run_id, _ = process_kaggle._infer_from_source_id(slug)
                        except Exception:
                            old_run_id = int(notebook.get("run_id", 0))
                            if verbose:
                                logging.warning(
                                    "Could not infer run_id from kernel_id '%s'; falling back to notebook.run_id=%s",
                                    kernel_id,
                                    old_run_id,
                                )
                        total_runs = int(notebook.get("total_runs", 1))
                        next_run_id = old_run_id + 1

                        prior_resumed_from = notebook.get("resumed_from")
                        history_ids = list(notebook.get("history_ids") or [])
                        if prior_resumed_from:
                            history_ids.append(prior_resumed_from)
                        notebook["history_ids"] = history_ids
                        notebook["resumed_from"] = kernel_id
                        notebook["run_id"] = next_run_id

                        _, left_time, _ = _update_left_time(
                            node,
                            notebook,
                            _now_naive(),
                            is_tpu=is_tpu,
                            quota_hours_override=args.quota_hours,
                            verbose=verbose,
                        )

                        if next_run_id >= total_runs:
                            fin = dict(notebook)
                            fin["finished_time"] = _format_time(_now_naive())
                            finished_notebooks.append(fin)
                            notebooks.remove(notebook)
                            changed = True
                            continue

                        target_node = node
                        # left_time already includes elapsed time of the just-finished notebook.
                        # Only subtract elapsed time of other notebooks still running on this node.
                        other_notebooks = [nb for nb in (node.get("notebooks") or []) if nb is not notebook]
                        effective_left_time = process_kaggle._effective_left_time_for_selection(
                            left_time,
                            other_notebooks,
                            default_left_time=_quota_hours(is_tpu, args.quota_hours),
                            now=_now_naive(),
                        )
                        if effective_left_time <= 0:
                            target_node = _move_to_new_node(
                                node,
                                notebook,
                                nodes,
                                available,
                                exhausted,
                                is_tpu=is_tpu,
                                quota_hours_override=args.quota_hours,
                            )
                            if target_node is None:
                                continue
                            if verbose:
                                logging.info(
                                    "Moved notebook to node: %s -> %s",
                                    node.get("id"),
                                    target_node.get("id"),
                                )

                        resumed_from_id = notebook.get("resumed_from")
                        if not resumed_from_id:
                            logging.warning("Missing resumed_from for notebook.")
                            continue

                        cfg = _build_resumed_cfg(
                            base_cfg,
                            target_id=target_node["id"],
                            resumed_from=resumed_from_id,
                            run_id=next_run_id,
                            is_tpu=is_tpu,
                        )
                        new_kernel_id = process_kaggle._build_kernel_id(cfg)
                        notebook["kernel_id"] = new_kernel_id
                        notebook["start_time"] = _format_time(_now_naive())

                        if verbose:
                            logging.info("Submitting kernel: %s", new_kernel_id)
                        if args.dry:
                            logging.info(
                                "Dry run: python %s --run --concise --push-output-only",
                                str(BASE_DIR / "process_kaggle.py"),
                            )
                            ok, output = True, ""
                        else:
                            ok, output = _push_kernel(cfg)

                        quota_msg = "Maximum weekly GPU quota"
                        if quota_msg in output:
                            node["left_time"] = -10
                            history_ids = list(notebook.get("history_ids") or [])
                            history_ids.append(new_kernel_id)
                            notebook["history_ids"] = history_ids
                            target_node = _move_to_new_node(
                                target_node,
                                notebook,
                                nodes,
                                available,
                                exhausted,
                                is_tpu=is_tpu,
                                quota_hours_override=args.quota_hours,
                            )
                            if target_node is None:
                                continue
                            if verbose:
                                logging.info(
                                    "Quota reached. Moved notebook to node: %s",
                                    target_node.get("id"),
                                )
                            cfg["id"] = target_node.get("id")
                            cfg["enable_tpu"] = bool(is_tpu)
                            cfg["enable_gpu"] = not bool(is_tpu)
                            new_kernel_id = process_kaggle._build_kernel_id(cfg)
                            notebook["kernel_id"] = new_kernel_id
                            notebook["start_time"] = _format_time(_now_naive())
                            if verbose:
                                logging.info("Submitting kernel: %s", new_kernel_id)
                            if not args.dry:
                                ok, output = _push_kernel(cfg)

                        if not ok and verbose:
                            logging.error("Push failed for %s: %s", notebook.get("kernel_id"), output)
                        changed = True

                    node["notebooks"] = notebooks

            def _remove_exhausted():
                nonlocal changed
                for node in list(running_nodes):
                    node_notebooks = node.get("notebooks") or []
                    if not node_notebooks and float(node.get("left_time", 0)) <= 0:
                        running_nodes.remove(node)
                        exhausted_ids.append(node.get("id"))
                        changed = True

            try:
                _process_nodes(running_nodes, available_ids, exhausted_ids)
                _remove_exhausted()
            except Exception as exc:
                logging.error(exc)

            kcfg["running_nodes"] = running_nodes
            kcfg["available_ids"] = sorted(available_ids)
            kcfg["exhausted_ids"] = exhausted_ids
            kcfg["finished_notebooks"] = finished_notebooks
            # Mirror pos behavior: keep legacy field loaded but do not update/write it.
            _ = error_notebooks
            # kcfg["error_notebooks"] = error_notebooks
            if changed:
                _write_yaml(config_path, kcfg)

            del kcfg

        logging.info("sleeping for %.1f minutes...", poll_interval_minutes)
        time.sleep(poll_interval_minutes * 60)


if __name__ == "__main__":
    main()
