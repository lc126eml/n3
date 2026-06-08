import argparse
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "utils"))

import auto_resume  # noqa: E402
import process_kaggle  # noqa: E402
from lock_utils import file_lock  # noqa: E402
from supabase_utils import fetch_new_logs  # noqa: E402


USAGE = """\
Poll Supabase training_logs and resume successful Kaggle kernels.

Expected log fields:
- id: monotonically increasing Supabase row id
- task: kernel_id, for example owner/slug
- status: false for running, true for finished
- message: contains SUCCESS for successful finished kernels

The script stores last_supabase_id in config_kernel.yaml / config_kernel_tpu.yaml.
"""


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _log_id(log: dict):
    try:
        return int(log.get("id"))
    except (TypeError, ValueError):
        return None


def _log_kernel_id(log: dict) -> str:
    return str(log.get("task") or log.get("kernel_id") or "").strip()


def _is_success_log(log: dict) -> bool:
    return log.get("status") is True and "SUCCESS" in str(log.get("message") or "")


def _find_running_notebook(running_nodes, kernel_id: str):
    for node in running_nodes:
        notebooks = node.get("notebooks") or []
        for notebook in notebooks:
            if notebook.get("kernel_id") == kernel_id:
                return node, notebooks, notebook
    return None, None, None


def _remove_exhausted_nodes(running_nodes, exhausted_ids):
    changed = False
    for node in list(running_nodes):
        node_notebooks = node.get("notebooks") or []
        if not node_notebooks and float(node.get("left_time", 0)) <= 0:
            running_nodes.remove(node)
            exhausted_ids.append(node.get("id"))
            changed = True
    return changed


def _process_success_log(
    *,
    log: dict,
    running_nodes,
    available_ids,
    exhausted_ids,
    finished_notebooks,
    base_cfg,
    is_tpu: bool,
    quota_hours,
    dry: bool,
    verbose: bool,
):
    kernel_id = _log_kernel_id(log)
    if not kernel_id:
        return False

    node, notebooks, notebook = _find_running_notebook(running_nodes, kernel_id)
    if notebook is None:
        if verbose:
            logging.info("Skipping successful log for non-running kernel: %s", kernel_id)
        return False

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

    _, left_time, _ = auto_resume._update_left_time(
        node,
        notebook,
        auto_resume._now_naive(),
        is_tpu=is_tpu,
        quota_hours_override=quota_hours,
        verbose=verbose,
    )

    if next_run_id >= total_runs:
        finished = dict(notebook)
        finished["finished_time"] = auto_resume._format_time(auto_resume._now_naive())
        finished_notebooks.append(finished)
        notebooks.remove(notebook)
        return True

    target_node = node
    other_notebooks = [nb for nb in (node.get("notebooks") or []) if nb is not notebook]
    effective_left_time = process_kaggle._effective_left_time_for_selection(
        left_time,
        other_notebooks,
        default_left_time=auto_resume._quota_hours(is_tpu, quota_hours),
        now=auto_resume._now_naive(),
    )
    if effective_left_time <= 0:
        target_node = auto_resume._move_to_new_node(
            node,
            notebook,
            running_nodes,
            available_ids,
            exhausted_ids,
            is_tpu=is_tpu,
            quota_hours_override=quota_hours,
        )
        if target_node is None:
            return True
        if verbose:
            logging.info("Moved notebook to node: %s -> %s", node.get("id"), target_node.get("id"))

    resumed_from_id = notebook.get("resumed_from")
    if not resumed_from_id:
        logging.warning("Missing resumed_from for notebook.")
        return True

    git_branch = notebook.get("git_branch") or auto_resume._DEFAULT_RESUME_GIT_BRANCH
    notebook["git_branch"] = git_branch
    cfg = auto_resume._build_resumed_cfg(
        base_cfg,
        target_id=target_node["id"],
        resumed_from=resumed_from_id,
        run_id=next_run_id,
        is_tpu=is_tpu,
        git_branch=git_branch,
    )
    new_kernel_id = process_kaggle._build_kernel_id(cfg)
    notebook["kernel_id"] = new_kernel_id
    notebook["start_time"] = auto_resume._format_time(auto_resume._now_naive())

    if verbose:
        logging.info("Submitting kernel: %s", new_kernel_id)
    if dry:
        logging.info("Dry run: python %s --run --concise --push-output-only", str(BASE_DIR / "process_kaggle.py"))
        ok, output = True, ""
    else:
        ok, output = auto_resume._push_kernel(cfg)
        logging.info(output)

    quota_msg = "Maximum weekly GPU quota"
    if quota_msg in output:
        node["left_time"] = -10
        history_ids = list(notebook.get("history_ids") or [])
        history_ids.append(new_kernel_id)
        notebook["history_ids"] = history_ids
        target_node = auto_resume._move_to_new_node(
            target_node,
            notebook,
            running_nodes,
            available_ids,
            exhausted_ids,
            is_tpu=is_tpu,
            quota_hours_override=quota_hours,
        )
        if target_node is None:
            return True
        if verbose:
            logging.info("Quota reached. Moved notebook to node: %s", target_node.get("id"))
        cfg["id"] = target_node.get("id")
        cfg["enable_tpu"] = bool(is_tpu)
        cfg["enable_gpu"] = not bool(is_tpu)
        new_kernel_id = process_kaggle._build_kernel_id(cfg)
        notebook["kernel_id"] = new_kernel_id
        notebook["start_time"] = auto_resume._format_time(auto_resume._now_naive())
        if verbose:
            logging.info("Submitting kernel: %s", new_kernel_id)
        if not dry:
            ok, output = auto_resume._push_kernel(cfg)

    if not ok and verbose:
        logging.error("Push failed for %s: %s", notebook.get("kernel_id"), output)
    return True


def main():
    parser = argparse.ArgumentParser(description="Auto resume Kaggle kernels from Supabase logs.", epilog=USAGE)
    parser.add_argument(
        "--config",
        default=None,
        help="Override config path (defaults to config_kernel.yaml for GPU or config_kernel_tpu.yaml for TPU).",
    )
    parser.add_argument("--dry", action="store_true", help="Do not push kernels; only report actions.")
    parser.add_argument("--quiet", action="store_true", help="Suppress poll details.")
    parser.add_argument(
        "--quota-hours",
        type=float,
        default=None,
        help="Override per-node quota hours (default: 30 GPU / 20 TPU).",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--tpu", dest="is_tpu", action="store_true", help="Process TPU running nodes.")
    mode_group.add_argument("--gpu", dest="is_tpu", action="store_false", help="Process GPU running nodes.")
    parser.set_defaults(is_tpu=False)
    parser.add_argument("--no-lock", action="store_false", dest="lock", default=True, help="Disable config locking.")
    args = parser.parse_args()
    verbose = not args.quiet

    is_tpu = bool(args.is_tpu)
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (BASE_DIR / config_path).resolve()
    else:
        config_path = BASE_DIR / ("config_kernel_tpu.yaml" if is_tpu else "config_kernel.yaml")

    kcfg = auto_resume._load_yaml(config_path)
    if not kcfg:
        raise ValueError(f"Missing or empty config kernel file: {config_path}")

    sleep_time_hr = float(kcfg.get("sleep_time_hr", 0))
    if sleep_time_hr > 0:
        time.sleep(sleep_time_hr * 3600)

    while True:
        lock_ctx = file_lock(config_path) if args.lock else nullcontext()
        with lock_ctx:
            kcfg = auto_resume._load_yaml(config_path)
            base_cfg = auto_resume._load_yaml(BASE_DIR / "config.yaml")
            poll_interval_minutes = float(kcfg.get("poll_interval_minutes", 10))
            last_supabase_id = int(kcfg.get("last_supabase_id") or 0)

            logs = fetch_new_logs(last_id=last_supabase_id) or []
            if verbose:
                logging.info("Fetched %d Supabase logs after id %s", len(logs), last_supabase_id)

            changed = False
            running_nodes = kcfg.get("running_nodes") or []
            available_ids = set(kcfg.get("available_ids") or [])
            exhausted_ids = kcfg.get("exhausted_ids") or []
            finished_notebooks = kcfg.get("finished_notebooks") or []
            error_notebooks = kcfg.get("error_notebooks") or []
            max_seen_id = last_supabase_id

            for log in logs:
                log_id = _log_id(log)
                if log_id is not None:
                    max_seen_id = max(max_seen_id, log_id)

                if not _is_success_log(log):
                    if verbose:
                        logging.info(
                            "Skipping Supabase log id=%s kernel=%s status=%r message=%r",
                            log.get("id"),
                            _log_kernel_id(log),
                            log.get("status"),
                            log.get("message"),
                        )
                    continue
                elif verbose:
                    logging.info(
                        "Finished! id=%s kernel=%s status=%r message=%r",
                        log.get("id"),
                        _log_kernel_id(log),
                        log.get("status"),
                        log.get("message"),
                    )

                changed = (
                    _process_success_log(
                        log=log,
                        running_nodes=running_nodes,
                        available_ids=available_ids,
                        exhausted_ids=exhausted_ids,
                        finished_notebooks=finished_notebooks,
                        base_cfg=base_cfg,
                        is_tpu=is_tpu,
                        quota_hours=args.quota_hours,
                        dry=args.dry,
                        verbose=verbose,
                    )
                    or changed
                )

            if max_seen_id != last_supabase_id:
                kcfg["last_supabase_id"] = max_seen_id
                changed = True

            if _remove_exhausted_nodes(running_nodes, exhausted_ids):
                changed = True

            kcfg["running_nodes"] = running_nodes
            kcfg["available_ids"] = sorted(available_ids)
            kcfg["exhausted_ids"] = exhausted_ids
            kcfg["finished_notebooks"] = finished_notebooks
            _ = error_notebooks
            if changed and not args.dry:
                auto_resume._write_yaml(config_path, kcfg)

            del kcfg

        logging.info("sleeping for %.1f minutes...", poll_interval_minutes)
        time.sleep(poll_interval_minutes * 60)


if __name__ == "__main__":
    main()
