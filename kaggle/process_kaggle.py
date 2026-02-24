import argparse
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import yaml
from lock_utils import file_lock


BASE_DIR = Path(__file__).resolve().parent


def _load_yaml(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _unique_extend(target, values):
    for item in values:
        if item not in target:
            target.append(item)


def _bool_str(v: bool) -> str:
    return "true" if bool(v) else "false"


def _load_token(owner_id: str):
    tokens = _load_yaml(BASE_DIR / "tokens.yaml")
    if isinstance(tokens, dict):
        if owner_id in tokens:
            return tokens[owner_id]
        for item in _as_list(tokens.get("tokens")):
            if isinstance(item, dict) and owner_id in item:
                return item[owner_id]
    if isinstance(tokens, list):
        for item in tokens:
            if isinstance(item, dict) and owner_id in item:
                return item[owner_id]
    return None


def _slugify(text: str) -> str:
    out = []
    prev_dash = False
    for ch in str(text).lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "n3r"


def _derived_slug(cfg: dict) -> str:
    parts = ["n3r", str(cfg.get("config_file") or "default")]
    if cfg.get("desc") not in (None, ""):
        parts.append(str(cfg["desc"]))
    if cfg.get("run_id") not in (None, ""):
        parts.append(f"r{cfg['run_id']}")
    if cfg.get("seed") not in (None, ""):
        parts.append(str(cfg["seed"]))
    return _slugify("-".join(parts))


def _build_kernel_id(cfg: dict) -> str:
    owner = cfg["id"]
    slug = str(cfg.get("slug") or _derived_slug(cfg)).strip()
    if not slug:
        raise ValueError("config.yaml requires non-empty 'slug'")
    return f"{owner}/{slug}"


def _update_metadata(cfg: dict) -> dict:
    meta_path = BASE_DIR / "kernel-metadata.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    meta["id"] = _build_kernel_id(cfg)
    meta["title"] = str(cfg.get("title") or _build_kernel_id(cfg).split("/", 1)[1].replace("-", " "))
    meta["code_file"] = "../training/launch.py"
    meta["language"] = "python"
    meta["kernel_type"] = "script"
    meta["is_private"] = _bool_str(cfg.get("is_private", True))
    meta["enable_gpu"] = _bool_str(cfg.get("enable_gpu", True))
    meta["enable_tpu"] = _bool_str(cfg.get("enable_tpu", False))
    meta["enable_internet"] = _bool_str(cfg.get("enable_internet", True))
    dataset_sources = []
    _unique_extend(
        dataset_sources,
        [
            "liucong12601/timm-repos",
            "liucong12601/hsm-train-part01",
            "liucong12601/hsm-train-part02",
            "liucong12601/hsm-train-part03",
            "liucong12601/hsm-train-part04",
            "liucong12601/hsm-train-part05",
            "liucong12601/hsm-test-val",
            "liucong12601/ds-file-list",
            "sinayliu/temp-db",
            "sinayliu/scnt-p02",
            "sinayliu/scnt-p03",
            "sinayliu/scnt-p04",
            "sinayliu/scnt-p05",
            "sinayliu/scnt-p06",
            "sinayliu/scnt-p07",
            "sinayliu/scnt-p08",
            "sinayliu/scnt-p09",
            "sinayliu/scnt-p10",
            "sinayliu/scnt-p11",
            "sinayliu/scnt-p12",
        ],
    )
    _unique_extend(dataset_sources, [str(x) for x in _as_list(cfg.get("dataset_sources"))])
    meta["dataset_sources"] = dataset_sources
    meta["competition_sources"] = [str(x) for x in _as_list(cfg.get("competition_sources"))]
    meta["kernel_sources"] = [str(x) for x in _as_list(cfg.get("kernel_sources"))]
    meta["model_sources"] = [str(x) for x in _as_list(cfg.get("model_sources"))]

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _resolve_resume_source(cfg: dict):
    sources = _as_list(cfg.get("kernel_sources"))
    return str(sources[0]) if sources else None


def _iter_simple_updates(cfg: dict):
    simple = cfg.get("simple")
    if simple is None:
        return []
    if isinstance(simple, dict):
        items = []
        for k, v in simple.items():
            items.append((str(k), v))
        return items
    updates = []
    for item in _as_list(simple):
        if not isinstance(item, dict):
            continue
        for k, v in item.items():
            updates.append((str(k), v))
    return updates


def _set_by_dot_path(data, dot_path: str, value) -> None:
    parts = [p for p in str(dot_path).split(".") if p]
    if not parts:
        raise ValueError(f"Invalid simple key: {dot_path!r}")

    cur = data
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        if isinstance(cur, list):
            if not part.isdigit():
                raise ValueError(f"Expected list index at '{'.'.join(parts[:i+1])}'")
            idx = int(part)
            if idx < 0 or idx >= len(cur):
                raise ValueError(f"List index out of range at '{'.'.join(parts[:i+1])}'")
            if is_last:
                cur[idx] = value
            else:
                cur = cur[idx]
            continue

        if not isinstance(cur, dict):
            raise ValueError(f"Cannot descend into non-container at '{'.'.join(parts[:i])}'")
        if part not in cur:
            raise ValueError(f"Key not found in default.yaml: '{'.'.join(parts[:i+1])}'")
        if is_last:
            cur[part] = value
        else:
            cur = cur[part]


def _apply_simple_to_default_yaml(cfg: dict) -> tuple[Path, list[tuple[str, object]]] | None:
    updates = _iter_simple_updates(cfg)
    if not updates:
        return None
    yaml_path = (BASE_DIR.parent / "training" / "configs" / "default.yaml").resolve()
    data = _load_yaml(yaml_path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Failed to load default.yaml: {yaml_path}")
    for key, value in updates:
        _set_by_dot_path(data, key, value)
    _write_yaml(yaml_path, data)
    return yaml_path, updates


def _add_running_node(
    cfg: dict,
    kernel_id: str,
    total_runs: int,
    config_kernel_path: Path,
    use_lock: bool = True,
    consume_available: bool = False,
) -> None:
    lock_ctx = file_lock(str(config_kernel_path) + ".lock", timeout_sec=600, poll_interval=3.0) if use_lock else nullcontext()
    with lock_ctx:
        kcfg = _load_yaml(config_kernel_path)
        running_nodes = kcfg.get("running_nodes") or []
        available_ids = kcfg.get("available_ids") or []
        owner_id = str(cfg["id"])

        if consume_available and owner_id in available_ids:
            available_ids.remove(owner_id)
            kcfg["available_ids"] = available_ids

        node = None
        for item in running_nodes:
            if item.get("id") == owner_id:
                node = item
                break
        if node is None:
            node = {"id": owner_id, "left_time": float(kcfg.get("default_quota_hours", 30)), "notebooks": []}
            running_nodes.append(node)

        notebooks = node.get("notebooks") or []
        if any(nb.get("kernel_id") == kernel_id for nb in notebooks):
            return

        notebook = {
            "kernel_id": kernel_id,
            "run_id": int(cfg.get("run_id") or 0),
            "total_runs": int(total_runs),
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "resumed_from": _resolve_resume_source(cfg),
            "history_ids": [],
        }
        notebooks.append(notebook)
        node["notebooks"] = notebooks
        kcfg["running_nodes"] = running_nodes
        _write_yaml(config_kernel_path, kcfg)


def _choose_id_for_add_node(config_kernel_path: Path, use_lock: bool = True) -> tuple[str, bool]:
    lock_ctx = file_lock(str(config_kernel_path) + ".lock", timeout_sec=600, poll_interval=3.0) if use_lock else nullcontext()
    with lock_ctx:
        kcfg = _load_yaml(config_kernel_path)
        running_nodes = kcfg.get("running_nodes") or []
        for node in running_nodes:
            if float(node.get("left_time", 0) or 0) <= 0:
                continue
            notebooks = node.get("notebooks") or []
            if len(notebooks) >= 2:
                continue
            node_id = node.get("id")
            if node_id:
                return str(node_id), False

        available_ids = kcfg.get("available_ids") or []
        if not available_ids:
            raise ValueError("No available_ids left to assign.")
        return str(available_ids[0]), True


def main():
    parser = argparse.ArgumentParser(description="Prepare/push n3r Kaggle kernel")
    parser.add_argument("--cfg", default=str(BASE_DIR / "config.yaml"))
    parser.add_argument("--run", action="store_true", help="Run kaggle kernels push -p .")
    parser.add_argument("--concise", action="store_true", help="Minimal output")
    parser.add_argument("--push-output-only", action="store_true", help="Only output kaggle kernels push stdout/stderr.")
    parser.add_argument("--add-node", "--add-running-node", action="store_true", dest="add_running_node")
    parser.add_argument("--total-runs", type=int, default=None)
    parser.add_argument("--config-kernel", default=str(BASE_DIR / "config_kernel.yaml"))
    parser.add_argument("--no-lock", action="store_false", dest="lock", default=True, help="Disable config_kernel.yaml locking.")
    args = parser.parse_args()

    cfg_path = Path(args.cfg).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (BASE_DIR / cfg_path).resolve()
    cfg = _load_yaml(cfg_path)
    if not cfg:
        raise ValueError(f"Empty or missing config: {cfg_path}")
    config_kernel_path = Path(args.config_kernel).expanduser()
    if not config_kernel_path.is_absolute():
        config_kernel_path = (BASE_DIR / config_kernel_path).resolve()

    selected_from_available = False
    if args.add_running_node and not cfg.get("id"):
        chosen_id, selected_from_available = _choose_id_for_add_node(config_kernel_path, use_lock=args.lock)
        cfg["id"] = chosen_id
    if "id" not in cfg or cfg.get("id") in (None, ""):
        raise ValueError("config.yaml requires 'id' (Kaggle owner id), or use --add-running-node with config_kernel.yaml.available_ids")

    simple_result = _apply_simple_to_default_yaml(cfg)
    meta = _update_metadata(cfg)

    if args.concise:
        print(f"kernel_id: {meta['id']}")
    else:
        print(f"Updated {BASE_DIR / 'kernel-metadata.json'}")
        if simple_result is not None:
            yaml_path, updates = simple_result
            print(f"Updated {yaml_path}")
            print(f"simple updates: {[k for k, _ in updates]}")
        print(f"Kernel id: {meta['id']}")
        print(f"Title: {meta['title']}")
        print(f"dataset_sources: {meta.get('dataset_sources', [])}")
        print(f"kernel_sources: {meta.get('kernel_sources', [])}")

    if not args.run:
        return

    owner_id = str(cfg["id"])
    token = _load_token(owner_id)
    if not token:
        raise ValueError(f"No Kaggle API token found for owner id '{owner_id}' in {BASE_DIR / 'tokens.yaml'}")

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = ["kaggle", "kernels", "push", "-p", "."]

    if args.push_output_only:
        result = subprocess.run(cmd, cwd=BASE_DIR, env=env, capture_output=True, text=True)
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    else:
        subprocess.check_call(cmd, cwd=BASE_DIR, env=env)
    if args.add_running_node:
        total_runs = args.total_runs if args.total_runs is not None else int(cfg.get("total_runs") or 1)
        _add_running_node(
            cfg,
            meta["id"],
            total_runs,
            config_kernel_path,
            use_lock=args.lock,
            consume_available=selected_from_available,
        )


if __name__ == "__main__":
    main()
