import argparse
import copy
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import yaml
from ruamel.yaml import YAML as _RuamelYAML
from lock_utils import file_lock


BASE_DIR = Path(__file__).resolve().parent
_ROUNDTRIP_YAML = _RuamelYAML()
_ROUNDTRIP_YAML.preserve_quotes = True
_ROUNDTRIP_YAML.width = 4096


def _load_yaml(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


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


def _effective_left_time_for_selection(left_time, notebooks, default_left_time=30.0, now=None):
    if now is None:
        now = datetime.now()
    if left_time is None:
        left_time = default_left_time
    left_time = float(left_time or 0)
    for notebook in notebooks or []:
        start_time = _parse_time(notebook.get("start_time"))
        if start_time is None:
            continue
        elapsed_hr = (now - start_time).total_seconds() / 3600.0
        if elapsed_hr > 0:
            left_time -= elapsed_hr
    return left_time


def _load_yaml_roundtrip(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return _ROUNDTRIP_YAML.load(f) or {}


def _write_yaml_roundtrip(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        _ROUNDTRIP_YAML.dump(data, f)


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
    parts = []
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
    enable_tpu = bool(cfg.get("enable_tpu", False))
    enable_gpu_cfg = cfg.get("enable_gpu")
    enable_gpu = bool(enable_gpu_cfg) if enable_gpu_cfg is not None else (not enable_tpu)
    meta["is_private"] = _bool_str(cfg.get("is_private", True))
    meta["enable_gpu"] = _bool_str(enable_gpu)
    meta["enable_tpu"] = _bool_str(enable_tpu)
    meta["enable_internet"] = _bool_str(cfg.get("enable_internet", True))
    dataset_sources = []
    _unique_extend(
        dataset_sources,
        [
            "liucong12601/timm-repos",
            "sinayliu/dino-ds",
            "liucong12601/hsm-train-part01",
            "liucong12601/hsm-train-part02",
            "liucong12601/hsm-train-part03",
            "liucong12601/hsm-train-part04",
            "liucong12601/hsm-train-part05",
            "liucong12601/hsm-test-val",
            "liucong12601/ds-file-list",
            # "sinayliu/temp-db",
            # "sinayliu/scnt-p02",
            # "sinayliu/scnt-p03",
            # "sinayliu/scnt-p04",
            # "sinayliu/scnt-p05",
            # "sinayliu/scnt-p06",
            # "sinayliu/scnt-p07",
            # "sinayliu/scnt-p08",
            # "sinayliu/scnt-p09",
            # "sinayliu/scnt-p10",
            # "sinayliu/scnt-p11",
            # "sinayliu/scnt-p12",
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
    if not cfg.get("resume_full_ckpt"):
        return None
    if cfg.get("resume_source") != "kernel":
        return None
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
            raise ValueError(f"Key not found in yaml: '{'.'.join(parts[:i+1])}'")
        if is_last:
            cur[part] = value
        else:
            cur = cur[part]


def _resolve_default_group_yaml(default_cfg: dict, group_name: str) -> Path | None:
    defaults = default_cfg.get("defaults")
    if not isinstance(defaults, list):
        return None

    selected = None
    for item in defaults:
        if isinstance(item, dict) and group_name in item:
            selected = item[group_name]
            break
    if selected in (None, ""):
        return None

    rel = Path(*str(selected).split("/")).with_suffix(".yaml")
    return (BASE_DIR.parent / "training" / "configs" / group_name / rel).resolve()


def _apply_simple_to_default_yaml(cfg: dict) -> list[tuple[Path, list[tuple[str, object]]]] | None:
    updates = _iter_simple_updates(cfg)
    if not updates:
        return None

    default_yaml_path = (BASE_DIR.parent / "training" / "configs" / "default.yaml").resolve()
    default_data = _load_yaml_roundtrip(default_yaml_path)
    if not isinstance(default_data, dict) or not default_data:
        raise ValueError(f"Failed to load default.yaml: {default_yaml_path}")

    routed_updates: dict[Path, list[tuple[str, object]]] = {}
    default_updates: list[tuple[str, object]] = []

    for key, value in updates:
        top_key, sep, remainder = str(key).partition(".")
        if top_key in default_data:
            default_updates.append((key, value))
            continue

        group_yaml = _resolve_default_group_yaml(default_data, top_key)
        if group_yaml is None:
            raise ValueError(
                f"simple key '{key}' not found in default.yaml and no Hydra defaults group entry for '{top_key}'"
            )
        if not remainder:
            raise ValueError(f"simple key '{key}' must include a nested path inside group '{top_key}'")
        routed_updates.setdefault(group_yaml, []).append((remainder, value))

    results: list[tuple[Path, list[tuple[str, object]]]] = []

    if default_updates:
        for key, value in default_updates:
            _set_by_dot_path(default_data, key, value)
        _write_yaml_roundtrip(default_yaml_path, default_data)
        results.append((default_yaml_path, default_updates))

    for yaml_path, path_updates in routed_updates.items():
        group_data = _load_yaml_roundtrip(yaml_path)
        if not isinstance(group_data, dict) or not group_data:
            raise ValueError(f"Failed to load group config yaml: {yaml_path}")
        for key, value in path_updates:
            _set_by_dot_path(group_data, key, value)
        _write_yaml_roundtrip(yaml_path, group_data)
        results.append((yaml_path, path_updates))

    return results


def _apply_special_to_default_yaml(cfg: dict) -> list[tuple[Path, list[tuple[str, object]]]] | None:
    updates: list[tuple[str, object]] = []
    if cfg.get("resume_checkpoint_path") is not None:
        updates.append(("checkpoint.resume_checkpoint_path", str(cfg.get("resume_checkpoint_path"))))
    if not updates:
        return None

    default_yaml_path = (BASE_DIR.parent / "training" / "configs" / "default.yaml").resolve()
    default_data = _load_yaml_roundtrip(default_yaml_path)
    if not isinstance(default_data, dict) or not default_data:
        raise ValueError(f"Failed to load default.yaml: {default_yaml_path}")
    for key, value in updates:
        _set_by_dot_path(default_data, key, value)
    _write_yaml_roundtrip(default_yaml_path, default_data)
    return [(default_yaml_path, updates)]


def _add_running_node(
    cfg: dict,
    kernel_id: str,
    total_runs: int,
    config_kernel_path: Path,
    use_lock: bool = True,
    consume_available: bool = False,
) -> None:
    lock_ctx = file_lock(config_kernel_path, timeout_sec=600, poll_interval=3.0) if use_lock else nullcontext()
    with lock_ctx:
        kcfg = _load_yaml(config_kernel_path)
        running_nodes = kcfg.get("running_nodes") or []
        available_ids = kcfg.get("available_ids") or []
        owner_id = str(cfg["id"])
        default_quota_hours = _default_quota_hours_for_add_node(cfg, config_kernel_path)

        if consume_available and owner_id in available_ids:
            available_ids.remove(owner_id)
            kcfg["available_ids"] = available_ids

        node = None
        for item in running_nodes:
            if item.get("id") == owner_id:
                node = item
                break
        if node is None:
            node = {
                "id": owner_id,
                "left_time": float(kcfg.get("default_quota_hours", default_quota_hours) or default_quota_hours),
                "notebooks": [],
            }
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
            "cfg": copy.deepcopy(cfg),
        }
        notebooks.append(notebook)
        node["notebooks"] = notebooks
        kcfg["running_nodes"] = running_nodes
        _write_yaml(config_kernel_path, kcfg)


def _is_tpu_add_node_config(cfg: dict, config_kernel_path: Path) -> bool:
    # Respect explicit TPU config, and fall back to TPU config filename convention.
    if bool(cfg.get("enable_tpu", False)):
        return True
    return "tpu" in config_kernel_path.name.lower()


def _default_total_runs_for_add_node(cfg: dict, config_kernel_path: Path) -> int:
    # Preserve legacy behavior: GPU add-node jobs default to 8 runs; TPU stays single-run unless configured.
    return 1 if _is_tpu_add_node_config(cfg, config_kernel_path) else 8


def _default_quota_hours_for_add_node(cfg: dict, config_kernel_path: Path) -> float:
    # Preserve legacy quota defaults when default_quota_hours is absent in queue config.
    return 20.0 if _is_tpu_add_node_config(cfg, config_kernel_path) else 30.0


def _choose_id_for_add_node(
    config_kernel_path: Path,
    *,
    notebook_limit: int = 2,
    default_quota_hours: float = 30.0,
    use_lock: bool = True,
) -> tuple[str, bool]:
    lock_ctx = file_lock(config_kernel_path, timeout_sec=600, poll_interval=3.0) if use_lock else nullcontext()
    with lock_ctx:
        kcfg = _load_yaml(config_kernel_path)
        running_nodes = kcfg.get("running_nodes") or []
        select_now = datetime.now()
        default_quota_hours = float(kcfg.get("default_quota_hours", default_quota_hours) or default_quota_hours)
        for node in running_nodes:
            if _effective_left_time_for_selection(
                node.get("left_time"),
                node.get("notebooks"),
                default_left_time=default_quota_hours,
                now=select_now,
            ) <= 0:
                continue
            notebooks = node.get("notebooks") or []
            if len(notebooks) >= notebook_limit:
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
    parser.add_argument(
        "--config-kernel",
        default=None,
        help="Queue config path (default: TPU uses config_kernel_tpu.yaml, otherwise config_kernel.yaml).",
    )
    parser.add_argument("--no-lock", action="store_false", dest="lock", default=True, help="Disable config_kernel.yaml locking.")
    args = parser.parse_args()

    cfg_path = Path(args.cfg).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (BASE_DIR / cfg_path).resolve()
    cfg = _load_yaml(cfg_path)
    if not cfg:
        raise ValueError(f"Empty or missing config: {cfg_path}")
    if args.config_kernel:
        config_kernel_path = Path(args.config_kernel).expanduser()
        if not config_kernel_path.is_absolute():
            config_kernel_path = (BASE_DIR / config_kernel_path).resolve()
    else:
        config_kernel_name = "config_kernel_tpu.yaml" if bool(cfg.get("enable_tpu", False)) else "config_kernel.yaml"
        config_kernel_path = (BASE_DIR / config_kernel_name).resolve()

    selected_from_available = False
    if args.add_running_node and not cfg.get("id"):
        is_tpu_add = _is_tpu_add_node_config(cfg, config_kernel_path)
        notebook_limit = 1 if is_tpu_add else 2
        chosen_id, selected_from_available = _choose_id_for_add_node(
            config_kernel_path,
            notebook_limit=notebook_limit,
            default_quota_hours=(20.0 if is_tpu_add else 30.0),
            use_lock=args.lock,
        )
        cfg["id"] = chosen_id
    if "id" not in cfg or cfg.get("id") in (None, ""):
        raise ValueError("config.yaml requires 'id' (Kaggle owner id), or use --add-running-node with config_kernel.yaml.available_ids")

    simple_result = _apply_simple_to_default_yaml(cfg)
    special_result = _apply_special_to_default_yaml(cfg)
    meta = _update_metadata(cfg)

    # Mirror pos behavior: this flag is an internal mode used by auto_resume.py.
    suppress_prep_output = bool(args.push_output_only)
    if not suppress_prep_output:
        if args.concise:
            print(f"kernel_id: {meta['id']}")
        else:
            print(f"Updated {BASE_DIR / 'kernel-metadata.json'}")
            all_cfg_updates = []
            if simple_result is not None:
                all_cfg_updates.extend(simple_result)
            if special_result is not None:
                all_cfg_updates.extend(special_result)
            if all_cfg_updates:
                for yaml_path, updates in all_cfg_updates:
                    print(f"Updated {yaml_path}")
                    print(f"config updates: {[k for k, _ in updates]}")
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
        cfg_total_runs = cfg.get("total_runs")
        if args.total_runs is not None:
            total_runs = args.total_runs
        elif cfg_total_runs not in (None, ""):
            total_runs = int(cfg_total_runs)
        else:
            total_runs = _default_total_runs_for_add_node(cfg, config_kernel_path)
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
