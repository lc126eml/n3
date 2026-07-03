#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np


def iter_pose_matrices(pose):
    pose = np.asarray(pose)
    if pose.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"unsupported pose shape {pose.shape}")
    flat = pose.reshape((-1,) + pose.shape[-2:])
    for idx, mat in enumerate(flat):
        yield idx, mat


def rotation_stats(rot):
    finite = np.isfinite(rot).all()
    if not finite:
        return {
            "finite": False,
            "orth": np.nan,
            "det": np.nan,
            "scale_est": np.nan,
            "normalized_orth": np.nan,
            "normalized_det": np.nan,
        }

    det = float(np.linalg.det(rot))
    orth = float(np.max(np.abs(rot.T @ rot - np.eye(3))))
    scale_est = float(abs(det) ** (1.0 / 3.0)) if np.isfinite(det) else np.nan
    normalized_orth = np.nan
    normalized_det = np.nan
    if np.isfinite(scale_est) and scale_est > 0:
        rot_no_scale = rot / scale_est
        normalized_orth = float(np.max(np.abs(rot_no_scale.T @ rot_no_scale - np.eye(3))))
        normalized_det = float(np.linalg.det(rot_no_scale))

    return {
        "finite": True,
        "orth": orth,
        "det": det,
        "scale_est": scale_est,
        "normalized_orth": normalized_orth,
        "normalized_det": normalized_det,
    }


def is_abnormal(stats, orth_threshold, det_threshold):
    if not stats["finite"]:
        return True
    return (
        stats["orth"] > orth_threshold
        or abs(stats["det"] - 1.0) > det_threshold
    )


def is_scaling_fixable(stats, normalized_orth_threshold, normalized_det_threshold, det_eps):
    if not stats["finite"]:
        return False
    if not np.isfinite(stats["det"]) or abs(stats["det"]) <= det_eps:
        return False
    if not np.isfinite(stats["scale_est"]) or stats["scale_est"] <= 0:
        return False
    if not np.isfinite(stats["normalized_orth"]) or not np.isfinite(stats["normalized_det"]):
        return False
    if stats["normalized_orth"] > normalized_orth_threshold:
        return False
    return abs(stats["normalized_det"] - 1.0) <= normalized_det_threshold


def load_npz(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def save_npz_atomic(path, arrays):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp_path, path)


def process_file(
    path,
    orth_threshold,
    det_threshold,
    normalized_orth_threshold,
    normalized_det_threshold,
    det_eps,
    write,
):
    arrays = load_npz(path)
    if "pose" not in arrays:
        raise KeyError("missing key 'pose'")

    pose = np.array(arrays["pose"], copy=True)
    original_dtype = pose.dtype
    flat_pose = pose.reshape((-1,) + pose.shape[-2:])

    reports = []
    changed = False
    for pose_idx, mat in iter_pose_matrices(pose):
        rot = mat[:3, :3].astype(np.float64)
        stats = rotation_stats(rot)
        if not is_abnormal(stats, orth_threshold, det_threshold):
            continue

        fixable = is_scaling_fixable(
            stats,
            normalized_orth_threshold,
            normalized_det_threshold,
            det_eps,
        )
        action = "skip"
        fixed_stats = None
        if fixable:
            fixed_rot = rot / stats["scale_est"]
            fixed_stats = rotation_stats(fixed_rot)
            action = "would_fix"
            if write:
                flat_pose[pose_idx][:3, :3] = fixed_rot.astype(original_dtype, copy=False)
                changed = True
                action = "fixed"

        reports.append(
            {
                "path": path,
                "pose_idx": pose_idx,
                "pose_shape": pose.shape,
                "action": action,
                "stats": stats,
                "fixed_stats": fixed_stats,
            }
        )

    if changed:
        arrays["pose"] = pose
        save_npz_atomic(path, arrays)

    return reports, changed


def print_report(row):
    stats = row["stats"]
    print(
        f"{row['action']}: {row['path']} pose_idx={row['pose_idx']} "
        f"shape={row['pose_shape']} "
        f"orth={stats['orth']:.9g} det={stats['det']:.9g} "
        f"scale_est={stats['scale_est']:.9g} "
        f"normalized_orth={stats['normalized_orth']:.9g} "
        f"normalized_det={stats['normalized_det']:.9g}"
    )
    fixed_stats = row["fixed_stats"]
    if fixed_stats is not None:
        print(
            "  after_scale_fix: "
            f"orth={fixed_stats['orth']:.9g} det={fixed_stats['det']:.9g}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recursively validate *_cam.npz poses and fix abnormal rotations that "
            "are valid rotations multiplied by a uniform scale."
        )
    )
    parser.add_argument("dir", type=Path, help="Directory to recursively scan.")
    parser.add_argument("--pattern", default="*_cam.npz", help="Glob pattern, default: *_cam.npz")
    parser.add_argument("--orth-threshold", type=float, default=1e-3)
    parser.add_argument("--det-threshold", type=float, default=1e-3)
    parser.add_argument("--normalized-orth-threshold", type=float, default=1e-3)
    parser.add_argument("--normalized-det-threshold", type=float, default=1e-3)
    parser.add_argument("--det-eps", type=float, default=1e-12)
    parser.add_argument("--max-print", type=int, default=200)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually rewrite fixable npz files. Without this flag, only report what would change.",
    )
    args = parser.parse_args()

    root = args.dir
    if not root.is_dir():
        raise NotADirectoryError(root)

    paths = sorted(root.rglob(args.pattern))
    print(f"root: {root}")
    print(f"files: {len(paths)}")
    print(f"mode: {'write' if args.write else 'dry-run'}")
    if not paths:
        return

    all_reports = []
    errors = []
    changed_files = 0
    for path in paths:
        try:
            reports, changed = process_file(
                path,
                args.orth_threshold,
                args.det_threshold,
                args.normalized_orth_threshold,
                args.normalized_det_threshold,
                args.det_eps,
                args.write,
            )
        except Exception as exc:
            errors.append((path, exc))
            continue
        all_reports.extend(reports)
        changed_files += int(changed)

    fixable = [row for row in all_reports if row["action"] in ("would_fix", "fixed")]
    skipped = [row for row in all_reports if row["action"] == "skip"]
    print(f"abnormal_poses: {len(all_reports)}")
    print(f"fixable_by_scaling: {len(fixable)}")
    print(f"not_fixable_by_scaling: {len(skipped)}")
    print(f"changed_files: {changed_files}")
    print(f"read_or_write_errors: {len(errors)}")

    if all_reports:
        print("\nabnormal poses")
        for row in all_reports[: args.max_print]:
            print_report(row)

    if errors:
        print("\nerrors")
        for path, exc in errors[: args.max_print]:
            print(f"{path}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
