#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np


AXIS_FLIP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


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


def read_orientations(path, dataset_key):
    with h5py.File(path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"missing dataset '{dataset_key}'")
        orientations = f[dataset_key][:]

    orientations = np.asarray(orientations)
    if orientations.shape[-2:] != (3, 3):
        raise ValueError(f"unsupported orientation shape {orientations.shape}")
    return orientations.reshape((-1, 3, 3))


def validate_file(path, dataset_key, apply_axis_flip):
    rows = []
    orientations = read_orientations(path, dataset_key)
    for frame_idx, rot in enumerate(orientations):
        rot = rot.astype(np.float64)
        if apply_axis_flip:
            rot = rot @ AXIS_FLIP
        rows.append(
            {
                "path": path,
                "frame_idx": frame_idx,
                "rot": rot,
                **rotation_stats(rot),
            }
        )
    return rows


def is_abnormal(row, orth_threshold, det_threshold, normalized_orth_threshold):
    if not row["finite"]:
        return True
    if row["orth"] > orth_threshold:
        return True
    if abs(row["det"] - 1.0) > det_threshold:
        return True
    if (
        np.isfinite(row["normalized_orth"])
        and row["normalized_orth"] > normalized_orth_threshold
    ):
        return True
    return False


def abnormal_score(row):
    values = [
        row["orth"],
        abs(row["det"] - 1.0),
        row["normalized_orth"],
        abs(abs(row["normalized_det"]) - 1.0),
    ]
    finite_values = [v for v in values if np.isfinite(v)]
    return max(finite_values) if finite_values else np.inf


def print_row(row):
    print(
        f"{row['path']} frame_idx={row['frame_idx']} "
        f"orth={row['orth']:.9g} det={row['det']:.9g} "
        f"scale_est={row['scale_est']:.9g} "
        f"normalized_orth={row['normalized_orth']:.9g} "
        f"normalized_det={row['normalized_det']:.9g}"        
    )
    print(row["rot"])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recursively validate Hypersim camera_keyframe_orientations.hdf5 files "
            "and print abnormal frame rotations."
        )
    )
    parser.add_argument("dir", type=Path, help="Directory to recursively scan.")
    parser.add_argument(
        "--pattern",
        default="camera_keyframe_orientations.hdf5",
        help="File name/glob pattern, default: camera_keyframe_orientations.hdf5",
    )
    parser.add_argument("--dataset-key", default="dataset")
    parser.add_argument("--orth-threshold", type=float, default=1e-3)
    parser.add_argument("--det-threshold", type=float, default=1e-3)
    parser.add_argument("--normalized-orth-threshold", type=float, default=1e-3)
    parser.add_argument("--max-print", type=int, default=100)
    parser.add_argument(
        "--apply-axis-flip",
        action="store_true",
        help="Validate after the same OpenGL axis flip used by preprocess_hypersim.py.",
    )
    args = parser.parse_args()

    root = args.dir
    if not root.is_dir():
        raise NotADirectoryError(root)

    paths = sorted(root.rglob(args.pattern))
    print(f"root: {root}")
    print(f"files: {len(paths)}")
    if not paths:
        return

    rows = []
    errors = []
    for path in paths:
        try:
            rows.extend(validate_file(path, args.dataset_key, args.apply_axis_flip))
        except Exception as exc:
            errors.append((path, exc))

    finite_rows = [row for row in rows if row["finite"]]
    abnormal_rows = [
        row
        for row in rows
        if is_abnormal(
            row,
            args.orth_threshold,
            args.det_threshold,
            args.normalized_orth_threshold,
        )
    ]

    print(f"frames: {len(rows)}")
    print(f"read_errors: {len(errors)}")
    print(f"finite_rotations: {len(finite_rows)}/{len(rows)}")
    print(f"abnormal_frames: {len(abnormal_rows)}")

    if finite_rows:
        orth = np.array([row["orth"] for row in finite_rows])
        det = np.array([row["det"] for row in finite_rows])
        normalized_orth = np.array([row["normalized_orth"] for row in finite_rows])
        print(
            "orth: "
            f"max={orth.max():.9g} mean={orth.mean():.9g} "
            f"bad>{args.orth_threshold:g}={(orth > args.orth_threshold).sum()}"
        )
        print(
            "det: "
            f"min={det.min():.9g} max={det.max():.9g} "
            f"bad>{args.det_threshold:g}={(np.abs(det - 1.0) > args.det_threshold).sum()}"
        )
        print(
            "normalized_orth_after_scale_removal: "
            f"max={np.nanmax(normalized_orth):.9g} "
            f"bad>{args.normalized_orth_threshold:g}="
            f"{(normalized_orth > args.normalized_orth_threshold).sum()}"
        )

    if errors:
        print("\nread errors")
        for path, exc in errors[: args.max_print]:
            print(f"{path}: {type(exc).__name__}: {exc}")

    if abnormal_rows:
        print("\nabnormal frames")
        for row in sorted(abnormal_rows, key=abnormal_score, reverse=True)[: args.max_print]:
            print_row(row)            


if __name__ == "__main__":
    main()
