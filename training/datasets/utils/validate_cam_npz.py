#!/usr/bin/env python3
import argparse
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
        }
    orth = np.max(np.abs(rot.T @ rot - np.eye(3)))
    det = np.linalg.det(rot)
    return {
        "finite": True,
        "orth": float(orth),
        "det": float(det),
    }


def geodesic_cos(rot1, rot2):
    return float((np.trace(rot1 @ rot2.T) - 1.0) / 2.0)


def validate_file(path):
    rows = []
    with np.load(path) as data:
        if "pose" not in data:
            raise KeyError("missing key 'pose'")
        pose = data["pose"]

    for pose_idx, mat in iter_pose_matrices(pose):
        rot = mat[:3, :3].astype(np.float64)
        stats = rotation_stats(rot)
        rows.append(
            {
                "path": path,
                "pose_idx": pose_idx,
                "pose_shape": pose.shape,
                "rot": rot,
                **stats,
            }
        )
    return rows


def print_worst(title, rows, key_fn, limit):
    print(f"\n{title}")
    for row in sorted(rows, key=key_fn, reverse=True)[:limit]:
        print(
            f"{row['path']} pose_idx={row['pose_idx']} shape={row['pose_shape']} "
            f"orth={row['orth']:.9g} det={row['det']:.9g}"
        )
        print(f"{row['rot']}")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively validate rotation matrices in *_cam.npz files."
    )
    parser.add_argument("dir", type=Path, help="Directory to recursively scan.")
    parser.add_argument("--pattern", default="*_cam.npz", help="Glob pattern, default: *_cam.npz")
    parser.add_argument("--orth-threshold", type=float, default=1e-3)
    parser.add_argument("--det-threshold", type=float, default=1e-3)
    parser.add_argument("--cos-tol", type=float, default=1e-6)
    parser.add_argument("--worst", type=int, default=10)
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Also check all pairwise geodesic cosines within each leaf directory.",
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
            rows.extend(validate_file(path))
        except Exception as exc:
            errors.append((path, exc))

    finite_rows = [row for row in rows if row["finite"]]
    bad_finite = [row for row in rows if not row["finite"]]
    orth_bad = [row for row in finite_rows if row["orth"] > args.orth_threshold]
    det_bad = [row for row in finite_rows if abs(row["det"] - 1.0) > args.det_threshold]

    print(f"poses: {len(rows)}")
    print(f"read_errors: {len(errors)}")
    print(f"finite_rotations: {len(finite_rows)}/{len(rows)}")
    print(f"nonfinite_rotations: {len(bad_finite)}")

    if finite_rows:
        orth = np.array([row["orth"] for row in finite_rows])
        det = np.array([row["det"] for row in finite_rows])
        print(
            "orth: "
            f"max={orth.max():.9g} mean={orth.mean():.9g} "
            f"p95={np.percentile(orth, 95):.9g} p99={np.percentile(orth, 99):.9g} "
            f"bad>{args.orth_threshold:g}={len(orth_bad)}"
        )
        print(
            "det: "
            f"min={det.min():.9g} max={det.max():.9g} mean={det.mean():.9g} "
            f"max_abs_err={np.max(np.abs(det - 1.0)):.9g} "
            f"bad>{args.det_threshold:g}={len(det_bad)}"
        )

        print_worst("worst orthogonality", finite_rows, lambda row: row["orth"], args.worst)
        print_worst("worst determinant error", finite_rows, lambda row: abs(row["det"] - 1.0), args.worst)

    if errors:
        print("\nread errors")
        for path, exc in errors[: args.worst]:
            print(f"{path}: {type(exc).__name__}: {exc}")

    if bad_finite:
        print("\nnonfinite rotations")
        for row in bad_finite[: args.worst]:
            print(f"{row['path']} pose_idx={row['pose_idx']} shape={row['pose_shape']}")

    if args.pairwise:
        outside_pairs = []
        grouped = {}
        for row in finite_rows:
            grouped.setdefault(row["path"].parent, []).append(row)

        total_pairs = 0
        cos_min = np.inf
        cos_max = -np.inf
        for parent, group in grouped.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    total_pairs += 1
                    cos = geodesic_cos(group[i]["rot"], group[j]["rot"])
                    cos_min = min(cos_min, cos)
                    cos_max = max(cos_max, cos)
                    if cos < -1.0 - args.cos_tol or cos > 1.0 + args.cos_tol:
                        outside_pairs.append((parent, group[i], group[j], cos))

        print("\npairwise geodesic cosine by parent directory")
        print(f"pairs: {total_pairs}")
        if total_pairs:
            print(f"cos_min={cos_min:.9g} cos_max={cos_max:.9g}")
        print(f"outside [-1, 1] by tol {args.cos_tol:g}: {len(outside_pairs)}")
        for parent, row1, row2, cos in sorted(
            outside_pairs,
            key=lambda item: max(item[3] - 1.0, -1.0 - item[3]),
            reverse=True,
        )[: args.worst]:
            print(
                f"{parent}: {row1['path'].name}[{row1['pose_idx']}] "
                f"vs {row2['path'].name}[{row2['pose_idx']}] cos={cos:.9g}"
            )


if __name__ == "__main__":
    main()
