from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch


def _confidence_mask(
    confidence: Optional[np.ndarray],
    finite_points: np.ndarray,
    confidence_conf: Mapping[str, Any],
) -> np.ndarray:
    if confidence is None:
        return finite_points

    finite_conf = np.isfinite(confidence)
    mode = confidence_conf.get("mode", "threshold")
    if mode == "threshold":
        selected = confidence > float(confidence_conf.get("threshold", 1.0))
    elif mode == "percentile":
        percentile = float(confidence_conf.get("percentile", 85.0))
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(
                f"confidence percentile must be in [0, 100], got {percentile}"
            )
        usable = confidence[finite_conf & finite_points]
        if usable.size == 0:
            return np.zeros_like(finite_points, dtype=bool)
        selected = confidence >= np.percentile(usable, percentile)
    elif mode == "all_finite":
        selected = finite_conf
    else:
        raise ValueError(
            "confidence mode must be 'threshold', 'percentile', or 'all_finite', "
            f"got {mode!r}"
        )
    return finite_points & finite_conf & selected


def _invert_se3_numpy(pose: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float32)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def _estimate_one_pose(
    points: np.ndarray,
    confidence: Optional[np.ndarray],
    confidence_conf: Mapping[str, Any],
    pnp_conf: Mapping[str, Any],
    pose_convention: str,
) -> Tuple[np.ndarray, float, bool]:
    height, width, channels = points.shape
    if channels != 3:
        raise ValueError(f"Expected point maps shaped [H,W,3], got {points.shape}")

    finite_points = np.isfinite(points).all(axis=-1)
    mask = _confidence_mask(confidence, finite_points, confidence_conf)
    if int(mask.sum()) < 4:
        return np.eye(4, dtype=np.float32), float("nan"), True

    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    pixels = np.stack((x, y), axis=-1)
    object_points = np.ascontiguousarray(points[mask], dtype=np.float32)
    image_points = np.ascontiguousarray(pixels[mask], dtype=np.float32)

    image_extent = float(max(height, width))
    focal_count = int(pnp_conf.get("focal_candidates", 100))
    focal_min = image_extent * float(pnp_conf.get("focal_min_factor", 0.5))
    focal_max = image_extent * float(pnp_conf.get("focal_max_factor", 3.0))
    if focal_count <= 0 or focal_min <= 0 or focal_max < focal_min:
        raise ValueError(
            "PnP focal search requires focal_candidates > 0 and "
            f"0 < focal_min <= focal_max, got {focal_count=}, {focal_min=}, {focal_max=}"
        )
    focal_candidates = np.geomspace(focal_min, focal_max, num=focal_count)

    principal_point = (width / 2.0, height / 2.0)
    iterations = int(pnp_conf.get("iterations", 100))
    reprojection_error = float(pnp_conf.get("reprojection_error", 5.0))
    best = None
    for focal in focal_candidates:
        intrinsics = np.array(
            [
                [focal, 0.0, principal_point[0]],
                [0.0, focal, principal_point[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        try:
            success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                intrinsics,
                None,
                iterationsCount=iterations,
                reprojectionError=reprojection_error,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error:
            continue
        if not success or inliers is None:
            continue
        score = len(inliers)
        if best is None or score > best[0]:
            best = (score, rotation_vector, translation, float(focal))

    if best is None:
        return np.eye(4, dtype=np.float32), float("nan"), True

    _, rotation_vector, translation, focal = best
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = cv2.Rodrigues(rotation_vector)[0].astype(np.float32)
    world_to_camera[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    pose = (
        _invert_se3_numpy(world_to_camera)
        if pose_convention == "c2w"
        else world_to_camera
    )
    return pose, focal, False


def estimate_poses_from_world_points(
    world_points: torch.Tensor,
    confidence: Optional[torch.Tensor],
    confidence_conf: Mapping[str, Any],
    pnp_conf: Mapping[str, Any],
    pose_convention: str = "c2w",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate one camera pose per dense, pixel-aligned world point map."""
    if pose_convention not in {"c2w", "w2c"}:
        raise ValueError(
            f"pose_convention must be 'c2w' or 'w2c', got {pose_convention!r}"
        )
    if world_points.ndim != 5 or world_points.shape[-1] != 3:
        raise ValueError(
            f"world_points must have shape [B,S,H,W,3], got {tuple(world_points.shape)}"
        )
    if confidence is not None and confidence.shape != world_points.shape[:-1]:
        raise ValueError(
            "confidence must match world_points without its XYZ dimension, "
            f"got {tuple(confidence.shape)} for {tuple(world_points.shape)}"
        )

    points_numpy = world_points.detach().to(device="cpu", dtype=torch.float32).numpy()
    confidence_numpy = (
        confidence.detach().to(device="cpu", dtype=torch.float32).numpy()
        if confidence is not None
        else None
    )
    batch_size, num_views = points_numpy.shape[:2]
    if batch_size == 0 or num_views == 0:
        raise ValueError(
            f"world_points must contain at least one batch item and view, got {tuple(world_points.shape)}"
        )
    poses = np.empty((batch_size, num_views, 4, 4), dtype=np.float32)
    focals = np.empty((batch_size, num_views), dtype=np.float32)
    failures = np.empty((batch_size, num_views), dtype=bool)

    view_indices = [
        (batch_idx, view_idx)
        for batch_idx in range(batch_size)
        for view_idx in range(num_views)
    ]

    def estimate_index(index):
        batch_idx, view_idx = index
        view_confidence = (
            confidence_numpy[batch_idx, view_idx]
            if confidence_numpy is not None
            else None
        )
        return _estimate_one_pose(
            points_numpy[batch_idx, view_idx],
            view_confidence,
            confidence_conf,
            pnp_conf,
            pose_convention,
        )

    with ThreadPoolExecutor(max_workers=min(32, len(view_indices))) as executor:
        results = executor.map(estimate_index, view_indices)
        for (batch_idx, view_idx), (pose, focal, failed) in zip(view_indices, results):
            poses[batch_idx, view_idx] = pose
            focals[batch_idx, view_idx] = focal
            failures[batch_idx, view_idx] = failed

    return torch.from_numpy(poses), torch.from_numpy(focals), torch.from_numpy(failures)
