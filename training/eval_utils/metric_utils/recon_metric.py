# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Built on top of https://github.com/HengyiWang/spann3r/blob/main/spann3r/tools/eval_recon.py

import numpy as np
from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import NearestNeighbors
import torch
from typing import Tuple, Union, Dict

def calculate_corresponding_points_error_torch_optimized(
    points_gt: torch.Tensor,
    points_pred: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Calculates reconstruction errors between two sets of points with known point-to-point correspondences using PyTorch.

    This function assumes that the i-th point in `points_gt` corresponds directly
    to the i-th point in `points_pred`.
    The computation will be performed on the device of the input tensors (CPU or GPU).

    Args:
        points_gt (torch.Tensor): The reference points, a PyTorch tensor of shape (N, 3).
        points_pred (torch.Tensor): The predicted points, a PyTorch tensor of shape (N, 3).

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the computed metrics as single-element tensors:
                                 'recon_mae' (Mean Absolute Error),
                                 'recon_abs_rel' (Absolute Relative Error),
    """
    # --- Pre-computation Checks ---
    if points_gt.shape != points_pred.shape:
        raise ValueError(f"Input point clouds must have the same shape. "
                         f"Got {points_gt.shape} and {points_pred.shape}.")
    finite_mask = torch.isfinite(points_gt).all(dim=-1) & torch.isfinite(points_pred).all(dim=-1)
    points_gt = points_gt[finite_mask]
    points_pred = points_pred[finite_mask]
    nan = torch.tensor(float("nan"), device=points_gt.device, dtype=points_gt.dtype)
    if points_gt.numel() == 0:
        return {
            'recon_mae': nan,
            'recon_abs_rel': nan,
        }

    # --- Error Calculation ---
    distances = torch.linalg.norm(points_gt - points_pred, dim=-1)

    # Mean Absolute Error (MAE)
    mae = torch.mean(distances)

    # --- Relative Metrics ---
    # Calculate the distance of each ground truth point from the origin.
    dist_from_origin = torch.linalg.norm(points_gt, dim=-1)
    
    # Create a mask to avoid division by zero for points at the origin.
    non_zero_mask = dist_from_origin > 1e-8
    
    abs_rel = torch.tensor(float("nan"), device=points_gt.device, dtype=points_gt.dtype)

    # Only compute relative metrics if there are points away from the origin.
    if torch.any(non_zero_mask):
        abs_rel = torch.mean(distances[non_zero_mask] / dist_from_origin[non_zero_mask])

    metrics = {
        'recon_mae': mae, 
        'recon_abs_rel': abs_rel,
    }

    return metrics


def _finite_points(points: torch.Tensor) -> torch.Tensor:
    points = points.reshape(-1, 3)
    return points[torch.isfinite(points).all(dim=-1)]


def _deterministic_subsample(points: torch.Tensor, max_points: int | None) -> torch.Tensor:
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=max_points,
        device=points.device,
    ).long()
    return points[indices]


def calculate_nearest_neighbor_pointcloud_metrics(
    points_gt: torch.Tensor,
    points_pred: torch.Tensor,
    threshold: float = 0.05,
    max_points: int | None = 200000,
    workers: int = 24,
) -> Dict[str, torch.Tensor]:
    """Compute unordered point-cloud nearest-neighbor metrics.

    Accuracy is pred->GT nearest-neighbor distance; completion is GT->pred
    nearest-neighbor distance. Precision/recall/F1 are thresholded versions
    of the same two directed distances.
    """
    device = points_gt.device
    dtype = points_gt.dtype
    nan = torch.tensor(float("nan"), device=device, dtype=dtype)

    points_gt = _deterministic_subsample(_finite_points(points_gt), max_points)
    points_pred = _deterministic_subsample(_finite_points(points_pred), max_points)

    if points_gt.numel() == 0 or points_pred.numel() == 0:
        return {
            "nn_acc_mean": nan,
            "nn_acc_median": nan,
            "nn_comp_mean": nan,
            "nn_comp_median": nan,
            f"nn_precision@{threshold:.3f}m": nan,
            f"nn_recall@{threshold:.3f}m": nan,
            f"nn_f1@{threshold:.3f}m": nan,
        }

    gt_np = points_gt.detach().to(device="cpu", dtype=torch.float32).numpy()
    pred_np = points_pred.detach().to(device="cpu", dtype=torch.float32).numpy()

    pred_to_gt, _ = KDTree(gt_np).query(pred_np, workers=workers)
    gt_to_pred, _ = KDTree(pred_np).query(gt_np, workers=workers)

    pred_to_gt = torch.from_numpy(pred_to_gt).to(device=device, dtype=dtype)
    gt_to_pred = torch.from_numpy(gt_to_pred).to(device=device, dtype=dtype)

    precision = (pred_to_gt < threshold).float().mean()
    recall = (gt_to_pred < threshold).float().mean()
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    return {
        "nn_acc_mean": pred_to_gt.mean(),
        "nn_acc_median": pred_to_gt.median(),
        "nn_comp_mean": gt_to_pred.mean(),
        "nn_comp_median": gt_to_pred.median(),
        f"nn_precision@{threshold:.3f}m": precision,
        f"nn_recall@{threshold:.3f}m": recall,
        f"nn_f1@{threshold:.3f}m": f1,
    }


def calculate_batched_nearest_neighbor_pointcloud_metrics(
    points_gt: torch.Tensor,
    points_pred: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.05,
    max_points: int | None = 200000,
    workers: int = 24,
) -> Dict[str, torch.Tensor]:
    """Compute NN metrics independently per scene and average scene results.

    The first dimension is the scene dimension. Remaining point dimensions are
    flattened within each scene, e.g. ``(B, S, H, W, 3) -> B x (S*H*W)``.
    """
    if points_gt.shape != points_pred.shape:
        raise ValueError(
            f"Batched point clouds must have the same shape. Got {points_gt.shape} and {points_pred.shape}."
        )
    if points_gt.ndim < 3 or points_gt.shape[-1] != 3:
        raise ValueError(f"Expected batched point clouds shaped (B, ..., 3), got {points_gt.shape}.")
    if points_gt.shape[0] == 0:
        raise ValueError("Batched point clouds must contain at least one scene.")
    if valid_mask is not None and valid_mask.shape != points_gt.shape[:-1]:
        raise ValueError(
            f"Mask shape must match point dimensions without XYZ. Got {valid_mask.shape} for {points_gt.shape}."
        )

    scene_metrics = []
    for scene_idx in range(points_gt.shape[0]):
        gt_scene = points_gt[scene_idx]
        pred_scene = points_pred[scene_idx]
        if valid_mask is not None:
            scene_mask = valid_mask[scene_idx].bool()
            gt_scene = gt_scene[scene_mask]
            pred_scene = pred_scene[scene_mask]

        scene_metrics.append(
            calculate_nearest_neighbor_pointcloud_metrics(
                gt_scene,
                pred_scene,
                threshold=threshold,
                max_points=max_points,
                workers=workers,
            )
        )

    aggregated = {}
    for key in scene_metrics[0]:
        values = torch.stack([metrics[key] for metrics in scene_metrics])
        finite = torch.isfinite(values)
        aggregated[key] = values[finite].mean() if finite.any() else values[0]
    return aggregated


# import faiss
def calculate_corresponding_points_error(points_gt, points_pred, metric='mean', include_relative=True):
    """
    Calculates the error between two sets of points with known point-to-point correspondences.

    This function assumes that the i-th point in `points_gt` corresponds directly
    to the i-th point in `points_pred`. It can compute absolute error and optionally
    a relative error, aggregated by either the mean or median.

    Args:
        points_gt (np.ndarray): The reference points, a numpy array of shape (N, 3).
        points_pred (np.ndarray): The second set of points, a numpy array of shape (N, 3).
        metric (str, optional): The aggregation metric to use.
                                Must be 'mean' or 'median'. Defaults to 'mean'.
        include_relative (bool, optional): If True, also calculates the relative error,
                                           defined as the absolute error divided by the
                                           magnitude of the reference point vector. Defaults to True.

    Returns:
        float: If `include_relative` is False, returns the single aggregated absolute error value.
        tuple[float, float]: If `include_relative` is True, returns a tuple containing:
                             - The aggregated absolute error.
                             - The aggregated relative error.
    """
    # --- Parameter Validation and Function Selection ---
    if metric == 'mean':
        agg_func = np.mean
    elif metric == 'median':
        agg_func = np.median
    else:
        raise ValueError(f"Invalid metric: '{metric}'. Must be 'mean' or 'median'.")

    # --- Pre-computation Checks ---
    # Ensure the point clouds have the same number of points for a valid correspondence
    if points_gt.shape != points_pred.shape:
        raise ValueError(f"Input point clouds must have the same shape. "
                         f"Got {points_gt.shape} and {points_pred.shape}.")

    # --- Absolute Error Calculation ---
    # Calculate the Euclidean distance for each corresponding point pair.
    distances = np.linalg.norm(points_gt - points_pred, axis=-1)
    absolute_pts_error = agg_func(distances)

    if not include_relative:
        return absolute_pts_error

    # --- Relative Error Calculation ---
    # Calculate the magnitude (L2 norm) of each reference point vector.
    dist_from_origin = np.linalg.norm(points_gt, axis=-1)
    # print('gt mean', agg_func(dist_from_origin))

    # Avoid division by zero for points at the origin.
    # We create a mask of points where the distance from the origin is not zero.
    non_zero_mask = dist_from_origin > 0
    
    # Calculate relative error only for the non-zero points.
    relative_pts_error = distances[non_zero_mask] / dist_from_origin[non_zero_mask]
    
    relative_pts_error = agg_func(relative_pts_error)

    return absolute_pts_error, relative_pts_error


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points, workers=24)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None, device=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=24)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None, device=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=24)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


def compute_iou(pred_vox, target_vox):
    # Get voxel indices
    v_pred_indices = [voxel.grid_index for voxel in pred_vox.get_voxels()]
    v_target_indices = [voxel.grid_index for voxel in target_vox.get_voxels()]

    # Convert to sets for set operations
    v_pred_filled = set(tuple(np.round(x, 4)) for x in v_pred_indices)
    v_target_filled = set(tuple(np.round(x, 4)) for x in v_target_indices)

    # Compute intersection and union
    intersection = v_pred_filled & v_target_filled
    union = v_pred_filled | v_target_filled

    # Compute IoU
    iou = len(intersection) / len(union)
    return iou
