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

    # --- Error Calculation ---
    distances = torch.linalg.norm(points_gt - points_pred, dim=-1)

    # Mean Absolute Error (MAE)
    mae = torch.mean(distances)

    # --- Relative Metrics ---
    # Calculate the distance of each ground truth point from the origin.
    dist_from_origin = torch.linalg.norm(points_gt, dim=-1)
    
    # Create a mask to avoid division by zero for points at the origin.
    non_zero_mask = dist_from_origin > 1e-8
    
    # Initialize relative metrics to zero.
    abs_rel = torch.tensor(0.0, device=points_gt.device)    

    # Only compute relative metrics if there are points away from the origin.
    if torch.any(non_zero_mask):
        abs_rel = torch.mean(distances[non_zero_mask] / dist_from_origin[non_zero_mask])

    metrics = {
        'recon_mae': mae, 
        'recon_abs_rel': abs_rel,
    }

    return metrics
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
