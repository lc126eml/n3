
from typing import Dict, Any
import torch

from .metric_utils.camera_metric import (
    compute_absolute_pose_error,
    compute_consecutive_relative_error,
    compute_absolute_pose_error_angle,
    compute_consecutive_relative_error_angle,
    compute_all_pairs_relative_error,
    calculate_accuracy_metrics,
    calculate_auc,
)
from .metric_utils.depth_metric import calculate_depth_metrics_optimized
from .metric_utils.recon_metric import calculate_corresponding_points_error_torch_optimized


def eval_batch(y_hat: Dict[str, Any], batch: Dict[str, Any], metrics_conf: Dict[str, Any], data_keys: Dict[str, Any], pred_data_keys:  Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a batch of predictions against ground truth using a flexible configuration.

    Args:
        y_hat (Dict[str, Any]): Dictionary containing model predictions (e.g., 'extrinsics', 'depth', 'world_points').
        batch (Dict[str, Any]): Dictionary containing ground truth data (e.g., 'extrinsics', 'depths', 'world_points', 'valid_mask').
        metrics_conf (Dict[str, Any]): A configuration dictionary specifying which metrics to compute.

    Returns:
        Dict[str, Any]: A dictionary containing the computed metrics.
    """
    all_metrics = {}

    # Helper to get data, avoiding KeyError
    def get_data(pred_key, gt_key, default=None):
        pred = y_hat.get(pred_key)
        gt = batch.get(gt_key)
        return pred, gt

    # --- Camera Metrics ---
    if metrics_conf.get('camera', {}).get('enabled'):
        cam_conf = metrics_conf['camera']
        pred_poses, gt_poses = get_data(data_keys.get('extrinsics'), data_keys.get('extrinsics'))

        if cam_conf.get('abs_err_angle'):
            results, rot_errors_deg, trans_angle_errors_deg, trans_errors = compute_absolute_pose_error_angle(gt_poses, pred_poses)
            all_metrics.update(results)
            if cam_conf.get('auc'):
                auc = calculate_auc(rot_errors_deg, trans_angle_errors_deg, max_threshold_deg=30)
                all_metrics['auc'] = auc
            

    # --- Depth Metrics ---
    if metrics_conf.get('depth', {}).get('enabled'):
        pred_depth, gt_depth = get_data(pred_data_keys.get('depths'), data_keys.get('depths'))
        valid_mask = batch.get(data_keys.get('valid_mask'))

        all_metrics.update(calculate_depth_metrics_optimized(gt_depth, pred_depth, valid_mask))

    # --- Reconstruction Metrics ---
    if metrics_conf.recon.pts_err and pred_data_keys.world_points in y_hat:
        pred_points_key = pred_data_keys.world_points
        aligned_world_points_key = pred_data_keys.get("aligned_world_points", "aligned_world_points")
        if aligned_world_points_key in y_hat:
            pred_points_key = aligned_world_points_key
        
        pred_pts, gt_pts = get_data(pred_key=pred_points_key, gt_key=data_keys.get('world_points'))
        valid_mask = batch.get(data_keys.get('valid_mask'))

        if valid_mask is not None:
            # Flatten and filter points
            gt_pts_flat = gt_pts[valid_mask]
            pred_pts_flat = pred_pts[valid_mask]
        else:
            gt_pts_flat = gt_pts.view(-1, 3)
            pred_pts_flat = pred_pts.view(-1, 3)

        recon_metrics = calculate_corresponding_points_error_torch_optimized(
            gt_pts_flat,
            pred_pts_flat
        )
        for key, value in recon_metrics.items():
            all_metrics[key] = value.item()
    
    if metrics_conf.recon.from_cam_err and pred_data_keys.get('global_from_cam') in y_hat:
        pred_pts, gt_pts = get_data(pred_key=pred_data_keys.get('global_from_cam'), gt_key=data_keys.get('world_points'))
        valid_mask = batch.get(data_keys.get('valid_mask'))

        if valid_mask is not None:
            # Flatten and filter points
            gt_pts_flat = gt_pts[valid_mask]
            pred_pts_flat = pred_pts[valid_mask]
        else:
            gt_pts_flat = gt_pts.view(-1, 3)
            pred_pts_flat = pred_pts.view(-1, 3)

        recon_metrics = calculate_corresponding_points_error_torch_optimized(
            gt_pts_flat,
            pred_pts_flat
        )
        for key, value in recon_metrics.items():
            all_metrics[f"cam_{key}"] = value.item()
    
    if metrics_conf.recon.from_depth_err and pred_data_keys.get('global_from_depth') in y_hat:
        pred_pts, gt_pts = get_data(pred_key=pred_data_keys.get('global_from_depth'), gt_key=data_keys.get('world_points'))
        valid_mask = batch.get(data_keys.get('valid_mask'))

        if valid_mask is not None:
            # Flatten and filter points
            gt_pts_flat = gt_pts[valid_mask]
            pred_pts_flat = pred_pts[valid_mask]
        else:
            gt_pts_flat = gt_pts.view(-1, 3)
            pred_pts_flat = pred_pts.view(-1, 3)

        recon_metrics = calculate_corresponding_points_error_torch_optimized(
            gt_pts_flat,
            pred_pts_flat
        )
        for key, value in recon_metrics.items():
            all_metrics[f"depth_{key}"] = value.item()

    return all_metrics
