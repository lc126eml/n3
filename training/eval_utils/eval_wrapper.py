from typing import Dict, Any

from vggt.utils.pose_enc import intri_to_fov_encoding

from .metric_utils.camera_metric import (
    compute_absolute_pose_error_angle,
    compute_batched_all_pairs_relative_metrics,
)
from .metric_utils.depth_metric import calculate_depth_metrics_optimized
from .metric_utils.recon_metric import (
    calculate_batched_nearest_neighbor_pointcloud_metrics,
    calculate_corresponding_points_error_torch_optimized,
)


def _calculate_reconstruction_metrics(
    pred_points,
    gt_points,
    valid_mask,
    prefix: str,
    nn_metrics_enabled: bool,
    nn_threshold: float,
    nn_max_points: int,
) -> Dict[str, float]:
    if valid_mask is not None:
        pred_points_flat = pred_points[valid_mask]
        gt_points_flat = gt_points[valid_mask]
    else:
        pred_points_flat = pred_points.reshape(-1, 3)
        gt_points_flat = gt_points.reshape(-1, 3)

    metrics = calculate_corresponding_points_error_torch_optimized(
        gt_points_flat,
        pred_points_flat,
    )
    if nn_metrics_enabled:
        metrics.update(
            calculate_batched_nearest_neighbor_pointcloud_metrics(
                gt_points,
                pred_points,
                valid_mask=valid_mask,
                threshold=nn_threshold,
                max_points=nn_max_points,
            )
        )

    return {f"{prefix}{key}": value.item() for key, value in metrics.items()}


def eval_batch(
    y_hat: Dict[str, Any],
    batch: Dict[str, Any],
    metrics_conf: Dict[str, Any],
    data_keys: Dict[str, Any],
    pred_data_keys: Dict[str, Any],
    pose_convention: str = "c2w",
) -> Dict[str, Any]:
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
        pred_poses, gt_poses = get_data(
            pred_data_keys.get('extrinsics', data_keys.get('extrinsics')),
            data_keys.get('extrinsics'),
        )

        if cam_conf.get('abs_err'):
            results, rot_errors_deg, trans_angle_errors_deg, trans_errors = compute_absolute_pose_error_angle(gt_poses, pred_poses)
            all_metrics.update(results)
        if cam_conf.get('auc') or cam_conf.get('rel_err'):
            rel_metrics = compute_batched_all_pairs_relative_metrics(
                gt_poses,
                pred_poses,
                pose_convention=pose_convention,
                include_auc=cam_conf.get('auc'),
                max_threshold_deg=30,
            )
            for key, value in rel_metrics.items():
                all_metrics[key] = value

    # --- FoV Metric ---
    if metrics_conf.get('intrinsics', {}).get('enabled'):
        intrinsics_conf = metrics_conf['intrinsics']
        pred_pose_enc = y_hat.get('pose_enc')
        gt_intrinsics = batch.get(data_keys.get('intrinsics'))

        if pred_pose_enc is not None and gt_intrinsics is not None:
            image_size_hw = intrinsics_conf.get('image_size_hw')
            if image_size_hw is None and batch.get('img') is not None:
                image_size_hw = batch['img'].shape[-2:]
            if image_size_hw is None:
                raise ValueError("image_size_hw is required to calculate the FoV metric")

            pred_fov = pred_pose_enc[..., 7:]
            gt_fov = intri_to_fov_encoding(gt_intrinsics, image_size_hw)
            all_metrics['fov_error'] = (pred_fov - gt_fov).abs().mean().item()

    # --- Depth Metrics ---
    if metrics_conf.get('depth', {}).get('enabled'):
        depth_conf = metrics_conf['depth']
        pred_depth, gt_depth = get_data(pred_data_keys.get('depths'), data_keys.get('depths'))
        valid_mask = batch.get(data_keys.get('valid_mask'))

        all_metrics.update(
            calculate_depth_metrics_optimized(
                gt_depth,
                pred_depth,
                valid_mask,
                include_sq_rel=depth_conf.get('sq_rel', False),
                include_rmse=depth_conf.get('rmse', False),
                include_log_rmse=depth_conf.get('log_rmse', False),
                include_delta=depth_conf.get('delta', False),
                align=depth_conf.get('align', 'median'),
            )
        )

    # --- Reconstruction Metrics ---
    recon_conf = metrics_conf.recon
    nn_metrics_enabled = recon_conf.get('nn_metrics', False)
    nn_threshold = recon_conf.get('nn_threshold', 0.05)
    nn_max_points = recon_conf.get('nn_max_points', 200000)

    world_points_key = pred_data_keys.get('world_points')
    if recon_conf.get('pts_err') and world_points_key in y_hat:
        gt_points = batch.get(data_keys.get('world_points'))
        valid_mask = batch.get(data_keys.get('valid_mask'))
        aligned_world_points_key = pred_data_keys.get(
            'aligned_world_points', 'aligned_world_points'
        )
        pred_points_key = (
            aligned_world_points_key
            if aligned_world_points_key in y_hat
            else world_points_key
        )
        all_metrics.update(
            _calculate_reconstruction_metrics(
                y_hat[pred_points_key],
                gt_points,
                valid_mask,
                prefix='',
                nn_metrics_enabled=nn_metrics_enabled,
                nn_threshold=nn_threshold,
                nn_max_points=nn_max_points,
            )
        )

    global_from_cam_key = pred_data_keys.get('global_from_cam')
    if recon_conf.get('from_cam_err') and global_from_cam_key in y_hat:
        gt_points = batch.get(data_keys.get('world_points'))
        valid_mask = batch.get(data_keys.get('valid_mask'))
        all_metrics.update(
            _calculate_reconstruction_metrics(
                y_hat[global_from_cam_key],
                gt_points,
                valid_mask,
                prefix='cam_',
                nn_metrics_enabled=nn_metrics_enabled,
                nn_threshold=nn_threshold,
                nn_max_points=nn_max_points,
            )
        )

    global_from_depth_key = pred_data_keys.get('global_from_depth')
    if recon_conf.get('from_depth_err') and global_from_depth_key in y_hat:
        gt_points = batch.get(data_keys.get('world_points'))
        valid_mask = batch.get(data_keys.get('valid_mask'))
        all_metrics.update(
            _calculate_reconstruction_metrics(
                y_hat[global_from_depth_key],
                gt_points,
                valid_mask,
                prefix='depth_',
                nn_metrics_enabled=nn_metrics_enabled,
                nn_threshold=nn_threshold,
                nn_max_points=nn_max_points,
            )
        )

    return all_metrics
