import torch
from typing import Optional, Tuple

def calculate_depth_scale(depthmaps, valid_mask, eps=1e-3, mode='mean', scale=2.0):
    """
    depthmaps: B, S, H, W
    valid_mask: B, S, H, W
    """
    dist = depthmaps.abs()

    if mode == 'mean':
        dist_sum = (dist * valid_mask).sum(dim=[1,2,3])
        valid_count = valid_mask.sum(dim=[1,2,3])

        avg_scale = (dist_sum / (valid_count + eps))
    else:
        B = dist.shape[0]
        flat_dist = dist.reshape(B, -1)
        valid_mask = valid_mask.reshape(B, -1).bool()
        nan_fill_value = torch.tensor(float('nan'), device=flat_dist.device, dtype=flat_dist.dtype)
        masked_flat_dist = torch.where(valid_mask, flat_dist, nan_fill_value)

        # Calculate median along the flattened dimension (N) for each batch item.
        # torch.nanmedian ignores NaN values and returns a namedtuple (values, indices).
        # We only need the median values.
        median_scales = torch.nanmedian(masked_flat_dist, dim=-1).values

        # Handle cases where a batch item might have had no valid pixels.
        # In such cases, nanmedian would return NaN. Replace these NaNs with eps.
        avg_scale = torch.nan_to_num(median_scales, nan=eps)
        
    # Apply clamping similar to the 'mean' mode for consistency and to ensure sensible scale values.
    avg_scale = avg_scale.clamp(min=eps, max=1e3)
    # shape ([B])

    return avg_scale

def normalize_pointcloud_invariant(
    pts3d: torch.Tensor, 
    valid_mask: torch.Tensor, 
    c2w_poses: Optional[torch.Tensor] = None, 
    eps: float = 1e-6, 
    return_pts: bool = False
) -> Tuple[torch.Tensor, ...]:
    """
    Normalizes a point cloud (and optionally c2w poses) to be translation 
    and scale invariant.
    
    Transformation: P_new = (P_old - centroid) / scalar
    
    Args:
        pts3d (torch.Tensor): The 3D points tensor. 
                              Shape (B, S, H, W, 3).
        valid_mask (torch.Tensor): A boolean or float mask.
                                   Shape (B, S, H, W).
        c2w_poses (Optional[torch.Tensor]): The original c2w poses.
                                            Shape (B, S, 3, 4) or (B, S, 4, 4).
        eps (float): A small epsilon value for numerical stability.
        return_pts (bool): If True, returns normalized entities and transform params.
                           If False, returns only the transform params.
    
    Returns:
        If return_pts is True:
            pts3d_normalized (torch.Tensor): (B, S, H, W, 3)
            c2w_poses_normalized (Optional[torch.Tensor]): (B, S, 3, 4 or 4, 4)
            centroid (torch.Tensor): The calculated centroid. (B, 3)
            inv_scale (torch.Tensor): The calculated inv avg scale. (B,)
        If return_pts is False:
            centroid (torch.Tensor): (B, 3)
            inv_scale (torch.Tensor): (B,)
    """
    
    # --- 1. Calculate Centroid and Scale (from point cloud) ---
    B, S, H, W, C = pts3d.shape
    device = pts3d.device
    
    valid_mask_float = valid_mask.float()
    
    # valid_count shape: (B,)
    valid_count = valid_mask_float.sum(dim=[1, 2, 3]).clamp(min=eps)

    # valid_mask_expanded shape: (B, S, H, W, 1)
    valid_mask_expanded = valid_mask_float.unsqueeze(-1)
    
    # pts3d_sum shape: (B, 3)
    pts3d_sum = (pts3d * valid_mask_expanded).sum(dim=[1, 2, 3])
    
    # centroid shape: (B, 3)
    centroid = pts3d_sum / valid_count.view(B, 1).clamp(min=eps)
    
    # Center the point cloud
    # centroid_broadcast_pts shape: (B, 1, 1, 1, 3)
    centroid_broadcast_pts = centroid.view(B, 1, 1, 1, 3)
    pts3d_centered = pts3d - centroid_broadcast_pts

    # Calculate average scale
    # dist_centered shape: (B, S, H, W)
    dist_centered = torch.linalg.norm(pts3d_centered + 1e-8, dim=-1)
    
    # dist_sum shape: (B,)
    dist_sum = (dist_centered * valid_mask_float).sum(dim=[1, 2, 3])
    
    # avg_scale shape: (B,)
    inv_scale = 1.0 / (dist_sum / valid_count.clamp(min=eps)).clamp(min=eps, max=1e5)
    
    # --- 2. Handle 'return_pts=False' case ---
    if not return_pts:
        return centroid, inv_scale

    # --- 3. Normalize 3D Points ---
    
    # .view() reshapes avg_scale for broadcasting: (B,) -> (B, 1, 1, 1, 1)
    pts3d_normalized = pts3d_centered * inv_scale.view(B, 1, 1, 1, 1)

    # --- 4. Normalize c2w Poses (if provided) ---
    
    c2w_poses_normalized = None
    if c2w_poses is not None:        
        # Decompose Original Poses
        R_e = c2w_poses[..., :3, :3]  # Shape (B, S, 3, 3)
        t_e = c2w_poses[..., :3, 3]   # Shape (B, S, 3)

        # --- Apply Pose Transformation ---        
        R_aligned = R_e 
        
        # t'_e = s * (t_e - centroid)
        # (B, S, 3) - (B, 1, 3) -> (B, S, 3)
        t_e_centered = t_e - centroid.view(B, 1, 3)
        
        # (B, 1, 1) * (B, S, 3) -> (B, S, 3)
        t_aligned = t_e_centered * inv_scale.view(B, 1, 1)

        # Reconstruct the Aligned Poses
        c2w_poses_normalized = c2w_poses.clone()
        c2w_poses_normalized[..., :3, :3] = R_aligned
        c2w_poses_normalized[..., :3, 3] = t_aligned
            
    return pts3d_normalized, c2w_poses_normalized, centroid, inv_scale

def normalize_pointcloud_vggt(pts3d, valid_mask, eps=1e-3, return_pts=True):
    """
    Normalizes a point cloud by its average distance from the origin.

    Args:
        pts3d (torch.Tensor): The 3D points tensor of shape (B, S, H, W, 3).
        valid_mask (torch.Tensor): A boolean or float mask of shape (B, S, H, W) 
                                   indicating which points are valid.
        eps (float): A small epsilon value to prevent division by zero.
        return_pts (bool): If True, returns the normalized points and the scale factor.
                           If False, returns only the scale factor.
    
    Returns:
        If return_pts is True:
            torch.Tensor: The normalized 3D points.
            torch.Tensor: The calculated average scale factor for each item in the batch.
        If return_pts is False:
            torch.Tensor: The calculated average scale factor for each item in the batch.
    """
    # Calculate the Euclidean distance of each point from the origin
    dist = torch.linalg.norm(pts3d, dim=-1)

    # Sum the distances of only the valid points
    dist_sum = (dist * valid_mask).sum(dim=[1, 2, 3])
    # Count the number of valid points
    valid_count = valid_mask.sum(dim=[1, 2, 3])

    # Calculate the average scale (average distance)
    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e5)

    if return_pts:
        # Normalize the point cloud by the average scale
        # .view() reshapes avg_scale for broadcasting: (B,) -> (B, 1, 1, 1, 1)
        pts3d_normalized = pts3d / avg_scale.view(-1, 1, 1, 1, 1)
        return pts3d_normalized, avg_scale
    else:
        return avg_scale

def normalize_pointclouds(all_pr_pts, all_gt_pts, mask_gt, not_metric_mask=None):
    """
    Normalizes predicted and ground truth point clouds to the same scale.

    The scale is determined by the ground truth points unless otherwise specified
    by not_metric_mask. This version returns tensors directly.

    Args:
        all_pr_pts (torch.Tensor): Predicted 3D points tensor of shape (B, S, H, W, 3).
        all_gt_pts (torch.Tensor): Ground truth 3D points tensor of shape (B, S, H, W, 3).
        mask_gt (torch.Tensor): The validity mask for the ground truth points.
        not_metric_mask (torch.Tensor, optional): A boolean tensor of shape (B,) indicating
                                                  which samples should be scaled by their own
                                                  prediction's metric instead of the GT metric.
    
    Returns:
        torch.Tensor: Normalized predicted points tensor of shape (B, S, H, W, 3).
        torch.Tensor: Normalized ground truth points tensor of shape (B, S, H, W, 3).
        torch.Tensor: The normalization factors used for the predicted points.
        torch.Tensor: The normalization factors derived from the ground truth points.
    """
    # 1. Normalize the ground truth points and get the GT scale factor
    all_gt_pts_normalized, norm_factor_gt = normalize_pointcloud_vggt(all_gt_pts, mask_gt)

    # 2. By default, use the GT scale factor for the predicted points
    norm_factor_pr = norm_factor_gt.clone()
    
    # 3. For samples where GT metric is not available, calculate scale from the prediction
    if not_metric_mask is not None and not_metric_mask.sum() > 0:
        norm_factor_pr[not_metric_mask] = normalize_pointcloud_vggt(
            all_pr_pts[not_metric_mask], mask_gt[not_metric_mask], return_pts=False
        )

    # 4. Normalize the entire predicted point cloud tensor.
    all_pr_pts_normalized = all_pr_pts / norm_factor_pr.view(-1, 1, 1, 1, 1)
    
    return all_pr_pts_normalized, all_gt_pts_normalized, norm_factor_pr, norm_factor_gt

def normalize_pr_pointcloud(all_pr_pts, norm_factor_gt, mask_gt, not_metric_mask=None):
    """
    Normalizes predicted and ground truth point clouds to the same scale.

    The scale is determined by the ground truth points unless otherwise specified
    by not_metric_mask. This version returns tensors directly.

    Args:
        all_pr_pts (torch.Tensor): Predicted 3D points tensor of shape (B, S, H, W, 3).
        norm_factor_gt (torch.Tensor): The normalization factors derived from the ground truth points.
        mask_gt (torch.Tensor): The validity mask for the ground truth points.
        not_metric_mask (torch.Tensor, optional): A boolean tensor of shape (B,) indicating
                                                  which samples should be scaled by their own
                                                  prediction's metric instead of the GT metric.
    
    Returns:
        torch.Tensor: Normalized predicted points tensor of shape (B, S, H, W, 3).
        torch.Tensor: The normalization factors used for the predicted points.
    """

    # 2. By default, use the GT scale factor for the predicted points
    norm_factor_pr = norm_factor_gt.clone()
    
    # 3. For samples where GT metric is not available, calculate scale from the prediction
    if not_metric_mask is not None and not_metric_mask.sum() > 0:
        norm_factor_pr[not_metric_mask] = normalize_pointcloud_vggt(
            all_pr_pts[not_metric_mask], mask_gt[not_metric_mask], return_pts=False
        )

    # 4. Normalize the entire predicted point cloud tensor.
    all_pr_pts_normalized = all_pr_pts / norm_factor_pr.view(-1, 1, 1, 1, 1)
    
    return all_pr_pts_normalized, norm_factor_pr

def normalize_depth_cam_extrinsics(
    norm_factor: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    cam_points: Optional[torch.Tensor] = None,
    extrinsics: Optional[torch.Tensor] = None,
    global_points3d: Optional[torch.Tensor] = None, 
    inv_scale: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], ...]:
    """
    Normalizes depths, camera points, global points, and camera extrinsics by a given scale factor.

    This function scales down the metric values by dividing them by the provided `norm_factor`.
    For extrinsics, only the translation component is scaled, as rotation is scale-invariant.

    Args:
        norm_factor (torch.Tensor): A batch-wise normalization factor of shape `(B,)`.
        depths (Optional[torch.Tensor]): Depth maps of shape `(B, S, H, W)`.
        cam_points (Optional[torch.Tensor]): 3D points in camera coordinates. 
                                            Shape `(B, ..., 3)`.
        global_points3d (Optional[torch.Tensor]): 3D points in world coordinates.
                                                  Shape `(B, ..., 3)`.
        extrinsics (Optional[torch.Tensor]): Camera extrinsic matrices.
                                             Shape `(B, S, 4, 4)`.

    Returns:
        Tuple containing:
        - `normalized_depths` (torch.Tensor | None): Scaled depths.
        - `normalized_cam_points` (torch.Tensor | None): Scaled camera points.
        - `normalized_extrinsics` (torch.Tensor | None): Extrinsics with scaled translation.
        - `normalized_global_points3d` (torch.Tensor | None): Scaled global points.
    """
    # Initialize return values
    normalized_depths = None
    normalized_cam_points = None
    normalized_global_points3d = None 
    normalized_extrinsics = None

    if inv_scale is None:
        inv_scale = 1.0 / norm_factor

    # 1. Normalize depths 📏
    # Reshapes the norm_factor from (B,) to (B, 1, 1, 1) to enable broadcasting.
    if depths is not None:
        normalized_depths = depths * inv_scale.view([inv_scale.shape[0]] + [1] * (depths.ndim - 1))

    # 2. Normalize camera points
    if cam_points is not None:
        normalized_cam_points = cam_points * inv_scale.view([inv_scale.shape[0]] + [1] * (cam_points.ndim - 1))

    # 3. Normalize global points
    if global_points3d is not None:
        normalized_global_points3d = global_points3d * inv_scale.view([inv_scale.shape[0]] + [1] * (global_points3d.ndim - 1))

    # 4. Normalize camera extrinsics 📷
    if extrinsics is not None:
        # It is crucial to clone to avoid modifying the original tensor, which can cause side effects.
        normalized_extrinsics = extrinsics.clone()
        
        # Apply scaling ONLY to the translation part `t` of the [R|t] matrix.
        # The translation vector is in the last column of the top 3 rows.
        normalized_extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * inv_scale.view(-1, 1, 1)

    return normalized_depths, normalized_cam_points, normalized_extrinsics, normalized_global_points3d