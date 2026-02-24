import torch
import roma
from typing import Optional, Tuple, Dict, Union

def align_rotation_only_torch(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    pred_conf: Optional[torch.Tensor] = None,
    conf_threshold: Optional[float] = None,
    conf_percentage: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes the optimal Rotation matrix to align pred_points to gt_points.
    Assumes Scale=1.0 and Translation=0.0 (Rotation around Origin).

    Args:
        pred_points: (B, S, W, H, 3) or (B, N, 3) Source points (PRE-NORMALIZED)
        gt_points:   (B, S, W, H, 3) or (B, N, 3) Target points (PRE-NORMALIZED)
        valid_mask:  (B, S, W, H) or (B, N) Optional boolean mask
        pred_conf:   (B, S, W, H) or (B, N) Optional confidence scores for filtering
        conf_threshold: Float, drops points with conf < threshold
        conf_percentage: Float (0.0-1.0), drops points not in top k% confidence

    Returns:
        R: (B, 3, 3) Rotation matrices
    """
    B = pred_points.shape[0]
    
    # 1. Efficient Reshape: (B, 3, N)
    # (B, N, 3) -> (B, 3, N) layout is optimal for the upcoming H matmul
    pts1 = pred_points.reshape(B, -1, 3).transpose(1, 2)
    pts2 = gt_points.reshape(B, -1, 3).transpose(1, 2)
    N = pts1.shape[2]

    # 2. Apply Mask
    if valid_mask is not None:
        # Flatten mask to (B, 1, N)
        mask = valid_mask.reshape(B, 1, -1).float()
    else:
        mask = torch.ones((B, 1, N), device=pred_points.device, dtype=pred_points.dtype)

    # 3. Update Mask based on Confidence (Hard Filtering)
    if pred_conf is not None:
        conf_flat = pred_conf.reshape(B, 1, -1) # (B, 1, N)
        
        # A. Apply Threshold
        if conf_threshold is not None:
            # Hard cut: multiply mask by 0 where conf <= threshold
            mask = mask * (conf_flat > conf_threshold).float()
            
        # B. Apply Percentage (Keep Top k%)
        elif conf_percentage is not None:
            # Calculate the quantile threshold per batch
            # 0.9 percentage -> we want top 90% -> cutoff at 0.1 quantile
            q = max(0.0, 1.0 - conf_percentage)
            
            # Use float32 for quantile to ensure GPU support
            # (B, 1, 1) threshold value
            thresh_val = torch.quantile(conf_flat.float(), q, dim=2, keepdim=True).to(dtype=pred_points.dtype)
            
            # Hard cut: multiply mask by 0 where conf < quantile
            mask = mask * (conf_flat >= thresh_val).float()

    # 4. Apply Mask to Points
    # Points excluded by mask become (0,0,0) and contribute nothing to H
    pts1 = pts1 * mask
    pts2 = pts2 * mask

    # 3. Covariance H = P1 @ P2^T
    # (B, 3, N) @ (B, N, 3) -> (B, 3, 3)
    # float32 is safe here because inputs are pre-normalized
    H = pts1 @ pts2.transpose(1, 2)

    # 4. SVD
    # Add a small epsilon to the diagonal for numerical stability
    # H_sq_sum = (H**2).sum(dim=(1, 2), keepdim=True)
    # is_ill_conditioned = (H_sq_sum < 1e-8) # 1e-12 is (1e-6)**2

    # # Perturbation: Epsilon * Identity
    # # We use 1e-5. Note: 1e-5 squared is 1e-10, which is > 1e-12, ensuring stability.
    # eps = 1e-5
    # perturbation = torch.eye(3, device=H.device, dtype=H.dtype).view(1, 3, 3) * eps

    # # Apply ONLY to ill-conditioned matrices
    # H_stable = H + (is_ill_conditioned.float() * perturbation)    
    # H_stable = H + torch.eye(H.shape[-1], device=H.device, dtype=H.dtype) * 1e-6
    perturbation = torch.diag(torch.tensor([1e-6, 2e-6, 3e-6], device=H.device, dtype=H.dtype)).view(1, 3, 3)  
    H_stable = H + perturbation 
    try:
        U, _, Vh = torch.linalg.svd(H_stable.to(torch.float32))
    except Exception as e:
        H_sq_sum = (H**2).sum(dim=(1, 2), keepdim=True)
        print(f"Error in roma.rigid_points_registration: {e}", f"{H_sq_sum=}")
        breakpoint()
        raise e
    # 5. Compute Initial Rotation R = V @ U.T
    # R = Vh.T @ U.T = (U @ Vh).T
    R = (U @ Vh).transpose(-2, -1)

    # 6. Reflection Fix (Rank-1 Update)
    # R_final = R - 2 * (v3 @ u3.T) where det(R) < 0
    det = torch.det(R.to(torch.float32))
    
    # Create mask: 1.0 if reflection needed, 0.0 otherwise
    # We avoid 'if mask.any()' to prevent CPU-GPU sync stall
    neg_det_mask = (det < 0).float().view(B, 1, 1)

    # Extract 3rd singular vectors (last column of V, last column of U)
    # V = Vh.T -> last row of Vh
    v3 = Vh[:, 2, :].unsqueeze(-1) # (B, 3, 1)
    u3 = U[:, :, 2].unsqueeze(-1)  # (B, 3, 1)

    # Compute correction term: 2 * (v3 @ u3.T)
    # Outer product: (B, 3, 1) @ (B, 1, 3) -> (B, 3, 3)
    correction = 2.0 * (v3 @ u3.transpose(-2, -1))

    # Apply correction masked. 
    # For valid rotations, neg_det_mask is 0, so R remains unchanged.
    R_opt = R - (neg_det_mask * correction)

    return R_opt
def align_pred_to_gt_torch_batch_roma(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    pred_conf: Optional[torch.Tensor] = None,
    conf_threshold: Optional[float] = None,
    conf_percentage: Optional[float] = None,
    with_scale: bool = True,
    return_points: bool = True,
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Processes a batch of predicted 3D points and aligns each to its corresponding
    ground truth points using roma.rigid_points_registration.

    This function iterates over the batch dimension, filters points based on
    validity and confidence, computes the optimal transformation using roma,
    and then aggregates the results.

    Args:
        pred_points (torch.Tensor): Batched predicted 3D points. Shape (B, S, W, H, 3).
        gt_points (torch.Tensor): Batched ground truth 3D points. Shape (B, S, W, H, 3).
        valid_mask (Optional[torch.Tensor]): Batched boolean mask for valid points.
                                            Shape (B, S, W, H). Defaults to None.
        pred_conf (Optional[torch.Tensor]): Batched confidence scores.
                                            Shape (B, S, W, H). Defaults to None.
        conf_threshold (Optional[float]): A confidence score below which points are ignored.
        conf_percentage (Optional[float]): The top percentage of confident points to use for alignment.
                                           For example, 0.9 means use the top 90% of points.
        with_scale (bool): If True, compute and apply scaling. If False,
                           only compute rotation and translation (scale=1.0)
                           and return None for 'scale' in the output dict.

    Returns:
        Tuple[torch.Tensor, Dict[str, Union[list, torch.Tensor]]]: A tuple containing:
            - batch_aligned_points (torch.Tensor): The batch of transformed predicted points.
            - batch_transform_params (Dict): A dictionary with aggregated
                                            'rotation' (tensor), 'translation' (tensor),
                                            and 'scale' (tensor or None).
    """
    if pred_points.dim() != 5:
        raise ValueError(f"Expected pred_points to be a 5D tensor (B, S, W, H, 3), but got {pred_points.dim()}D.")

    batch_size = pred_points.shape[0]
    original_shape = pred_points.shape[1:]  # Keep original (S, W, H, 3) shape for output
    original_dtype = pred_points.dtype

    # Lists to store results from each sample in the batch
    aligned_points_list = []
    scales, rotations, translations = [], [], []

    for i in range(batch_size):
        # --- 1. Slice the batch and flatten points from (S,W,H,3) to (N,3) ---
        pred_sample = pred_points[i].reshape(-1, 3)
        gt_sample = gt_points[i].reshape(-1, 3)

        # --- 2. Build a mask to select points for registration ---
        # Start with a mask of all valid points
        if valid_mask is not None:
            mask = valid_mask[i].reshape(-1)
        else:
            mask = torch.ones(pred_sample.shape[0], dtype=torch.bool, device=pred_points.device)

        # Refine mask and get weights from confidence scores
        weights = None
        if pred_conf is not None:
            conf_sample = pred_conf[i].reshape(-1)

            # Apply confidence threshold
            if conf_threshold is not None:
                mask = mask & (conf_sample > conf_threshold)

            # Apply confidence percentage
            elif conf_percentage is not None:
                valid_confs = conf_sample[mask]
                if valid_confs.numel() > 0:
                    # conf_percentage is expected as 0-100.
                    conf_frac = min(max(conf_percentage, 0.0), 100.0) / 100.0
                    # Calculate the threshold for the top percentage
                    k = max(0.0, 1.0 - conf_frac)
                    percentile_val = torch.quantile(valid_confs.to(torch.float32), k)
                    mask = mask & (conf_sample >= percentile_val)
            
            # Final weights are the confidence scores of the selected points
            weights = conf_sample[mask]
            weights = torch.clamp(weights, min=0.0) + 1e-6
        # Select the points to be used for computing the transformation
        pts1_for_reg = pred_sample[mask]
        pts2_for_reg = gt_sample[mask]

        # --- 3. Compute transformation using roma ---
        # Need at least 3 points to determine a rigid transformation
        if pts1_for_reg.shape[0] < 3:
            s = torch.tensor(1.0, device=pred_points.device, dtype=original_dtype)
            R = torch.eye(3, device=pred_points.device, dtype=original_dtype)
            T = torch.zeros(3, device=pred_points.device, dtype=original_dtype)
            aligned_sample = pred_sample
        else:
            try:
                # roma expects batched input, so we add a temporary batch dimension
                with torch.amp.autocast("cuda", enabled=False):
                    # Store result in one variable first
                    transform_result = roma.rigid_points_registration(
                        pts1_for_reg.unsqueeze(0).to(torch.float32),
                        pts2_for_reg.unsqueeze(0).to(torch.float32),
                        weights=weights.unsqueeze(0).to(torch.float32) if (weights is not None and weights.numel() > 0) else None,
                        compute_scaling=with_scale,
                    )
            except Exception as e:
                mask_flat = mask.reshape(mask.shape[0], -1)
                print(f"Error in roma.rigid_points_registration: {e}", f"valid points {mask_flat.sum(dim=1)}", f"total {mask_flat.size(1)}")
                breakpoint()
                raise e
            
            # Unpack results based on with_scale
            if with_scale:
                R, T, s = transform_result
                R, T = R.squeeze(0), T.squeeze(0)
                s = s.squeeze(0)
                if return_points:
                    aligned_sample = s * (pred_sample @ R.T) + T
            else:
                R, T = transform_result
                R, T = R.squeeze(0), T.squeeze(0)
                if return_points:
                    aligned_sample = (pred_sample @ R.T) + T

        if return_points:        
            # Append results to lists
            aligned_points_list.append(aligned_sample.reshape(original_shape))
        
        # Only store scale if it was computed
        if with_scale:
            scales.append(s)
            
        rotations.append(R)
        translations.append(T)

    # --- 5. Aggregate results into batch tensors ---
    batch_aligned_points = None
    if return_points:
        batch_aligned_points = torch.stack(aligned_points_list, dim=0)
    batch_transform_params = {
        'scale': torch.stack(scales, dim=0) if with_scale else None,
        'rotation': torch.stack(rotations, dim=0),
        'translation': torch.stack(translations, dim=0),
    }

    return batch_aligned_points, batch_transform_params

def umeyama_alignment_torch(x: torch.Tensor, y: torch.Tensor, with_scale: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the optimal similarity transformation between two sets of corresponding points using PyTorch.

    This function finds the scale, rotation, and translation that minimizes the root-mean-square
    error between the transformed points of x and the points of y.

    Args:
        x (torch.Tensor): The first set of points, shape (N, 3).
        y (torch.Tensor): The second set of corresponding points, shape (N, 3).
        with_scale (bool): Whether to estimate the scale factor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - s (torch.Tensor): The scale factor (0-dim tensor).
            - R (torch.Tensor): The 3x3 rotation matrix.
            - t (torch.Tensor): The 3D translation vector.
    """
    if x.device != y.device:
        raise ValueError("Input tensors must be on the same device.")
    device = x.device
    x = x.to(torch.float32)
    y = y.to(torch.float32)

    # Calculate centroids
    mu_x = x.mean(dim=0)
    mu_y = y.mean(dim=0)

    # Center the points
    x_centered = x - mu_x
    y_centered = y - mu_y

    # Calculate the covariance matrix
    # Sigma = x_centered.T @ y_centered
    Sigma = y_centered.T @ x_centered

    # Add a small value to the diagonal for numerical stability of SVD backward
    Sigma = Sigma + torch.eye(Sigma.shape[-1], device=Sigma.device, dtype=Sigma.dtype) * 1e-9

    # Perform Singular Value Decomposition (SVD), casting to float32 for compatibility
    U, D, Vt = torch.linalg.svd(Sigma.to(torch.float32))
    
    # Ensure a right-handed coordinate system
    S = torch.eye(x.shape[1], device=device, dtype=x.dtype)
    # Cast to float32 for determinant calculation
    if torch.linalg.det(U.to(torch.float32)) * torch.linalg.det(Vt.to(torch.float32)) < 0:
        S[-1, -1] = -1
        
    # Calculate the rotation matrix
    R = U @ S @ Vt

    # Calculate the scale factor
    if with_scale:
        var_x = torch.sum(x_centered ** 2)
        s = torch.sum(D * torch.diag(S)) / var_x if var_x > 1e-8 else torch.tensor(1.0, device=device)
    else:
        s = torch.tensor(1.0, device=device)

    # Calculate the translation vector
    t = mu_y - s * (R @ mu_x)

    return s, R, t

def align_pred_to_gt_torch(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    pred_conf: Optional[torch.Tensor] = None,
    conf_threshold: Optional[float] = None,
    conf_percentage: Optional[float] = None,
    with_scale: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Union[float, torch.Tensor]]]:
    """
    Aligns predicted 3D points to ground truth points using the Umeyama algorithm (PyTorch version).

    The function filters points based on optional masks/thresholds, calculates the
    optimal similarity transformation on this subset, and applies it to the
    *original, unfiltered* predicted points. All computations are done on the tensor's device (CPU or GPU).

    Args:
        pred_points (torch.Tensor): Predicted 3D points. Shape (S, H, W, 3).
        gt_points (torch.Tensor): Ground truth 3D points. Shape (S, H, W, 3).
        valid_mask (Optional[torch.Tensor]): Boolean mask for valid points. If None, all
                                              points are considered valid. Shape (S, H, W).
        pred_conf (Optional[torch.Tensor]): Confidence scores for predicted points.
                                             Shape (S, H, W). Defaults to None.
        conf_threshold (Optional[float]): Minimum absolute confidence to consider a point.
                                          Defaults to None.
        conf_percentage (Optional[float]): Top percentage of confident points to use.
                                           Value between 0-100. Defaults to None.

    Returns:
        Tuple[torch.Tensor, Dict[str, Union[float, torch.Tensor]]]: A tuple containing:
            - aligned_pred_points (torch.Tensor): Transformed predicted points, on the same device.
            - transform_params (Dict): A dictionary with 'scale', 'rotation', and 'translation'.
    """
    device = pred_points.device
    S, H, W, _ = pred_points.shape
    
    # --- 2. Reshape data for processing ---
    pred_points_flat = pred_points.reshape(-1, 3)
    gt_points_flat = gt_points.reshape(-1, 3)

    # --- 3. Create a combined filter mask ---
    if valid_mask is None:
        combined_mask = torch.full((gt_points_flat.shape[0],), True, dtype=torch.bool, device=device)
    else:
        combined_mask = valid_mask.flatten().bool()

    if pred_conf is not None:
        flat_conf = pred_conf.flatten()
        if conf_threshold is not None:
            combined_mask = combined_mask & (flat_conf >= conf_threshold)
        elif conf_percentage is not None:
            if not (0 < conf_percentage <= 100):
                raise ValueError("conf_percentage must be between 0 and 100.")
            current_valid_conf_scores = flat_conf[combined_mask]
            if current_valid_conf_scores.numel() > 0:
                percentile_value = torch.quantile(current_valid_conf_scores, (100 - conf_percentage) / 100.0)
                combined_mask = combined_mask & (flat_conf >= percentile_value)

    # --- 4. Filter points for finding the alignment ---
    pred_for_align = pred_points_flat[combined_mask]
    gt_for_align = gt_points_flat[combined_mask]

    if pred_for_align.shape[0] < 3:
        print("Warning: Fewer than 3 valid points. Returning identity transformation.")
        identity_transform = {
            'scale': torch.tensor(1.0, device=device),
            'rotation': torch.eye(3, device=device),
            'translation': torch.zeros(3, device=device)
        }
        return pred_points, identity_transform
        
    # --- 5. Use Umeyama algorithm to find the transformation parameters ---
    scale, rotation, translation = umeyama_alignment_torch(x=pred_for_align, y=gt_for_align, with_scale=with_scale)

    # --- 6. Apply transformation to the *original, unfiltered* predicted points ---
    if with_scale:
        aligned_pred_points_flat = (scale * pred_points_flat @ rotation.T) + translation
    else:
        aligned_pred_points_flat = (pred_points_flat @ rotation.T) + translation

    # --- 7. Reshape the aligned points back to the original format ---
    aligned_pred_points = aligned_pred_points_flat.reshape(S, H, W, 3)

    transform_params = {'scale': scale, 'rotation': rotation, 'translation': translation}
    return aligned_pred_points, transform_params


def align_pred_to_gt_torch_batch(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    pred_conf: Optional[torch.Tensor] = None,
    conf_threshold: Optional[float] = None,
    conf_percentage: Optional[float] = None,
    with_scale: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Processes a batch of predicted 3D points and aligns each to its corresponding
    ground truth points using the Umeyama algorithm.

    This function iterates over the batch dimension and calls `align_pred_to_gt_torch`
    for each sample, then aggregates the results.

    Args:
        pred_points (torch.Tensor): Batched predicted 3D points. Shape (B, S, W, H, 3).
        gt_points (torch.Tensor): Batched ground truth 3D points. Shape (B, S, W, H, 3).
        valid_mask (Optional[torch.Tensor]): Batched boolean mask for valid points.
                                              Shape (B, S, W, H). Defaults to None.
        pred_conf (Optional[torch.Tensor]): Batched confidence scores.
                                             Shape (B, S, W, H). Defaults to None.
        conf_threshold (Optional[float]): See `align_pred_to_gt_torch`.
        conf_percentage (Optional[float]): See `align_pred_to_gt_torch`.

    Returns:
        Tuple[torch.Tensor, Dict[str, Union[list, torch.Tensor]]]: A tuple containing:
            - batch_aligned_points (torch.Tensor): The batch of transformed predicted points.
            - batch_transform_params (Dict): A dictionary with aggregated 'scale' (tensor),
                                             'rotation' (tensor), and 'translation' (tensor).
    """
    if pred_points.dim() != 5:
        raise ValueError(f"Expected pred_points to be a 5D tensor (B, S, W, H, 3), but got {pred_points.dim()}D.")
    
    batch_size = pred_points.shape[0]
    # print(pred_points.shape, gt_points.shape)
    
    # Lists to store results from each sample in the batch
    aligned_points_list = []
    transform_params_list = []

    for i in range(batch_size):
        # Slice the batch dimension for all inputs
        pred_points_sample = pred_points[i]
        gt_points_sample = gt_points[i]
        
        valid_mask_sample = valid_mask[i] if valid_mask is not None else None
        pred_conf_sample = pred_conf[i] if pred_conf is not None else None

        # Call the single-sample alignment function
        aligned_points, transform_params = align_pred_to_gt_torch(
            pred_points=pred_points_sample,
            gt_points=gt_points_sample,
            valid_mask=valid_mask_sample,
            pred_conf=pred_conf_sample,
            conf_threshold=conf_threshold,
            conf_percentage=conf_percentage,
            with_scale=with_scale
        )
        
        aligned_points_list.append(aligned_points)
        transform_params_list.append(transform_params)
        # print('gt', torch.linalg.norm(gt_points_sample, dim=-1).mean())
        # print('pred', torch.linalg.norm(pred_points_sample, dim=-1).mean())
        # print('pred2', torch.linalg.norm(aligned_points, dim=-1).mean())

    # Aggregate the results
    batch_aligned_points = torch.stack(aligned_points_list, dim=0)
    
    # Aggregate transform parameters into a single dictionary
    batch_transform_params = {
        'scale': torch.stack([p['scale'] for p in transform_params_list], dim=0),
        'rotation': torch.stack([p['rotation'] for p in transform_params_list], dim=0),
        'translation': torch.stack([p['translation'] for p in transform_params_list], dim=0)
    }
    # print(f"Scaled points by {batch_transform_params['scale']}")
    # print(f"{batch_transform_params=}")
    # diff = (batch_aligned_points - pred_points).abs()
    # print("diff", diff.max(), diff.mean())

    return batch_aligned_points, batch_transform_params

def align_extrinsics_torch(
    extrinsics: torch.Tensor,
    transform_params: Dict[str, torch.Tensor],
    with_scale: bool = True
) -> torch.Tensor:
    """
    Applies a batch of similarity transformations to a batch of camera extrinsics.

    This function updates the camera poses (extrinsics) to align them with the
    transformed point cloud space defined by the `transform_params`.

    Args:
        extrinsics (torch.Tensor): The original camera extrinsics.
                                   Shape can be (B, S, 3, 4) or (B, S, 4, 4).
        transform_params (Dict[str, torch.Tensor]): A dictionary from `align_pred_to_gt_torch_batch`
                                                    containing 'scale', 'rotation', and 'translation'.
                                                    - 'scale': (B,)
                                                    - 'rotation': (B, 3, 3)
                                                    - 'translation': (B, 3)
        with_scale (bool): If True, applies the scaling factor to the translation component.
                           This should match the `with_scale` argument used to generate
                           the `transform_params`.

    Returns:
        torch.Tensor: The aligned camera extrinsics, with the same shape as the input.
    """
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    dtype = extrinsics.dtype

    # --- 2. Unpack and Reshape Transformation Parameters for Broadcasting ---
    # The Umeyama parameters are per-batch-item, so we need to unsqueeze
    # them to broadcast across the sequence (S) dimension.
    
    # Rotation: (B, 3, 3) -> (B, 1, 3, 3)
    umeyama_R = transform_params['rotation'].view(B, 1, 3, 3)
    
    # Translation: (B, 3) -> (B, 1, 3, 1)
    umeyama_t = transform_params['translation'].view(B, 1, 3, 1)

    # --- 3. Decompose Original Extrinsics ---
    # R_e: (B, S, 3, 3), t_e: (B, S, 3)
    R_e = extrinsics[..., :3, :3]
    t_e = extrinsics[..., :3, 3]

    # --- 4. Apply the Transformation Formulas ---
    # The new extrinsic rotation R'_e is R_e * R^T
    # The new extrinsic translation t'_e is s * t_e - R'_e * t
    # where s, R, t are the Umeyama parameters.
    
    # R_e (B, S, 3, 3) @ umeyama_R.transpose (B, 1, 3, 3) -> R_aligned (B, S, 3, 3)
    R_aligned = R_e @ umeyama_R.transpose(-2, -1)
    
    # R_aligned (B, S, 3, 3) @ umeyama_t (B, 1, 3, 1) -> (B, S, 3, 1)
    # .squeeze(-1) removes the last dimension to make it (B, S, 3) for subtraction.
    t_offset = (R_aligned @ umeyama_t).squeeze(-1)      
    
    if with_scale:
        # Scale: (B,) -> (B, 1, 1)
        scale = transform_params['scale'].view(B, 1, 1)

        # scale (B, 1, 1) * t_e (B, S, 3) -> (B, S, 3)
        t_aligned = scale * t_e - t_offset
    else:
        t_aligned = t_e - t_offset

    # --- 5. Reconstruct the Aligned Extrinsics Matrix ---
    aligned_extrinsics = extrinsics.clone()
    aligned_extrinsics[..., :3, :3] = R_aligned
    aligned_extrinsics[..., :3, 3] = t_aligned

    return aligned_extrinsics
def align_c2w_poses_points_torch(
    transform_params: Dict[str, torch.Tensor],
    c2w_poses: Optional[torch.Tensor] = None,
    points3D: Optional[torch.Tensor] = None,
    with_scale: bool = False
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Applies a batch of similarity transformations (from Umeyama) to 
    a batch of camera-to-world (c2w) poses and 3D points.

    This function implements the correct transformation for c2w poses:
    R'_e = R_U @ R_e
    t'_e = s * (R_U @ t_e) + t_U

    And for points:
    p' = s * (p @ R_U.T) + t_U
    (Assuming roma's p' = s(p R^T) + t output convention)

    Args:
        transform_params (Dict[str, torch.Tensor]): From align_pred_to_gt_torch_batch
                                                    - 'scale': (B,)
                                                    - 'rotation': (B, 3, 3) (R_U)
                                                    - 'translation': (B, 3) (t_U)
        c2w_poses (Optional[torch.Tensor]): The original c2w poses.
                                            Shape (B, S, 3, 4) or (B, S, 4, 4).
        points3D (Optional[torch.Tensor]): The original 3D world points.
                                           Shape (B, ..., 3).
        with_scale (bool): If True, applies the scaling factor.

    Returns:
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        - aligned_c2w_poses (or None)
        - aligned_points3D (or None)
    """

    # --- 1. Define Broadcast-Ready Transformation Parameters ---
    
    # Umeyama Rotation: (B, 3, 3)
    umeyama_R = transform_params['rotation']
    B = umeyama_R.shape[0]
    device = umeyama_R.device
    
    # Umeyama Translation: (B, 1, 3) (broadcastable)
    umeyama_t_broadcast = transform_params['translation'].unsqueeze(1)

    # Prepare Scale
    scale_broadcast = None
    if with_scale and transform_params.get('scale') is not None:
        # Scale: (B, 1, 1) (broadcastable)
        scale_broadcast = transform_params['scale'].view(B, 1, 1)

    # --- 2. Align c2w Poses (Optional) ---
    aligned_c2w_poses = None
    
    if c2w_poses is not None:
        # Decompose Original Poses
        R_e = c2w_poses[..., :3, :3]  # (B, S, 3, 3)
        t_e = c2w_poses[..., :3, 3]   # (B, S, 3)

        # Apply R' = R_U @ R_e
        # (B, 1, 3, 3) @ (B, S, 3, 3) -> (B, S, 3, 3)
        R_aligned = umeyama_R.unsqueeze(1) @ R_e
        
        # Apply t' = s * (R_U @ t_e) + t_U
        # (B, 1, 3, 3) @ (B, S, 3, 1) -> (B, S, 3, 1) -> (B, S, 3)
        t_e_rotated = (umeyama_R.unsqueeze(1) @ t_e.unsqueeze(-1)).squeeze(-1)
        
        if scale_broadcast is not None:
            # (B, 1, 1) * (B, S, 3) + (B, 1, 3) -> (B, S, 3)
            t_aligned = scale_broadcast * t_e_rotated + umeyama_t_broadcast
        else:
            t_aligned = t_e_rotated + umeyama_t_broadcast

        # Reconstruct
        aligned_c2w_poses = c2w_poses.clone()
        aligned_c2w_poses[..., :3, :3] = R_aligned
        aligned_c2w_poses[..., :3, 3] = t_aligned

    # --- 3. Align 3D Points (Optional) ---
    aligned_points3D = None
    
    if points3D is not None:
        original_pts_shape = points3D.shape
        pts_flat = points3D.reshape(B, -1, 3) # (B, N, 3)

        # p' = s * (p @ R_U.T) + t_U
        
        # R_U Transpose: (B, 3, 3)
        umeyama_R_T = umeyama_R.transpose(-2, -1)
        
        # (B, N, 3) @ (B, 3, 3) -> (B, N, 3)
        scaled_pts = pts_flat @ umeyama_R_T
        
        if scale_broadcast is not None:
            # (B, 1, 1) * (B, N, 3) -> (B, N, 3)
            scaled_pts = scale_broadcast * scaled_pts
        
        # (B, N, 3) + (B, 1, 3) -> (B, N, 3)
        aligned_pts_flat = scaled_pts + umeyama_t_broadcast

        aligned_points3D = aligned_pts_flat.reshape(original_pts_shape)

    return aligned_c2w_poses, aligned_points3D
def align_c2w_poses_points_rotation_only(
    R: torch.Tensor,
    c2w_poses: Optional[torch.Tensor] = None,
    points3D: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Applies a batch of Rotations (B, 3, 3) to c2w poses and 3D points.
    Assumes Scale=1.0 and Translation=0.0.
    
    Math:
        R'_e = R @ R_e
        t'_e = R @ t_e  (Camera center is rotated around origin)
        p'   = p @ R^T  (Points are rotated around origin)

    Args:
        R (torch.Tensor): Rotation matrices. Shape (B, 3, 3).
        c2w_poses (Optional[torch.Tensor]): Original poses. Shape (B, S, 3, 4) or (B, S, 4, 4).
        points3D (Optional[torch.Tensor]): Original points. Shape (B, ..., 3).

    Returns:
        Tuple of (aligned_poses, aligned_points)
    """
    B = R.shape[0]
    
    # --- 1. Align c2w Poses ---
    aligned_c2w_poses = None
    
    if c2w_poses is not None:
        # Decompose
        R_e = c2w_poses[..., :3, :3]  # (B, S, 3, 3)
        t_e = c2w_poses[..., :3, 3]   # (B, S, 3)

        # Apply Rotation to Orientation: R' = R_rot @ R_cam
        # (B, 1, 3, 3) @ (B, S, 3, 3) -> (B, S, 3, 3)
        R_aligned = R.unsqueeze(1) @ R_e
        
        # Apply Rotation to Position: t' = R_rot @ t_cam
        # (B, 1, 3, 3) @ (B, S, 3, 1) -> (B, S, 3, 1) -> (B, S, 3)
        t_aligned = (R.unsqueeze(1) @ t_e.unsqueeze(-1)).squeeze(-1)

        # Reconstruct
        aligned_c2w_poses = c2w_poses.clone()
        aligned_c2w_poses[..., :3, :3] = R_aligned
        aligned_c2w_poses[..., :3, 3] = t_aligned

    # --- 2. Align 3D Points ---
    aligned_points3D = None
    
    if points3D is not None:
        original_pts_shape = points3D.shape
        pts_flat = points3D.reshape(B, -1, 3) 

        # p' = p @ R^T
        # (B, N, 3) @ (B, 3, 3) -> (B, N, 3)
        aligned_pts_flat = pts_flat @ R.transpose(-2, -1)

        aligned_points3D = aligned_pts_flat.reshape(original_pts_shape)

    return aligned_c2w_poses, aligned_points3D
