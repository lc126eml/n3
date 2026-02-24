#!/usr/bin/env python

import torch
from typing import Dict, Optional
def calculate_depth_metrics_optimized(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculates standard depth estimation metrics, optimized for efficiency by reducing redundant computations.
    The ground truth and predicted depth maps are assumed to be scale-aligned.
    This function will automatically handle invalid ground truth pixels (e.g., depth <= 0).

    Args:
        depth_gt (torch.Tensor): The ground truth depth map. Shape: (B, H, W) or (H, W).
        depth_pred (torch.Tensor): The predicted depth map. Shape: (B, H, W) or (H, W).
        valid_mask (Optional[torch.Tensor]): A boolean mask for valid pixels.

    Returns:
        Dict[str, float]: A dictionary containing the computed metrics:
                          'abs_rel', 'sq_rel', 'rmse', 'log_rmse',
                          'threshold_1' (delta < 1.25),
                          'threshold_2' (delta < 1.25^2),
                          'threshold_3' (delta < 1.25^3).
    """
    # --- 1. Input Validation and Masking ---
    depth_pred = depth_pred.squeeze(-1)
    if depth_gt.shape != depth_pred.shape:
        raise ValueError(f"Input shapes must match. Got gt: {depth_gt.shape}, pred: {depth_pred.shape}")

    # Create mask for valid ground truth pixels (depth > 0)
    mask = depth_gt > 1e-8
    if valid_mask is not None:
        mask = mask & valid_mask

    # Clamp predicted depth for numerical stability
    clamped_pred = torch.clamp(depth_pred, min=1e-13)

    # Apply the mask
    gt_masked = depth_gt[mask]
    pred_masked = clamped_pred[mask]

    # Handle edge case of no valid pixels
    if gt_masked.numel() == 0:
        print("Warning: No valid pixels found for metric calculation. Returning zeros.")
        return {
            'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 'log_rmse': 0.0,
            'threshold_1': 0.0, 'threshold_2': 0.0, 'threshold_3': 0.0
        }

    # --- 2. Pre-compute Errors (Core Optimization) ---
    # Compute the difference and its square once and reuse
    diff = gt_masked - pred_masked
    
    # --- 3. Error Metrics ---
    abs_rel = (torch.abs(diff) / gt_masked).mean()
    abe = torch.abs(diff).mean() # Absolute Error

    # diff_sq = diff ** 2
    # sq_rel = (diff_sq / gt_masked).mean()
    # rmse = torch.sqrt(diff_sq.mean())

    # --- 4. Threshold Accuracy Metrics (Delta) ---
    # ratio = gt_masked / pred_masked
    # delta = torch.maximum(ratio, 1.0 / ratio)
    
    # threshold_1 = (delta < 1.25).float().mean()
    # threshold_2 = (delta < 1.25 ** 2).float().mean()
    # threshold_3 = (delta < 1.25 ** 3).float().mean()

    # --- 5. Compile Results ---
    metrics = {
        'abs_rel': abs_rel.item(),
        'abe': abe.item(),
        # 'sq_rel': sq_rel.item(),
        # 'rmse': rmse.item(),
        # 'threshold_1': threshold_1.item(),
        # 'threshold_2': threshold_2.item(),
        # 'threshold_3': threshold_3.item()
    }

    return metrics

def calculate_depth_metrics(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculates standard depth estimation metrics on GPU using PyTorch.

    The ground truth and predicted depth maps are assumed to be scale-aligned.
    This function will automatically handle invalid ground truth pixels (e.g., depth <= 0).

    Args:
        depth_gt (torch.Tensor): The ground truth depth map. Shape: (B, H, W) or (H, W).
        depth_pred (torch.Tensor): The predicted depth map. Shape: (B, H, W) or (H, W).
        valid_mask (Optional[torch.Tensor]): A boolean mask where True indicates a valid
                                             pixel to be included in the evaluation.
                                             Shape: (B, H, W) or (H, W).

    Returns:
        Dict[str, float]: A dictionary containing the computed metrics:
                          'abs_rel', 'sq_rel', 'rmse', 'log_rmse',
                          'threshold_1' (delta < 1.25),
                          'threshold_2' (delta < 1.25^2),
                          'threshold_3' (delta < 1.25^3).
    """
    # --- 1. Input Validation and Masking ---
    depth_pred = depth_pred.squeeze(-1)
    if depth_gt.shape != depth_pred.shape:
        raise ValueError(f"Input shapes must match. Got gt: {depth_gt.shape}, pred: {depth_pred.shape}")

    # Create a mask for valid ground truth pixels (depth > 0)
    # Using a small epsilon for float comparison safety
    mask = depth_gt > 1e-13

    # If an additional valid_mask is provided, combine it
    if valid_mask is not None:
        if valid_mask.shape != depth_gt.shape:
            raise ValueError(f"Mask shape must match inputs. Got mask: {valid_mask.shape}")
        mask = mask & valid_mask

    # Clamp predicted depth to avoid division by zero or log(0)
    # This is a common practice in depth evaluation scripts
    clamped_pred = torch.clamp(depth_pred, min=1e-13)

    # Apply the mask to both ground truth and prediction
    gt_masked = depth_gt[mask]
    pred_masked = clamped_pred[mask]
    # print(depth_gt.max(), depth_gt.min(), depth_gt.mean())
    # print(depth_pred.max(), depth_pred.min(), depth_pred.mean())
    # print('gt depth', gt_masked.max(), gt_masked.min(), gt_masked.mean())
    # print(pred_masked.max(), pred_masked.min(), pred_masked.mean())
    
    # If no valid pixels are found, return zero for all metrics
    if gt_masked.numel() == 0:
        print("Warning: No valid pixels found for metric calculation. Returning zeros.")
        return {
            'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 'log_rmse': 0.0,
            'threshold_1': 0.0, 'threshold_2': 0.0, 'threshold_3': 0.0
        }

    # --- 2. Error Metrics ---
    # Absolute Relative Error
    abs_rel = (torch.abs(gt_masked - pred_masked) / gt_masked).mean()

    # Square Relative Error
    sq_rel = (((gt_masked - pred_masked) ** 2) / gt_masked).mean()

    # Root Mean Squared Error
    rmse = torch.sqrt(((gt_masked - pred_masked) ** 2).mean())
    abe = torch.abs((gt_masked - pred_masked)).mean()

    # Root Mean Squared Logarithmic Error
    log_rmse = torch.sqrt(((torch.log(gt_masked) - torch.log(pred_masked)) ** 2).mean())

    # --- 3. Threshold Accuracy Metrics (Delta) ---
    # δ = max(gt / pred, pred / gt)
    delta = torch.maximum((gt_masked / pred_masked), (pred_masked / gt_masked))
    
    threshold_1 = (delta < 1.25).float().mean()
    threshold_2 = (delta < 1.25 ** 2).float().mean()
    threshold_3 = (delta < 1.25 ** 3).float().mean()

    # --- 4. Compile Results ---
    metrics = {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'abe': abe.item(),
        'log_rmse': log_rmse.item(),
        'threshold_1': threshold_1.item(),
        'threshold_2': threshold_2.item(),
        'threshold_3': threshold_3.item()
    }

    return metrics
