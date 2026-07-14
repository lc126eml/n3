#!/usr/bin/env python

import torch
from typing import Dict, Literal, Optional

DepthAlignMode = Literal["none", "median", "scale", "scale_shift", "scale_and_shift"]


def _align_depth_for_metrics(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    align: DepthAlignMode = "median",
    eps: float = 1e-8,
) -> torch.Tensor:
    if align == "none":
        return depth_pred

    if depth_pred.dim() >= 3:
        aligned = []
        for pred_i, gt_i, mask_i in zip(depth_pred, depth_gt, mask):
            aligned_i = _align_depth_for_metrics(
                pred_i.reshape(-1),
                gt_i.reshape(-1),
                mask_i.reshape(-1),
                align=align,
                eps=eps,
            ).reshape_as(pred_i)
            aligned.append(aligned_i)
        return torch.stack(aligned, dim=0)

    pred_valid = depth_pred[mask]
    gt_valid = depth_gt[mask]
    if pred_valid.numel() == 0:
        return depth_pred

    if align == "median":
        median_pred = torch.median(pred_valid)
        if median_pred.abs() <= eps:
            return depth_pred
        scale = torch.median(gt_valid) / median_pred
        return scale * depth_pred

    if align == "scale":
        denom = torch.sum(pred_valid * pred_valid).clamp_min(eps)
        scale = torch.sum(pred_valid * gt_valid) / denom
        return scale * depth_pred

    if align in {"scale_shift", "scale_and_shift"}:
        num = pred_valid.numel()
        sum_p2 = torch.sum(pred_valid * pred_valid)
        sum_p = torch.sum(pred_valid)
        sum_g = torch.sum(gt_valid)
        sum_pg = torch.sum(pred_valid * gt_valid)
        det = sum_p2 * num - sum_p * sum_p
        if det.abs() <= eps:
            scale = sum_pg / sum_p2.clamp_min(eps)
            shift = torch.zeros_like(scale)
        else:
            scale = (sum_pg * num - sum_p * sum_g) / det
            shift = (sum_p2 * sum_g - sum_p * sum_pg) / det
        return scale * depth_pred + shift

    raise ValueError(
        f"Unsupported depth align mode {align!r}. "
        "Expected one of: 'none', 'median', 'scale', 'scale_shift', 'scale_and_shift'."
    )


def calculate_depth_metrics_optimized(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    include_sq_rel: bool = False,
    include_rmse: bool = False,
    include_log_rmse: bool = False,
    include_delta: bool = False,
    align: DepthAlignMode = "median",
    eps: float = 1e-8,
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
    depth_pred = depth_pred.squeeze(-1)
    depth_gt = depth_gt.squeeze(-1)
    if depth_gt.shape != depth_pred.shape:
        raise ValueError(f"Input shapes must match. Got gt: {depth_gt.shape}, pred: {depth_pred.shape}")

    mask = (depth_gt > eps) & torch.isfinite(depth_gt) & torch.isfinite(depth_pred)
    if valid_mask is not None:
        valid_mask = valid_mask.squeeze(-1)
        if valid_mask.shape != depth_gt.shape:
            raise ValueError(f"Mask shape must match inputs. Got mask: {valid_mask.shape}")
        mask = mask & valid_mask

    depth_pred = _align_depth_for_metrics(depth_pred, depth_gt, mask, align=align, eps=eps)
    mask = mask & torch.isfinite(depth_pred)
    clamped_pred = torch.clamp(depth_pred, min=eps)
    gt_masked = depth_gt[mask]
    pred_masked = clamped_pred[mask]

    metric_names = ["abs_rel", "abe"]
    if include_sq_rel:
        metric_names.append("sq_rel")
    if include_rmse:
        metric_names.append("rmse")
    if include_log_rmse:
        metric_names.append("log_rmse")
    if include_delta:
        metric_names.extend(["threshold_1", "threshold_2", "threshold_3"])

    if gt_masked.numel() == 0:
        return {name: float("nan") for name in metric_names}

    diff = gt_masked - pred_masked
    abs_diff = torch.abs(diff)

    metrics = {
        'abs_rel': (abs_diff / gt_masked).mean().item(),
        'abe': abs_diff.mean().item(),
    }
    if include_sq_rel or include_rmse:
        diff_sq = diff.square()
        if include_sq_rel:
            metrics['sq_rel'] = (diff_sq / gt_masked).mean().item()
        if include_rmse:
            metrics['rmse'] = torch.sqrt(diff_sq.mean()).item()
    if include_log_rmse:
        metrics['log_rmse'] = torch.sqrt(
            (torch.log(gt_masked) - torch.log(pred_masked)).square().mean()
        ).item()
    if include_delta:
        delta = torch.maximum(gt_masked / pred_masked, pred_masked / gt_masked)
        metrics['threshold_1'] = (delta < 1.25).float().mean().item()
        metrics['threshold_2'] = (delta < 1.25 ** 2).float().mean().item()
        metrics['threshold_3'] = (delta < 1.25 ** 3).float().mean().item()

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
    return calculate_depth_metrics_optimized(
        depth_gt,
        depth_pred,
        valid_mask,
        include_sq_rel=True,
        include_rmse=True,
        include_log_rmse=True,
        include_delta=True,
    )
