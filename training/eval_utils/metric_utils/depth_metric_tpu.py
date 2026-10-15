#!/usr/bin/env python

import torch
from typing import Dict, Literal, Optional

DepthAlignMode = Literal["none", "median", "scale", "scale_shift", "scale_and_shift"]


def _masked_flat_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-item lower median over masked entries of flattened (B, N) tensors.

    Invalid entries are padded with +inf before sorting so the k-th smallest
    value is exactly the lower median of the valid set (torch.median semantics).
    Returns NaN for items whose mask is empty. Dense and fixed-shape, so it is
    safe for XLA/TPU graphs.
    """
    sorted_vals = torch.where(mask, values, torch.full_like(values, float("inf"))).sort(dim=-1).values
    count = mask.sum(dim=-1)
    k = ((count - 1).clamp_min(0) // 2).unsqueeze(-1)
    median = sorted_vals.gather(-1, k).squeeze(-1)
    return torch.where(count > 0, median, torch.full_like(median, float("nan")))


def _align_depth_for_metrics(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    align: DepthAlignMode = "median",
    eps: float = 1e-8,
) -> torch.Tensor:
    if align == "none":
        return depth_pred
    if align not in {"median", "scale", "scale_shift", "scale_and_shift"}:
        raise ValueError(
            f"Unsupported depth align mode {align!r}. "
            "Expected one of: 'none', 'median', 'scale', 'scale_shift', 'scale_and_shift'."
        )

    # Align per batch item for ndim >= 3, otherwise over the whole tensor.
    # Everything below is dense and fixed-shape (no boolean indexing, no
    # data-dependent host branches) so it stays inside one XLA graph.
    lead_dim = depth_pred.dim()
    batch = depth_pred.shape[0] if lead_dim >= 3 else 1
    flat_pred = depth_pred.reshape(batch, -1)
    flat_gt = depth_gt.reshape(batch, -1)
    flat_mask = mask.reshape(batch, -1).bool()
    count = flat_mask.sum(dim=-1)
    has_valid = count > 0

    if align == "median":
        median_pred = _masked_flat_median(flat_pred, flat_mask)
        median_gt = _masked_flat_median(flat_gt, flat_mask)
        # Unscaled fallback when the mask is empty or the predicted median is
        # too close to zero (matches the legacy early-return branches).
        safe_denom = torch.where(median_pred.abs() > eps, median_pred, torch.ones_like(median_pred))
        raw_scale = median_gt / safe_denom
        use_scale = has_valid & (median_pred.abs() > eps)
        scale = torch.where(use_scale, raw_scale, torch.ones_like(raw_scale))
        shift = torch.zeros_like(scale)
    else:
        safe_pred = torch.where(flat_mask, flat_pred, torch.zeros_like(flat_pred))
        safe_gt = torch.where(flat_mask, flat_gt, torch.zeros_like(flat_gt))
        sum_p2 = (safe_pred * safe_pred).sum(dim=-1)
        sum_p = safe_pred.sum(dim=-1)
        sum_g = safe_gt.sum(dim=-1)
        sum_pg = (safe_pred * safe_gt).sum(dim=-1)
        if align == "scale":
            raw_scale = sum_pg / sum_p2.clamp_min(eps)
            scale = torch.where(has_valid, raw_scale, torch.ones_like(raw_scale))
            shift = torch.zeros_like(scale)
        else:
            num = count.to(flat_pred.dtype)
            det = sum_p2 * num - sum_p * sum_p
            det_ok = det.abs() > eps
            safe_det = torch.where(det_ok, det, torch.ones_like(det))
            lstsq_scale = (sum_pg * num - sum_p * sum_g) / safe_det
            lstsq_shift = (sum_p2 * sum_g - sum_p * sum_pg) / safe_det
            fallback_scale = sum_pg / sum_p2.clamp_min(eps)
            scale = torch.where(det_ok, lstsq_scale, fallback_scale)
            shift = torch.where(det_ok, lstsq_shift, torch.zeros_like(scale))
            scale = torch.where(has_valid, scale, torch.ones_like(scale))
            shift = torch.where(has_valid, shift, torch.zeros_like(shift))

    view_shape = [batch] + [1] * (lead_dim - 1)
    return depth_pred * scale.view(view_shape) + shift.view(view_shape)


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

    All reductions are dense and fixed-shape (masked sums instead of boolean
    indexing), so the computation is safe for XLA/TPU graphs.

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

    weights = mask.to(depth_gt.dtype)
    count = weights.sum()
    has_valid = count > 0
    safe_count = count.clamp_min(1.0)
    nan = torch.tensor(float("nan"), device=depth_gt.device, dtype=depth_gt.dtype)
    zero = torch.zeros_like(depth_gt)

    def _masked_mean(value_map: torch.Tensor) -> torch.Tensor:
        return torch.where(has_valid, value_map.sum() / safe_count, nan)

    diff = depth_gt - clamped_pred
    abs_diff_map = torch.where(mask, diff.abs(), zero)
    abs_rel_map = torch.where(mask, diff.abs() / depth_gt.clamp_min(eps), zero)

    metrics = {
        'abs_rel': _masked_mean(abs_rel_map).item(),
        'abe': _masked_mean(abs_diff_map).item(),
    }
    if include_sq_rel or include_rmse:
        diff_sq = diff.square()
        if include_sq_rel:
            sq_rel_map = torch.where(mask, diff_sq / depth_gt.clamp_min(eps), zero)
            metrics['sq_rel'] = _masked_mean(sq_rel_map).item()
        if include_rmse:
            rmse_map = torch.where(mask, diff_sq, zero)
            metrics['rmse'] = torch.sqrt(_masked_mean(rmse_map)).item()
    if include_log_rmse:
        log_diff_sq_map = torch.where(
            mask,
            (torch.log(depth_gt.clamp_min(eps)) - torch.log(clamped_pred)).square(),
            zero,
        )
        metrics['log_rmse'] = torch.sqrt(_masked_mean(log_diff_sq_map)).item()
    if include_delta:
        delta = torch.maximum(
            depth_gt / clamped_pred,
            clamped_pred / depth_gt.clamp_min(eps),
        )
        metrics['threshold_1'] = _masked_mean(torch.where(mask, (delta < 1.25).to(depth_gt.dtype), zero)).item()
        metrics['threshold_2'] = _masked_mean(torch.where(mask, (delta < 1.25 ** 2).to(depth_gt.dtype), zero)).item()
        metrics['threshold_3'] = _masked_mean(torch.where(mask, (delta < 1.25 ** 3).to(depth_gt.dtype), zero)).item()

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
                                             Shape: (B, H, W).

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
