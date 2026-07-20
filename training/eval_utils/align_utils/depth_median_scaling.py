import torch
from typing import Optional, Tuple


def _estimate_depth_scale(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    pred_conf: Optional[torch.Tensor],
    conf_threshold: Optional[float],
    conf_percentage: Optional[float],
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate a scalar depth scale and whether the estimate is usable."""
    if mode not in {"mean", "median"}:
        raise ValueError(f"mode must be 'mean' or 'median', got {mode!r}")

    combined_mask = gt_depth > 1e-8
    if valid_mask is not None:
        combined_mask &= valid_mask.bool()

    if pred_conf is not None:
        if conf_threshold is not None:
            combined_mask &= pred_conf >= conf_threshold
        elif conf_percentage is not None:
            if not (0 < conf_percentage <= 100):
                raise ValueError("conf_percentage must be between 0 and 100.")

            valid_conf_scores = pred_conf[combined_mask]
            if valid_conf_scores.numel() > 0:
                percentile_value = torch.quantile(
                    valid_conf_scores.float(), (100 - conf_percentage) / 100.0
                )
                combined_mask &= pred_conf >= percentile_value

    gt_subset = gt_depth[combined_mask]
    pred_subset = pred_depth[combined_mask]

    if gt_subset.numel() < 2 or pred_subset.numel() < 2:
        print("Warning: Fewer than 2 valid points for scaling. Returning identity scale.")
        return (
            pred_depth.new_tensor(1.0),
            pred_depth.new_tensor(False, dtype=torch.bool),
        )

    if mode == "mean":
        gt_stat = gt_subset.mean()
        pred_stat = pred_subset.mean()
    else:
        gt_stat = torch.median(gt_subset)
        pred_stat = torch.median(pred_subset)

    near_zero = pred_stat.abs() < 1e-8
    safe_pred_stat = torch.where(near_zero, torch.ones_like(pred_stat), pred_stat)
    scale = torch.where(near_zero, torch.ones_like(pred_stat), gt_stat / safe_pred_stat)
    scale = scale.to(dtype=pred_depth.dtype)
    return scale, ~near_zero


def median_scale_depth_torch(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    pred_conf: Optional[torch.Tensor] = None,
    conf_threshold: Optional[float] = None,
    conf_percentage: Optional[float] = None,
    detach_scale: bool = True,
    mode: str = "median",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns a predicted depth map to a ground truth depth map using median scaling (PyTorch version).

    This function calculates a single scale factor based on the median values of a
    filtered subset of pixels and applies it to the entire predicted depth map.
    All computations are performed on the input tensor's device (CPU or GPU).

    Args:
        pred_depth (torch.Tensor): The predicted depth map(s).
                                   Shape (H, W, 1).
        gt_depth (torch.Tensor): The ground truth depth map(s). Shape (H, W).
        valid_mask (Optional[torch.Tensor]): A boolean mask where True indicates a valid
                                             pixel. If None, all pixels are considered valid.
                                             Shape must match gt_depth.
        pred_conf (Optional[torch.Tensor]): A confidence map for the predictions.
                                            Shape must match gt_depth. Defaults to None.
        conf_threshold (Optional[float]): The minimum absolute confidence value for a prediction
                                          to be used for calculating the scale. Defaults to None.
        conf_percentage (Optional[float]): The top percentage of confident points to use
                                           (e.g., 90 for top 90%). Value between 0-100.
                                           Defaults to None.
        detach_scale (bool): Whether to detach the estimated scale before applying and
                             returning it. Defaults to True.
        mode (str): Statistic used to estimate scale, either ``"mean"`` or
                    ``"median"``. Defaults to ``"median"``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - The scaled predicted depth map.
            - The scale applied to the predicted depth map.
            - A scalar boolean indicating whether the scale estimate is valid.
    """
    scale, scale_valid = _estimate_depth_scale(
        pred_depth,
        gt_depth,
        valid_mask,
        pred_conf,
        conf_threshold,
        conf_percentage,
        mode,
    )

    if detach_scale:
        scale = scale.detach()

    return pred_depth * scale, scale, scale_valid


def median_scale_depth_torch_batch(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    pred_conf: Optional[torch.Tensor] = None,
    conf_threshold: Optional[float] = None,
    conf_percentage: Optional[float] = None,
    detach_scale: bool = True,
    mode: str = "median",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Processes a batch of predicted depth maps and aligns each to its corresponding
    ground truth depth map using median scaling.

    This function estimates one scale per sample, then applies all scales in one
    broadcasted batch operation.

    Args:
        pred_depth (torch.Tensor): Batched predicted depth maps. Shape (B, S, H, W, 1).
        gt_depth (torch.Tensor): Batched ground truth depth maps. Shape (B, S, H, W).
        valid_mask (Optional[torch.Tensor]): Batched boolean mask for valid pixels.
                                              Shape (B, S, H, W). Defaults to None.
        pred_conf (Optional[torch.Tensor]): Batched confidence maps.
                                             Shape (B, S, H, W). Defaults to None.
        conf_threshold (Optional[float]): See `median_scale_depth_torch`.
        conf_percentage (Optional[float]): See `median_scale_depth_torch`.
        detach_scale (bool): Whether to detach the estimated scales before applying and
                             returning them. Defaults to True.
        mode (str): Statistic used to estimate each scale, either ``"mean"`` or
                    ``"median"``. Defaults to ``"median"``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - The batch of scaled predicted depth maps.
            - The batch of applied scale values.
            - The batch of boolean scale-validity values.
    """

    if pred_depth.dim() != 5:
        raise ValueError(f"Expected pred_depth to be a 5D tensor (B, S, H, W, 1), but got {pred_depth.shape}.")

    batch_size = pred_depth.shape[0]

    # Lists to store the scalar results from each sample
    scale_list = []
    scale_valid_list = []
    # print("original depth", pred_depth.max(), pred_depth.min(), pred_depth.mean())

    for i in range(batch_size):
        # Slice the batch dimension for all inputs
        pred_depth_sample = pred_depth[i]
        gt_depth_sample = gt_depth[i]
        
        valid_mask_sample = valid_mask[i] if valid_mask is not None else None
        pred_conf_sample = pred_conf[i] if pred_conf is not None else None

        scale, scale_valid = _estimate_depth_scale(
            pred_depth_sample,
            gt_depth_sample,
            valid_mask_sample,
            pred_conf_sample,
            conf_threshold,
            conf_percentage,
            mode,
        )
        
        scale_list.append(scale)
        scale_valid_list.append(scale_valid)

    # Apply all per-sample scales in one broadcasted batch operation.
    batch_scale = torch.stack(scale_list, dim=0)
    if detach_scale:
        batch_scale = batch_scale.detach()
    batch_scale_valid = torch.stack(scale_valid_list, dim=0)
    scale_shape = [batch_size] + [1] * (pred_depth.ndim - 1)
    batch_scaled_depth = pred_depth * batch_scale.view(scale_shape)

    return batch_scaled_depth, batch_scale, batch_scale_valid
