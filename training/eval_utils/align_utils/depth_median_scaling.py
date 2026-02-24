import torch
from typing import Optional, Tuple

def median_scale_depth_torch(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    pred_conf: Optional[torch.Tensor] = None,
    conf_threshold: Optional[float] = None,
    conf_percentage: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns a predicted depth map to a ground truth depth map using median scaling (PyTorch version).

    This function calculates a single scale factor based on the median values of a
    filtered subset of pixels and applies it to the entire predicted depth map.
    All computations are performed on the input tensor's device (CPU or GPU).

    Args:
        pred_depth (torch.Tensor): The predicted depth map(s).
                                   Shape (H, W, 1).
        gt_depth (torch.Tensor): The ground truth depth map(s). Must have the
                                 same shape as pred_depth.
        valid_mask (Optional[torch.Tensor]): A boolean mask where True indicates a valid
                                             pixel. If None, all pixels are considered valid.
                                             Shape must match pred_depth.
        pred_conf (Optional[torch.Tensor]): A confidence map for the predictions.
                                            Shape must match pred_depth. Defaults to None.
        conf_threshold (Optional[float]): The minimum absolute confidence value for a prediction
                                          to be used for calculating the scale. Defaults to None.
        conf_percentage (Optional[float]): The top percentage of confident points to use
                                           (e.g., 90 for top 90%). Value between 0-100.
                                           Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - The scaled predicted depth map.
            - The median of the filtered predicted depth subset.
            - The median of the filtered ground truth depth subset.
    """
    # --- 1. Input Validation and Device Setup ---
    # if pred_depth.shape[-1] == 1:
    #     pred_depth = pred_depth.squeeze(-1)
    
    # if pred_depth.shape != gt_depth.shape:
    #     raise ValueError("Prediction and GT depth shapes must match.")
    device = pred_depth.device

    # --- 2. Create the combined mask for selecting pixels ---
    combined_mask = gt_depth > 1e-8
    if valid_mask is not None:
        combined_mask &= valid_mask.bool()

    if pred_conf is not None:
        if conf_threshold is not None:
            combined_mask &= (pred_conf >= conf_threshold)
        
        elif conf_percentage is not None:
            if not (0 < conf_percentage <= 100):
                raise ValueError("conf_percentage must be between 0 and 100.")
            
            valid_conf_scores = pred_conf[combined_mask]
            
            if valid_conf_scores.numel() > 0:
                percentile_value = torch.quantile(valid_conf_scores.float(), (100 - conf_percentage) / 100.0)
                combined_mask &= (pred_conf >= percentile_value)

    # --- 3. Extract subsets for scaling ---
    gt_subset = gt_depth[combined_mask]
    pred_subset = pred_depth[combined_mask]

    # --- 4. Handle edge cases ---
    if gt_subset.numel() < 2 or pred_subset.numel() < 2:
        print("Warning: Fewer than 2 valid points for scaling. Returning unscaled prediction.")
        return pred_depth

    # --- 5. Calculate medians and scale factor ---
    median_gt = torch.median(gt_subset)
    median_pred = torch.median(pred_subset)

    if torch.abs(median_pred) < 1e-8:
        print("Warning: Median of predicted depth is close to zero. Returning unscaled prediction.")
        scale = torch.tensor(1.0, device=device)
    else:
        scale = median_gt / median_pred

    # print(f"Scaling depth with {scale}")
    # --- 6. Apply scaling and return ---
    return pred_depth * scale.item(), median_pred, median_gt


def median_scale_depth_torch_batch(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    pred_conf: Optional[torch.Tensor] = None,
    conf_threshold: Optional[float] = None,
    conf_percentage: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Processes a batch of predicted depth maps and aligns each to its corresponding
    ground truth depth map using median scaling.

    This function iterates over the batch dimension and calls `median_scale_depth_torch`
    for each sample, then aggregates the results.

    Args:
        pred_depth (torch.Tensor): Batched predicted depth maps. Shape (B, S, H, W, 1).
        gt_depth (torch.Tensor): Batched ground truth depth maps. Shape (B, S, H, W).
        valid_mask (Optional[torch.Tensor]): Batched boolean mask for valid pixels.
                                              Shape (B, S, H, W). Defaults to None.
        pred_conf (Optional[torch.Tensor]): Batched confidence maps.
                                             Shape (B, S, H, W). Defaults to None.
        conf_threshold (Optional[float]): See `median_scale_depth_torch`.
        conf_percentage (Optional[float]): See `median_scale_depth_torch`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - The batch of scaled predicted depth maps.
            - The batch of median predicted depth values.
            - The batch of median ground truth depth values.
    """

    if pred_depth.dim() != 5:
        raise ValueError(f"Expected pred_depth to be a 4D tensor (B, S, H, W, 1), but got {pred_depth.shape}.")

    batch_size = pred_depth.shape[0]

    # Lists to store results from each sample
    scaled_depth_list = []
    median_pred_list = []
    median_gt_list = []
    # print("original depth", pred_depth.max(), pred_depth.min(), pred_depth.mean())

    for i in range(batch_size):
        # Slice the batch dimension for all inputs
        pred_depth_sample = pred_depth[i]
        gt_depth_sample = gt_depth[i]
        
        valid_mask_sample = valid_mask[i] if valid_mask is not None else None
        pred_conf_sample = pred_conf[i] if pred_conf is not None else None

        # Call the single-sample scaling function
        scaled_depth, median_pred, median_gt = median_scale_depth_torch(
            pred_depth=pred_depth_sample,
            gt_depth=gt_depth_sample,
            valid_mask=valid_mask_sample,
            pred_conf=pred_conf_sample,
            conf_threshold=conf_threshold,
            conf_percentage=conf_percentage
        )
        
        scaled_depth_list.append(scaled_depth)
        median_pred_list.append(median_pred)
        median_gt_list.append(median_gt)

    # Stack the results into batched tensors
    batch_scaled_depth = torch.stack(scaled_depth_list, dim=0)
    batch_median_pred = torch.stack(median_pred_list, dim=0)
    batch_median_gt = torch.stack(median_gt_list, dim=0)

    return batch_scaled_depth, batch_median_pred, batch_median_gt

