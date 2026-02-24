import numpy as np
from typing import Optional, Union, Tuple

def median_scale_depth(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    pred_conf: Optional[np.ndarray] = None,
    conf_threshold: Optional[float] = None,
    conf_percentage: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Aligns a predicted depth map to a ground truth depth map using median scaling.

    This function calculates a single scale factor based on the median values of a
    filtered subset of pixels. The filtering is determined by an optional validity mask
    and optional confidence thresholds (absolute and/or percentage-based). This scale
    factor is then applied to the entire predicted depth map.

    Args:
        pred_depth (np.ndarray): The predicted depth map(s).
                                 Shape (H, W) or (B, H, W).
        gt_depth (np.ndarray): The ground truth depth map(s). Must have the
                               same shape as pred_depth.
        valid_mask (Optional[np.ndarray]): A boolean mask where True indicates a valid
                                           ground truth pixel. If None, all pixels
                                           are considered valid. Shape must match pred_depth.
        pred_conf (Optional[np.ndarray]): A confidence map for the predictions.
                                          Shape must match pred_depth. Defaults to None.
        conf_threshold (Optional[float]): The minimum absolute confidence value for a prediction
                                          to be used for calculating the scale. Defaults to None.
        conf_percentage (Optional[float]): The top percentage of confident points to use
                                           for calculating the scale (e.g., 90 for top 90%).
                                           This is applied *after* the conf_threshold.
                                           Value should be between 0 and 100. Defaults to None.

    Returns:
        Tuple[np.ndarray, float, float]: A tuple containing:
            - The scaled predicted depth map.
            - The median of the filtered predicted depth subset.
            - The median of the filtered ground truth depth subset.
    """
    # --- 1. Input Validation ---
    if pred_depth.shape != gt_depth.shape:
        raise ValueError("Prediction and GT depth shapes must match.")
    if valid_mask is not None and pred_depth.shape != valid_mask.shape:
        raise ValueError("Shape mismatch between depth and valid_mask.")
    if pred_conf is not None and pred_depth.shape != pred_conf.shape:
        raise ValueError("Shape mismatch between depth and pred_conf.")

    # --- 2. Create the combined mask for selecting pixels ---
    # If valid_mask is not provided, assume all pixels are valid initially.
    combined_mask = gt_depth > 1e-8

    if valid_mask is not None:
        combined_mask = np.logical_and(combined_mask, valid_mask.astype(bool))

    # If confidence scores are provided, apply confidence-based filtering
    if pred_conf is not None:
        # Apply the absolute confidence threshold first, if provided
        if conf_threshold is not None:
            conf_mask = pred_conf >= conf_threshold
            combined_mask = np.logical_and(combined_mask, conf_mask)

        # If a percentage is given, further filter to keep the top N%
        elif conf_percentage is not None:
            if not (0 < conf_percentage <= 100):
                raise ValueError("conf_percentage must be between 0 and 100.")

            # Get confidence values of the currently valid points
            valid_conf_scores = pred_conf[combined_mask]

            if valid_conf_scores.size > 0:
                # Calculate the threshold for the top percentage
                # e.g., for top 90%, we find the 10th percentile value
                percentile_threshold = np.percentile(valid_conf_scores, 100 - conf_percentage)

                # Create a new mask based on this percentile threshold
                percentage_mask = pred_conf >= percentile_threshold
                combined_mask = np.logical_and(combined_mask, percentage_mask)

    # --- 3. Extract the subsets for calculating the scale ---
    gt_subset = gt_depth[combined_mask]
    pred_subset = pred_depth[combined_mask]

    # --- 4. Handle edge cases where no valid pixels are found ---
    if gt_subset.size < 2 or pred_subset.size < 2:
        print("Warning: Fewer than 2 valid points for scaling. Returning unscaled prediction.")
        return pred_depth

    # --- 5. Calculate the medians and the scale factor ---
    median_gt = np.median(gt_subset)
    median_pred = np.median(pred_subset)

    if np.abs(median_pred) < 1e-8:
        print("Warning: Median of predicted depth is close to zero. Returning unscaled prediction.")
        scale = 1.0
    else:
        scale = median_gt / median_pred

    # --- 6. Apply the scaling to the entire prediction and return ---
    scaled_pred_depth = pred_depth * scale

    return scaled_pred_depth, median_pred, median_gt


