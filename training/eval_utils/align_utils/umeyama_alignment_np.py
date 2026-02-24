import numpy as np
from typing import Tuple, Dict, Optional, Union

def umeyama_alignment(x: np.ndarray, y: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes the optimal similarity transformation between two sets of corresponding points.

    This function finds the scale, rotation, and translation that minimizes the root-mean-square
    error between the transformed points of x and the points of y.

    Args:
        x (np.ndarray): The first set of points, shape (N, 3).
        y (np.ndarray): The second set of corresponding points, shape (N, 3).
        with_scale (bool): Whether to estimate the scale factor.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: A tuple containing:
            - s (float): The scale factor.
            - R (np.ndarray): The 3x3 rotation matrix.
            - t (np.ndarray): The 3x1 translation vector.
    """
    # Ensure input arrays are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate centroids
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)

    # Center the points
    x_centered = x - mu_x
    y_centered = y - mu_y

    # Calculate the covariance matrix
    Sigma = x_centered.T @ y_centered

    # Perform Singular Value Decomposition (SVD)
    U, D, Vt = np.linalg.svd(Sigma)
    
    # Ensure a right-handed coordinate system
    S = np.eye(x.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
        
    # Calculate the rotation matrix
    R = U @ S @ Vt

    # Calculate the scale factor
    if with_scale:
        var_x = np.var(x_centered, axis=0).sum()
        s = (1 / var_x) * np.sum(D) if var_x > 1e-8 else 1.0
    else:
        s = 1.0

    # Calculate the translation vector
    t = mu_y - s * (R @ mu_x)

    return s, R, t

def align_pred_to_gt(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    pred_conf: Optional[np.ndarray] = None,
    conf_threshold: Optional[float] = None,
    conf_percentage: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, Union[float, np.ndarray]]]:
    """
    Aligns predicted 3D points to ground truth points using the Umeyama algorithm.

    The function filters points based on a combination of optional masks and thresholds
    (validity, confidence threshold, confidence percentage). It then calculates the
    optimal similarity transformation on this filtered subset and applies it to the
    *original, unfiltered* predicted points.

    Args:
        pred_points (np.ndarray): Predicted 3D points. Shape (S, 3, W, H).
        gt_points (np.ndarray): Ground truth 3D points. Shape (S, 3, W, H).
        valid_mask (Optional[np.ndarray]): Boolean mask for valid points. If None, all
                                           points are considered valid. Shape (S, W, H).
        pred_conf (Optional[np.ndarray]): Confidence scores for predicted points.
                                           Shape (S, W, H). Defaults to None.
        conf_threshold (Optional[float]): Minimum absolute confidence to consider a point.
                                          Defaults to None.
        conf_percentage (Optional[float]): Top percentage of confident points to use
                                           (e.g., 90 for top 90%). Applied after
                                           conf_threshold. Value between 0-100.
                                           Defaults to None.

    Returns:
        Tuple[np.ndarray, Dict[str, Union[float, np.ndarray]]]: A tuple containing:
            - aligned_pred_points (np.ndarray): The transformed predicted points,
              in the original shape (S, 3, W, H).
            - transform_params (Dict): A dictionary with 'scale', 'rotation', and
              'translation' parameters.
    """
    # --- 1. Input Validation ---
    if pred_points.shape != gt_points.shape:
        raise ValueError("Prediction and GT point shapes must match.")
    shape_prefix = pred_points.shape[0:1] + pred_points.shape[2:]
    if valid_mask is not None and shape_prefix != valid_mask.shape:
        raise ValueError("Shape mismatch between points and valid_mask.")
    if pred_conf is not None and shape_prefix != pred_conf.shape:
        raise ValueError("Shape mismatch between points and pred_conf.")

    S, _, W, H = pred_points.shape
    
    # --- 2. Reshape data for processing ---
    pred_points_flat = pred_points.transpose(0, 2, 3, 1).reshape(-1, 3)
    gt_points_flat = gt_points.transpose(0, 2, 3, 1).reshape(-1, 3)

    # --- 3. Create a combined filter mask ---
    # Start with a mask of all True, or the user-provided valid_mask
    if valid_mask is None:
        combined_mask = np.full(gt_points_flat.shape[0], True, dtype=bool)
    else:
        combined_mask = valid_mask.flatten().astype(bool)

    # If confidence scores are provided, apply confidence-based filtering
    if pred_conf is not None:
        flat_conf = pred_conf.flatten()

        # Apply the absolute confidence threshold first, if provided
        if conf_threshold is not None:
            conf_mask = flat_conf >= conf_threshold
            combined_mask = np.logical_and(combined_mask, conf_mask)

        # If a percentage is given, further filter to keep the top N%
        elif conf_percentage is not None:
            if not (0 < conf_percentage <= 100):
                raise ValueError("conf_percentage must be between 0 and 100.")
            
            # Get confidence values of the currently valid points
            current_valid_conf_scores = flat_conf[combined_mask]
            
            if current_valid_conf_scores.size > 0:
                # Calculate the threshold for the top percentage
                percentile_value = np.percentile(current_valid_conf_scores, 100 - conf_percentage)
                percentage_mask = flat_conf >= percentile_value
                combined_mask = np.logical_and(combined_mask, percentage_mask)

    # --- 4. Filter points for finding the alignment ---
    pred_for_align = pred_points_flat[combined_mask]
    gt_for_align = gt_points_flat[combined_mask]

    # Check for enough points to have a stable alignment
    if pred_for_align.shape[0] < 3:
        print("Warning: Fewer than 3 valid points for alignment. Returning identity transformation.")
        identity_transform = {
            'scale': 1.0,
            'rotation': np.eye(3),
            'translation': np.zeros(3)
        }
        return pred_points, identity_transform
        
    # --- 5. Use Umeyama algorithm to find the transformation parameters ---
    scale, rotation, translation = umeyama_alignment(pred_for_align, gt_for_align)

    # --- 6. Apply transformation to the *original, unfiltered* predicted points ---
    aligned_pred_points_flat = scale * (rotation @ pred_points_flat.T).T + translation

    # --- 7. Reshape the aligned points back to the original format ---
    aligned_pred_points = aligned_pred_points_flat.reshape(S, W, H, 3).transpose(0, 3, 1, 2)

    # --- 8. Package the transformation parameters for returning ---
    transform_params = {
        'scale': scale,
        'rotation': rotation,
        'translation': translation
    }

    return aligned_pred_points, transform_params

