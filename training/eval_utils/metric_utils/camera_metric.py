# pose_metrics_final.py

import torch
from typing import Tuple, List, Dict, Optional

# --- Core Helper Functions ---

def _closed_form_inverse(se3: torch.Tensor) -> torch.Tensor:
    """Computes the inverse of a batch of 4x4 SE(3) matrices using the closed-form solution."""
    R = se3[:, :3, :3]
    t = se3[:, :3, 3].unsqueeze(2)
    R_transposed = R.transpose(1, 2)
    t_inv = -torch.bmm(R_transposed, t)
    inv_se3 = torch.zeros_like(se3)
    inv_se3[:, :3, :3] = R_transposed
    inv_se3[:, :3, 3] = t_inv.squeeze(2)
    inv_se3[:, 3, 3] = 1.0
    return inv_se3

def _so3_relative_angle(R1: torch.Tensor, R2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes the geodesic rotation angle between two batches of rotation matrices."""
    R_rel = torch.matmul(R1.transpose(-2, -1), R2)
    trace = torch.diagonal(R_rel, offset=0, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0)
    return torch.acos(cos_theta)

def _compare_translation_by_angle(t_gt: torch.Tensor, t_pred: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """Computes the angle between two sets of translation vectors (scale-invariant)."""
    t_pred_norm = torch.linalg.norm(t_pred, dim=-1, keepdim=True)
    t_pred_normalized = t_pred / (t_pred_norm + eps)
    t_gt_norm = torch.linalg.norm(t_gt, dim=-1, keepdim=True)
    t_gt_normalized = t_gt / (t_gt_norm + eps)
    
    dot_product = torch.sum(t_pred_normalized * t_gt_normalized, dim=-1)
    dot_product_clamped = torch.clamp(dot_product, -1.0, 1.0)
    angle = torch.acos(dot_product_clamped)
    both_zero = (t_pred_norm.squeeze(-1) <= eps) & (t_gt_norm.squeeze(-1) <= eps)
    return torch.where(both_zero, torch.zeros_like(angle), angle)


# --- Metric Calculation Functions ---

def compute_absolute_pose_error(poses_gt: torch.Tensor, poses_pred: torch.Tensor) -> Dict[str, any]:
    """Calculates absolute rotation and translation (L2 norm) error metrics."""
    R_gt, t_gt = poses_gt[:, :3, :3], poses_gt[:, :3, 3]
    R_pred, t_pred = poses_pred[:, :3, :3], poses_pred[:, :3, 3]
    
    rot_errors_rad = _so3_relative_angle(R_gt, R_pred)
    rot_errors_deg = torch.rad2deg(rot_errors_rad)
    trans_errors = torch.linalg.norm(t_pred - t_gt, dim=-1)

    return {
        'rot_error_mean_deg': rot_errors_deg.mean().item(),
        'trans_error_mean': trans_errors.mean().item(),
        'rot_errors_deg': rot_errors_deg,
        'trans_errors': trans_errors,
    }

def compute_consecutive_relative_error(poses_gt: torch.Tensor, poses_pred: torch.Tensor, pose_convention: str = "c2w") -> Dict[str, float]:
    """Calculates CONSECUTIVE-FRAME relative pose error (RRA, RTA using L2 norm)."""
    if poses_gt.shape[0] < 2:
        return {'rra_mean_deg': float('nan'), 'rta_mean': float('nan')}

    if pose_convention == "c2w":
        poses_gt_inv = _closed_form_inverse(poses_gt)
        poses_pred_inv = _closed_form_inverse(poses_pred)
        trans_gt_rel = torch.matmul(poses_gt_inv[:-1], poses_gt[1:])
        trans_pred_rel = torch.matmul(poses_pred_inv[:-1], poses_pred[1:])
    elif pose_convention == "w2c":
        poses_gt_inv = _closed_form_inverse(poses_gt)
        poses_pred_inv = _closed_form_inverse(poses_pred)
        trans_gt_rel = torch.matmul(poses_gt[:-1], poses_gt_inv[1:])
        trans_pred_rel = torch.matmul(poses_pred[:-1], poses_pred_inv[1:])
    else:
        raise ValueError(f"pose_convention must be 'c2w' or 'w2c', got {pose_convention!r}")
    
    R_gt_rel, t_gt_rel = trans_gt_rel[:, :3, :3], trans_gt_rel[:, :3, 3]
    R_pred_rel, t_pred_rel = trans_pred_rel[:, :3, :3], trans_pred_rel[:, :3, 3]
    
    rra_rad = _so3_relative_angle(R_gt_rel, R_pred_rel)
    rra_deg = torch.rad2deg(rra_rad)
    rta = torch.linalg.norm(t_pred_rel - t_gt_rel, dim=1)

    return {
        'rra_mean_deg': rra_deg.mean().item(),
        'rta_mean': rta.mean().item(),
    }

def compute_absolute_pose_error_angle(poses_gt: torch.Tensor, poses_pred: torch.Tensor) -> Tuple[Dict[str, any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates absolute rotation and translation (angular) error metrics."""
    # print(poses_gt.shape, poses_pred.shape)
    R_gt, t_gt = poses_gt[..., :3, :3], poses_gt[..., :3, 3]
    R_pred, t_pred = poses_pred[..., :3, :3], poses_pred[..., :3, 3]
    
    rot_errors_rad = _so3_relative_angle(R_gt, R_pred)
    rot_errors_deg = torch.rad2deg(rot_errors_rad)
    trans_angle_errors_rad = _compare_translation_by_angle(t_gt, t_pred)
    trans_angle_errors_deg = torch.rad2deg(trans_angle_errors_rad)
    trans_errors = torch.linalg.norm(t_pred - t_gt, dim=-1)

    return {
        'rot_error_mean_deg': rot_errors_deg.mean().item(),
        'trans_angle_error_mean_deg': trans_angle_errors_deg.mean().item(),
        'trans_error_mean': trans_errors.mean().item(),
    }, rot_errors_deg, trans_angle_errors_deg, trans_errors


def compute_consecutive_relative_error_angle(poses_gt: torch.Tensor, poses_pred: torch.Tensor, pose_convention: str = "c2w") -> Dict[str, float]:
    """Calculates CONSECUTIVE-FRAME relative pose error (RRA, RTA using angle)."""
    if poses_gt.shape[0] < 2:
        return {'rra_mean_deg': float('nan'), 'rta_angle_mean_deg': float('nan')}

    if pose_convention == "c2w":
        poses_gt_inv = _closed_form_inverse(poses_gt)
        poses_pred_inv = _closed_form_inverse(poses_pred)
        trans_gt_rel = torch.matmul(poses_gt_inv[:-1], poses_gt[1:])
        trans_pred_rel = torch.matmul(poses_pred_inv[:-1], poses_pred[1:])
    elif pose_convention == "w2c":
        poses_gt_inv = _closed_form_inverse(poses_gt)
        poses_pred_inv = _closed_form_inverse(poses_pred)
        trans_gt_rel = torch.matmul(poses_gt[:-1], poses_gt_inv[1:])
        trans_pred_rel = torch.matmul(poses_pred[:-1], poses_pred_inv[1:])
    else:
        raise ValueError(f"pose_convention must be 'c2w' or 'w2c', got {pose_convention!r}")
    
    R_gt_rel, t_gt_rel = trans_gt_rel[:, :3, :3], trans_gt_rel[:, :3, 3]
    R_pred_rel, t_pred_rel = trans_pred_rel[:, :3, :3], trans_pred_rel[:, :3, 3]
    
    rra_rad = _so3_relative_angle(R_gt_rel, R_pred_rel)
    rra_deg = torch.rad2deg(rra_rad)
    rta_angle_rad = _compare_translation_by_angle(t_gt_rel, t_pred_rel)
    rta_angle_deg = torch.rad2deg(rta_angle_rad)
    rta_angle_deg = torch.minimum(rta_angle_deg, (180.0 - rta_angle_deg).abs())

    return {
        'rra_mean_deg': rra_deg.mean().item(),
        'rta_angle_mean_deg': rta_angle_deg.mean().item(),
    }
    
def compute_all_pairs_relative_error(poses_gt: torch.Tensor, poses_pred: torch.Tensor, pose_convention: str = "c2w") -> Dict[str, float]:
    """Calculates ALL-PAIRS relative pose error (scale-invariant)."""
    metrics = compute_all_pairs_relative_metrics(
        poses_gt,
        poses_pred,
        pose_convention=pose_convention,
        include_auc=False,
    )
    return {k: v for k, v in metrics.items() if k != 'auc'}


def compute_all_pairs_relative_error_tensors(
    poses_gt: torch.Tensor,
    poses_pred: torch.Tensor,
    pose_convention: str = "c2w",
    translation_ambiguity: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns all-pairs relative rotation, translation-angle, and translation-L2 errors.

    The first two tensors are suitable for VGGT-style pose AUC: each pair is
    scored by max(rotation_error_deg, translation_angle_error_deg).
    """
    batch_size = poses_gt.shape[0]
    if batch_size < 2:
        empty = poses_gt.new_empty((0,))
        return empty, empty, empty

    idx1, idx2 = torch.combinations(torch.arange(batch_size, device=poses_gt.device)).unbind(-1)
    poses_gt_inv = _closed_form_inverse(poses_gt)
    poses_pred_inv = _closed_form_inverse(poses_pred)
    if pose_convention == "c2w":
        relative_pose_gt = poses_gt_inv[idx1] @ poses_gt[idx2]
        relative_pose_pred = poses_pred_inv[idx1] @ poses_pred[idx2]
    elif pose_convention == "w2c":
        relative_pose_gt = poses_gt[idx1] @ poses_gt_inv[idx2]
        relative_pose_pred = poses_pred[idx1] @ poses_pred_inv[idx2]
    else:
        raise ValueError(f"pose_convention must be 'c2w' or 'w2c', got {pose_convention!r}")
    
    rot_gt, rot_pred = relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    rel_rot_error_rad = _so3_relative_angle(rot_gt, rot_pred, eps=1e-5)
    rel_rot_error_deg = torch.rad2deg(rel_rot_error_rad)

    t_gt, t_pred = relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    rel_trans_error = torch.linalg.norm(t_pred - t_gt, dim=-1)
    rel_trans_angle_error_rad = _compare_translation_by_angle(t_gt, t_pred)
    rel_trans_angle_error_deg = torch.rad2deg(rel_trans_angle_error_rad)
    if translation_ambiguity:
        rel_trans_angle_error_deg = torch.minimum(
            rel_trans_angle_error_deg,
            (180.0 - rel_trans_angle_error_deg).abs(),
        )

    return rel_rot_error_deg, rel_trans_angle_error_deg, rel_trans_error


def compute_all_pairs_relative_metrics(
    poses_gt: torch.Tensor,
    poses_pred: torch.Tensor,
    pose_convention: str = "c2w",
    include_auc: bool = False,
    max_threshold_deg: int = 30,
) -> Dict[str, float]:
    """Calculates mean all-pairs relative pose metrics for one scene."""
    rel_rot_error_deg, rel_trans_angle_error_deg, rel_trans_error = compute_all_pairs_relative_error_tensors(
        poses_gt,
        poses_pred,
        pose_convention=pose_convention,
        translation_ambiguity=True,
    )
    if rel_rot_error_deg.numel() == 0:
        metrics = {
            'all_pairs_rot_error_deg': float('nan'),
            'all_pairs_trans_angle_error_deg': float('nan'),
            'all_pairs_trans_error': float('nan'),
        }
    else:
        metrics = {
            'all_pairs_rot_error_deg': rel_rot_error_deg.mean().item(),
            'all_pairs_trans_angle_error_deg': rel_trans_angle_error_deg.mean().item(),
            'all_pairs_trans_error': rel_trans_error.mean().item(),
        }

    if include_auc:
        metrics['auc'] = calculate_auc_vggt_style(
            rel_rot_error_deg,
            rel_trans_angle_error_deg,
            max_threshold_deg=max_threshold_deg,
        )
    return metrics


def compute_batched_all_pairs_relative_metrics(
    poses_gt: torch.Tensor,
    poses_pred: torch.Tensor,
    pose_convention: str = "c2w",
    include_auc: bool = False,
    max_threshold_deg: int = 30,
) -> Dict[str, float]:
    """Calculates scene-averaged all-pairs relative metrics for [S,4,4] or [B,S,4,4] poses."""
    if poses_gt.shape != poses_pred.shape:
        raise ValueError(f"Pose shape mismatch: gt={poses_gt.shape}, pred={poses_pred.shape}")

    if poses_gt.ndim == 3:
        poses_gt = poses_gt.unsqueeze(0)
        poses_pred = poses_pred.unsqueeze(0)
    elif poses_gt.ndim != 4:
        raise ValueError(f"Expected poses shaped [S,4,4] or [B,S,4,4], got {poses_gt.shape}")

    per_scene_metrics = [
        compute_all_pairs_relative_metrics(
            gt_scene,
            pred_scene,
            pose_convention=pose_convention,
            include_auc=include_auc,
            max_threshold_deg=max_threshold_deg,
        )
        for gt_scene, pred_scene in zip(poses_gt, poses_pred)
    ]
    keys = per_scene_metrics[0].keys() if per_scene_metrics else ()
    result = {}
    for key in keys:
        vals = [m[key] for m in per_scene_metrics if m[key] == m[key]]
        result[key] = float(sum(vals) / len(vals)) if vals else float('nan')
    return result

# --- Accuracy Calculation ---
def calculate_accuracy_metrics_rot(rot_errors_deg: torch.Tensor, thresholds_deg: List[int]) -> Dict[str, float]:
    """Calculates accuracy at various thresholds and an AUC proxy (mAA)."""
    results = {}
    accuracies = []
    for t in sorted(thresholds_deg):
        accuracy = (rot_errors_deg < t).float().mean().item()
        accuracies.append(accuracy)
        results[f'acc_{t}_deg'] = accuracy
    results['auc_proxy_mAA'] = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return results

def calculate_accuracy_metrics(
    rot_errors_deg: torch.Tensor, 
    rot_thresholds_deg: List[float],
    t_errors: Optional[torch.Tensor] = None,
    t_thresholds: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculates accuracy at various thresholds for rotation and optionally translation.
    If translation errors/thresholds are provided, accuracy requires BOTH errors to be below their thresholds.
    """
    results = {}
    
    # Ensure thresholds are sorted
    rot_thresholds_deg = sorted(rot_thresholds_deg)
    
    # Calculate rotation accuracy
    rot_accuracies = (rot_errors_deg.unsqueeze(1) < torch.tensor(rot_thresholds_deg, device=rot_errors_deg.device).unsqueeze(0)).float()
    
    if t_errors is not None and t_thresholds is not None:
        t_thresholds = sorted(t_thresholds)
        if len(rot_thresholds_deg) != len(t_thresholds):
            raise ValueError("Rotation and translation threshold lists must have the same length.")
            
        # Calculate translation accuracy
        t_accuracies = (t_errors.unsqueeze(1) < torch.tensor(t_thresholds, device=t_errors.device).unsqueeze(0)).float()
        
        # Combined accuracy requires both to be true
        accuracies = (rot_accuracies * t_accuracies).mean(dim=0)
        
        for i, (r_thresh, t_thresh) in enumerate(zip(rot_thresholds_deg, t_thresholds)):
            results[f'acc_{r_thresh}deg_{t_thresh:.2f}m'] = accuracies[i].item()
    else:
        accuracies = rot_accuracies.mean(dim=0)
        for i, r_thresh in enumerate(rot_thresholds_deg):
            results[f'acc_{r_thresh}deg'] = accuracies[i].item()
    
    results['mAA'] = accuracies.mean().item()
    return results

def calculate_auc_trapezoid(r_errors_deg: torch.Tensor, t_errors_deg: torch.Tensor, max_threshold_deg: int = 30) -> float:
    """
    Calculates the standard Area Under the Curve (AUC) using the trapezoidal rule.
    This is a strict metric where a pose is correct only if BOTH rotation and
    translation angle errors are below the threshold.
    """
    # Use the max of the two errors for a holistic evaluation
    max_errors = torch.max(r_errors_deg, t_errors_deg)

    thresholds = torch.linspace(0, max_threshold_deg, max_threshold_deg + 1).to(r_errors_deg.device)
    
    # accuracies[i] = percentage of poses with error < thresholds[i]
    accuracies = (max_errors.unsqueeze(1) < thresholds.unsqueeze(0)).float().mean(dim=0)
    
    # Integrate the accuracy curve using the trapezoidal rule and normalize
    auc = torch.trapezoid(accuracies, thresholds) / max_threshold_deg
    return auc.item()


def calculate_auc_vggt_style(r_errors_deg: torch.Tensor, t_errors_deg: torch.Tensor, max_threshold_deg: int = 30) -> float:
    """Calculates VGGT/MapAnything-style discrete pose AUC.

    This matches np.histogram(max_error, bins=np.arange(T + 1)) followed by the
    mean cumulative histogram. Values above T are counted as outliers.
    """
    if max_threshold_deg <= 0:
        raise ValueError(f"max_threshold_deg must be positive, got {max_threshold_deg}")
    if r_errors_deg.shape != t_errors_deg.shape:
        raise ValueError(f"Error shape mismatch: r={r_errors_deg.shape}, t={t_errors_deg.shape}")
    if r_errors_deg.numel() == 0 or t_errors_deg.numel() == 0:
        return float("nan")

    max_errors = torch.maximum(r_errors_deg, t_errors_deg)
    valid = torch.isfinite(max_errors) & (max_errors >= 0) & (max_errors <= max_threshold_deg)
    max_errors = max_errors[valid]

    num_pairs = float(r_errors_deg.numel())
    if num_pairs == 0:
        return float("nan")

    if max_errors.numel() == 0:
        return 0.0

    bin_idx = torch.floor(max_errors).to(torch.long)
    bin_idx = torch.clamp(bin_idx, max=max_threshold_deg - 1)
    histogram = torch.bincount(bin_idx, minlength=max_threshold_deg).to(dtype=r_errors_deg.dtype)
    auc = torch.cumsum(histogram / num_pairs, dim=0).mean()
    return auc.item()


def calculate_all_pairs_auc_vggt_style(
    poses_gt: torch.Tensor,
    poses_pred: torch.Tensor,
    pose_convention: str = "c2w",
    max_threshold_deg: int = 30,
) -> float:
    """Calculates batched all-pairs relative pose AUC using VGGT semantics.

    Args:
        poses_gt: Ground-truth poses shaped [S, 4, 4] or [B, S, 4, 4].
        poses_pred: Predicted poses with the same shape as poses_gt.
        pose_convention: Pose convention, either "c2w" or "w2c".
        max_threshold_deg: AUC threshold in degrees.

    Returns:
        Mean scene-level AUC. Scenes with fewer than two poses are skipped.
    """
    return compute_batched_all_pairs_relative_metrics(
        poses_gt,
        poses_pred,
        pose_convention=pose_convention,
        include_auc=True,
        max_threshold_deg=max_threshold_deg,
    )['auc']


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Running example on {device.type.upper()}.\n")
    N = 100
    
    # --- Generate synthetic data ---
    R_gt = torch.as_tensor(torch.rand(N, 3, 3), dtype=torch.float32, device=device)
    R_gt, _ = torch.linalg.qr(R_gt)
    t_gt = torch.randn(N, 3, device=device)
    poses_gt = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
    poses_gt[:, :3, :3] = R_gt
    poses_gt[:, :3, 3] = t_gt

    rot_noise = torch.randn(N, 3, 3, device=device) * 0.2
    R_pred, _ = torch.linalg.qr(R_gt + rot_noise)
    t_noise = torch.randn(N, 3, device=device) * 0.2
    t_pred = t_gt + t_noise
    scale_factor = 1.5 # Simulate scale drift
    poses_pred = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
    poses_pred[:, :3, :3] = R_pred
    poses_pred[:, :3, 3] = t_pred * scale_factor
    
    # --- 1. Calculate Errors ---
    # Scale-aware errors (translation in meters)
    scale_aware_errors = compute_absolute_pose_error(poses_gt, poses_pred)
    # Scale-invariant errors (translation in degrees)
    scale_invariant_errors, rot_errors_deg, trans_angle_errors_deg, _ = compute_absolute_pose_error_angle(poses_gt, poses_pred)
    
    print("--- 📏 Scale-Aware Error ---")
    print(f"Mean Rotation Error: {scale_aware_errors['rot_error_mean_deg']:.2f}°")
    print(f"Mean Translation Error: {scale_aware_errors['trans_error_mean']:.2f}m\n")

    print("--- ⚖️ Scale-Invariant Error ---")
    print(f"Mean Translation Angle Error: {scale_invariant_errors['trans_angle_error_mean_deg']:.2f}°\n")

    # --- 2. Calculate Accuracy (mAA) ---
    print("--- 🎯 Accuracy Metrics (mAA) ---")
    # A) Rotation Only
    rot_only_acc = calculate_accuracy_metrics(
        rot_errors_deg=scale_aware_errors['rot_errors_deg'],
        rot_thresholds_deg=[5, 10, 15]
    )
    print("Rotation-Only Accuracy:")
    for key, val in rot_only_acc.items():
        print(f"  {key}: {val:.3f}")

    # B) Strict: Both Rotation (deg) and Translation (meters) must be below thresholds
    strict_acc = calculate_accuracy_metrics(
        rot_errors_deg=scale_aware_errors['rot_errors_deg'], 
        rot_thresholds_deg=[5, 10, 15],
        t_errors=scale_aware_errors['trans_errors'],
        t_thresholds=[0.05, 0.10, 0.15] # Thresholds in meters
    )
    print("\nStrict Accuracy (Rotation & Translation):")
    for key, val in strict_acc.items():
        print(f"  {key}: {val:.3f}")

    # --- 3. Calculate AUC ---
    print("\n--- 📈 Area Under the Curve (AUC) ---")
    auc_score = calculate_auc_vggt_style(
        r_errors_deg=rot_errors_deg,
        t_errors_deg=trans_angle_errors_deg,
        max_threshold_deg=15
    )
    print(f"Strict AUC (max error < threshold) up to 15°: {auc_score:.4f}")
