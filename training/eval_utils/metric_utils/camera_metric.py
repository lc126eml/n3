# pose_metrics_final.py

import torch
import math
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
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos_theta)

def _compare_translation_by_angle(t_gt: torch.Tensor, t_pred: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """Computes the angle between two sets of translation vectors (scale-invariant)."""
    t_pred_norm = torch.linalg.norm(t_pred, dim=1, keepdim=True)
    t_pred_normalized = t_pred / (t_pred_norm + eps)
    t_gt_norm = torch.linalg.norm(t_gt, dim=1, keepdim=True)
    t_gt_normalized = t_gt / (t_gt_norm + eps)
    
    dot_product = torch.sum(t_pred_normalized * t_gt_normalized, dim=1)
    dot_product_clamped = torch.clamp(dot_product, -1.0, 1.0)
    return torch.acos(dot_product_clamped)


# --- Metric Calculation Functions ---

def compute_absolute_pose_error(poses_gt: torch.Tensor, poses_pred: torch.Tensor) -> Dict[str, any]:
    """Calculates absolute rotation and translation (L2 norm) error metrics."""
    R_gt, t_gt = poses_gt[:, :3, :3], poses_gt[:, :3, 3]
    R_pred, t_pred = poses_pred[:, :3, :3], poses_pred[:, :3, 3]
    
    rot_errors_rad = _so3_relative_angle(R_gt, R_pred)
    rot_errors_deg = torch.rad2deg(rot_errors_rad)
    trans_errors = torch.linalg.norm(t_pred - t_gt, dim=1)

    return {
        'rot_error_mean_deg': rot_errors_deg.mean().item(),
        'trans_error_mean': trans_errors.mean().item(),
        'rot_errors_deg': rot_errors_deg,
        'trans_errors': trans_errors,
    }

def compute_consecutive_relative_error(poses_gt: torch.Tensor, poses_pred: torch.Tensor) -> Dict[str, float]:
    """Calculates CONSECUTIVE-FRAME relative pose error (RRA, RTA using L2 norm)."""
    if poses_gt.shape[0] < 2:
        return {'rra_mean_deg': float('nan'), 'rta_mean': float('nan')}

    poses_gt_inv = _closed_form_inverse(poses_gt)
    poses_pred_inv = _closed_form_inverse(poses_pred)
    trans_gt_rel = torch.matmul(poses_gt_inv[1:], poses_gt[:-1])
    trans_pred_rel = torch.matmul(poses_pred_inv[1:], poses_pred[:-1])
    
    R_gt_rel, t_gt_rel = trans_gt_rel[:, :3, :3], trans_gt_rel[:, :3, 3]
    R_pred_rel, t_pred_rel = trans_pred_rel[:, :3, :3], trans_pred_rel[:, :3, 3]
    
    rra_rad = _so3_relative_angle(R_gt_rel, R_pred_rel)
    rra_deg = torch.rad2deg(rra_rad)
    rta = torch.linalg.norm(t_pred_rel - t_gt_rel, dim=1)

    return {
        'rra_mean_deg': rra_deg.mean().item(),
        'rta_mean': rta.mean().item(),
    }

def compute_absolute_pose_error_angle(poses_gt: torch.Tensor, poses_pred: torch.Tensor) -> Dict[str, any]:
    """Calculates absolute rotation and translation (angular) error metrics."""
    # print(poses_gt.shape, poses_pred.shape)
    R_gt, t_gt = poses_gt[..., :3, :3], poses_gt[..., :3, 3]
    R_pred, t_pred = poses_pred[..., :3, :3], poses_pred[..., :3, 3]
    
    rot_errors_rad = _so3_relative_angle(R_gt, R_pred)
    rot_errors_deg = torch.rad2deg(rot_errors_rad)
    trans_angle_errors_rad = _compare_translation_by_angle(t_gt, t_pred)
    trans_angle_errors_deg = torch.rad2deg(trans_angle_errors_rad)
    trans_errors = torch.linalg.norm(t_pred - t_gt, dim=1)

    return {
        'rot_error_mean_deg': rot_errors_deg.mean().item(),
        'trans_angle_error_mean_deg': trans_angle_errors_deg.mean().item(),
        'trans_error_mean': trans_errors.mean().item(),
    }, rot_errors_deg, trans_angle_errors_deg, trans_errors


def compute_consecutive_relative_error_angle(poses_gt: torch.Tensor, poses_pred: torch.Tensor) -> Dict[str, float]:
    """Calculates CONSECUTIVE-FRAME relative pose error (RRA, RTA using angle)."""
    if poses_gt.shape[0] < 2:
        return {'rra_mean_deg': float('nan'), 'rta_angle_mean_deg': float('nan')}

    poses_gt_inv = _closed_form_inverse(poses_gt)
    poses_pred_inv = _closed_form_inverse(poses_pred)
    trans_gt_rel = torch.matmul(poses_gt_inv[1:], poses_gt[:-1])
    trans_pred_rel = torch.matmul(poses_pred_inv[1:], poses_pred[:-1])
    
    R_gt_rel, t_gt_rel = trans_gt_rel[:, :3, :3], trans_gt_rel[:, :3, 3]
    R_pred_rel, t_pred_rel = trans_pred_rel[:, :3, :3], trans_pred_rel[:, :3, 3]
    
    rra_rad = _so3_relative_angle(R_gt_rel, R_pred_rel)
    rra_deg = torch.rad2deg(rra_rad)
    rta_angle_rad = _compare_translation_by_angle(t_gt_rel, t_pred_rel)
    rta_angle_deg = torch.rad2deg(rta_angle_rad)

    return {
        'rra_mean_deg': rra_deg.mean().item(),
        'rta_angle_mean_deg': rta_angle_deg.mean().item(),
    }
    
def compute_all_pairs_relative_error(poses_gt: torch.Tensor, poses_pred: torch.Tensor) -> Dict[str, float]:
    """Calculates ALL-PAIRS relative pose error (scale-invariant)."""
    batch_size = poses_gt.shape[0]
    if batch_size < 2:
        return {'all_pairs_rot_error_deg': float('nan'), 'all_pairs_trans_angle_error_deg': float('nan')}
    
    idx1, idx2 = torch.combinations(torch.arange(batch_size, device=poses_gt.device)).unbind(-1)
    relative_pose_gt = _closed_form_inverse(poses_gt[idx1]) @ poses_gt[idx2]
    relative_pose_pred = _closed_form_inverse(poses_pred[idx1]) @ poses_pred[idx2]
    
    rot_gt, rot_pred = relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    rel_rot_error_rad = _so3_relative_angle(rot_gt, rot_pred, eps=1e-5)
    rel_rot_error_deg = torch.rad2deg(rel_rot_error_rad)

    t_gt, t_pred = relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    rel_trans_angle_error_rad = _compare_translation_by_angle(t_gt, t_pred)
    rel_trans_angle_error_deg = torch.rad2deg(rel_trans_angle_error_rad)

    return {
        'all_pairs_rot_error_deg': rel_rot_error_deg.mean().item(),
        'all_pairs_trans_angle_error_deg': rel_trans_angle_error_deg.mean().item(),
    }

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

def calculate_auc(r_errors_deg: torch.Tensor, t_errors_deg: torch.Tensor, max_threshold_deg: int = 30) -> float:
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
    scale_invariant_errors = compute_absolute_pose_error_angle(poses_gt, poses_pred)
    
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
        t_errors=scale_aware_errors['trans_errors_m'],
        t_thresholds=[0.05, 0.10, 0.15] # Thresholds in meters
    )
    print("\nStrict Accuracy (Rotation & Translation):")
    for key, val in strict_acc.items():
        print(f"  {key}: {val:.3f}")

    # --- 3. Calculate AUC ---
    print("\n--- 📈 Area Under the Curve (AUC) ---")
    auc_score = calculate_auc(
        r_errors_deg=scale_invariant_errors['rot_errors_deg'],
        t_errors_deg=scale_invariant_errors['trans_angle_errors_deg'],
        max_threshold_deg=15
    )
    print(f"Strict AUC (max error < threshold) up to 15°: {auc_score:.4f}")