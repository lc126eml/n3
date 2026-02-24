import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
if os.getenv("TORCH_AUTOGRAD_ANOMALY", "0") == "1":
    torch.autograd.set_detect_anomaly(True)
from dust3r.utils.camera import (
    matrix_to_quaternion,
    relative_pose_absT_quatR,
    quaternion_conjugate,
    quaternion_multiply,
)
from train_utils.general import check_and_fix_inf_nan


eps = 1e-6
class CameraPoseLoss(nn.Module):
    """
    Computes camera pose loss in extrinsic space (3x4 matrices), 
    including an optional relative pose component that encourages 
    geometric consistency among cameras in a scene.
    """
    def __init__(
        self,
        alpha: float = 1.0,
        compute_relative: bool = True,
        compute_absolute: bool = False,
        relative_neighbors: int = -1,
        loss_type: str = "l2",
        beta: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.compute_relative = compute_relative
        self.compute_absolute = compute_absolute
        self.relative_neighbors = relative_neighbors
        self.loss_type = loss_type
        self.beta = beta


    def forward(self, pred_poses, gt_poses, compute_relative=None, compute_absolute=None, relative_neighbors=None):
        """
        Calculates the pose loss.

        Args:
            pred_poses (torch.Tensor): Predicted poses of shape (B, S, 3, 4) in extrinsic space.
            gt_poses (torch.Tensor): Ground truth poses of shape (B, S, 3, 4) in extrinsic space.

        Returns:
            A tuple containing (total_loss, abs_trans_err, abs_rot_err, rel_trans_err, rel_rot_err).
        """
        if compute_relative is None:
            compute_relative = self.compute_relative
        if compute_absolute is None:
            compute_absolute = self.compute_absolute
        if pred_poses.numel() == 0 or (not compute_absolute and not compute_relative):
            zero = torch.tensor(0.0, device=pred_poses.device)
            return zero, zero, zero, zero, zero

        # Extract translation and rotation components
        pred_trans = pred_poses[..., :3, 3]
        gt_trans = gt_poses[..., :3, 3]
        pred_rot = pred_poses[..., :3, :3]
        gt_rot = gt_poses[..., :3, :3]

        abs_trans_err = torch.tensor(0.0, device=pred_poses.device)
        abs_rot_err = torch.tensor(0.0, device=pred_poses.device)
        # Absolute pose error
        if compute_absolute:
            trans_err = self._trans_error(pred_trans, gt_trans)
            abs_trans_err = trans_err.mean()
            rot_err_rad = self._geodesic_distance_from_matrices(pred_rot, gt_rot)
            abs_rot_err = rot_err_rad.mean()

        # Initialize relative errors to zero
        rel_trans_err = torch.tensor(0.0, device=pred_poses.device)
        rel_rot_err = torch.tensor(0.0, device=pred_poses.device)

        # Relative pose error
        if relative_neighbors is None:
            relative_neighbors = self.relative_neighbors
        if compute_relative and pred_poses.shape[1] > 1:
            if relative_neighbors <= 0:
                rel_trans_err, rel_rot_err = self.compute_relative_pose_loss(
                    pred_trans, pred_rot, gt_trans, gt_rot
                )
            else:
                rel_trans_err, rel_rot_err = self.compute_relative_pose_loss_k1(
                    pred_trans, pred_rot, gt_trans, gt_rot
                )

        return abs_trans_err, abs_rot_err, rel_trans_err, rel_rot_err

    @staticmethod
    def _geodesic_distance_from_matrices(R1, R2):
        """Computes the geodesic distance between two batches of rotation matrices."""
        R_rel = torch.matmul(R1, R2.transpose(-1, -2))
        trace = R_rel.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        cos_theta = torch.clamp((trace - 1) / 2.0, -1.0 + eps, 1.0 - eps)
        return torch.acos(cos_theta)

    @staticmethod
    def _geodesic_distance_from_quaternions(q1, q2):
        """Computes the geodesic distance between two batches of quaternions."""
        q_diff = quaternion_multiply(q1, quaternion_conjugate(q2))
        w = q_diff[..., 0].abs()
        cos_half_theta = torch.clamp(w, -1.0 + eps, 1.0 - eps)
        return 2 * torch.acos(cos_half_theta)

    def _trans_error(self, pred_trans, gt_trans):
        # Translation loss type only; rotation always uses raw geodesic angles.
        if self.loss_type == "l2":
            return torch.norm(pred_trans - gt_trans, dim=-1)
        if self.loss_type == "l1":
            return (pred_trans - gt_trans).abs().mean(dim=-1)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(
                pred_trans,
                gt_trans,
                reduction="none",
                beta=self.beta,
            ).mean(dim=-1)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def compute_relative_pose_loss(self, pred_trans, pred_rot, gt_trans, gt_rot):
        """
        Computes the relative pose loss between all pairs of cameras.
        """
        B, S, _ = pred_trans.shape

        # Convert rotations to quaternions
        pred_q = matrix_to_quaternion(pred_rot)
        gt_q = matrix_to_quaternion(gt_rot)

        # Create pairwise matrices
        gt_trans1 = gt_trans.unsqueeze(2).expand(-1, -1, S, -1)
        gt_trans2 = gt_trans.unsqueeze(1).expand(-1, S, -1, -1)
        pred_trans1 = pred_trans.unsqueeze(2).expand(-1, -1, S, -1)
        pred_trans2 = pred_trans.unsqueeze(1).expand(-1, S, -1, -1)

        gt_q1 = gt_q.unsqueeze(2).expand(-1, -1, S, -1)
        gt_q2 = gt_q.unsqueeze(1).expand(-1, S, -1, -1)
        pred_q1 = pred_q.unsqueeze(2).expand(-1, -1, S, -1)
        pred_q2 = pred_q.unsqueeze(1).expand(-1, S, -1, -1)

        # Compute relative poses
        gt_t_rel, gt_q_rel = relative_pose_absT_quatR(gt_trans1, gt_q1, gt_trans2, gt_q2)
        pred_t_rel, pred_q_rel = relative_pose_absT_quatR(pred_trans1, pred_q1, pred_trans2, pred_q2)

        # Compute loss
        rel_trans_err = self._trans_error(pred_t_rel, gt_t_rel)
        rel_rot_err_rad = CameraPoseLoss._geodesic_distance_from_quaternions(pred_q_rel, gt_q_rel)

        # We only need the upper triangle of the S x S matrix, excluding the diagonal
        # to avoid double counting and self-comparison.
        mask = torch.triu(torch.ones(S, S, device=pred_trans.device), diagonal=1).bool()
        
        # Mean of the errors
        rel_trans_err_mean = rel_trans_err[:, mask].mean()
        rel_rot_err_mean = rel_rot_err_rad[:, mask].mean()

        return rel_trans_err_mean, rel_rot_err_mean

    shifts = {}
    @staticmethod
    def calculate_shifts(k_neighbors=2, S=10):
        import numpy as np
        shifts = [1]
        if k_neighbors > 1:
            lin_steps = np.linspace(0, S, k_neighbors - 1 + 2)
            global_shifts = [int(round(s)) for s in lin_steps[1:-1]]
            shifts.extend(global_shifts)
        return shifts

    def compute_relative_pose_loss_k1(self, pred_trans, pred_rot, gt_trans, gt_rot):
        """
        Computes the relative pose loss between each camera 'i' and its
        next neighbors in a circular fashion.

        This is an efficient O(S) implementation.
        """
        B, S, _ = pred_trans.shape

        # --- 1. Convert rotations to quaternions ---
        pred_q = matrix_to_quaternion(pred_rot) # (B, S, 4)
        gt_q = matrix_to_quaternion(gt_rot)     # (B, S, 4)

        # --- 2. Create generalized pairs (i, i+k) ---

        # --- "From" Tensors (pose 'i') ---
        pred_trans1_pairs = pred_trans
        gt_trans1_pairs = gt_trans
        pred_q1_pairs = pred_q
        gt_q1_pairs = gt_q

        # Use list comprehensions for a clean build
        pred_trans2_pairs = torch.roll(pred_trans, shifts=-1, dims=1)
        gt_trans2_pairs = torch.roll(gt_trans, shifts=-1, dims=1)
        pred_q2_pairs = torch.roll(pred_q, shifts=-1, dims=1)
        gt_q2_pairs = torch.roll(gt_q, shifts=-1, dims=1)

        # --- 3. Compute relative poses ---
        # The helper functions receive (B, S * k_neighbors, ...) tensors
        gt_t_rel, gt_q_rel = relative_pose_absT_quatR(gt_trans1_pairs, gt_q1_pairs, gt_trans2_pairs, gt_q2_pairs)
        pred_t_rel, pred_q_rel = relative_pose_absT_quatR(pred_trans1_pairs, pred_q1_pairs, pred_trans2_pairs, pred_q2_pairs)

        # --- 4. Compute loss ---
        # The errors are shape (B, S * k_neighbors)
        rel_trans_err = self._trans_error(pred_t_rel, gt_t_rel)
        rel_rot_err_rad = CameraPoseLoss._geodesic_distance_from_quaternions(pred_q_rel, gt_q_rel)

        # --- 5. Mean of errors ---
        # .mean() averages over all B * S * k_neighbors pairs.
        rel_trans_err_mean = rel_trans_err.mean()
        rel_rot_err_mean = rel_rot_err_rad.mean()

        return rel_trans_err_mean, rel_rot_err_mean
    def compute_relative_pose_loss_distributed_k(self, pred_trans, pred_rot, gt_trans, gt_rot, k_neighbors=1):
        """
        Computes the relative pose loss between each camera 'i' and its 'k'
        next neighbors (i+1, i+2, ..., i+k) in a circular fashion.

        This is an efficient O(S*k) implementation.
        """
        B, S, _ = pred_trans.shape

        # --- 1. Convert rotations to quaternions ---
        pred_q = matrix_to_quaternion(pred_rot) # (B, S, 4)
        gt_q = matrix_to_quaternion(gt_rot)     # (B, S, 4)

        # --- 2. Create generalized pairs (i, i+k) ---

        # --- "From" Tensors (pose 'i') ---
        # We just repeat the original tensor k_neighbors times.
        # Shape will be (B, S * k_neighbors, ...)
        pred_trans1_pairs = torch.cat([pred_trans] * k_neighbors, dim=1)
        gt_trans1_pairs = torch.cat([gt_trans] * k_neighbors, dim=1)
        pred_q1_pairs = torch.cat([pred_q] * k_neighbors, dim=1)
        gt_q1_pairs = torch.cat([gt_q] * k_neighbors, dim=1)

        key = (S, k_neighbors)
        if key not in CameraPoseLoss.shifts:
            CameraPoseLoss.shifts[key] = CameraPoseLoss.calculate_shifts(k_neighbors, S)
        
        # Use list comprehensions for a clean build
        shifts = CameraPoseLoss.shifts[key]
        all_pred_trans2 = [torch.roll(pred_trans, shifts=s, dims=1) for s in shifts]
        all_gt_trans2 = [torch.roll(gt_trans, shifts=s, dims=1) for s in shifts]
        all_pred_q2 = [torch.roll(pred_q, shifts=s, dims=1) for s in shifts]
        all_gt_q2 = [torch.roll(gt_q, shifts=s, dims=1) for s in shifts]

        # Concatenate the list of shifted tensors
        # Shape will also be (B, S * k_neighbors, ...)
        pred_trans2_pairs = torch.cat(all_pred_trans2, dim=1)
        gt_trans2_pairs = torch.cat(all_gt_trans2, dim=1)
        pred_q2_pairs = torch.cat(all_pred_q2, dim=1)
        gt_q2_pairs = torch.cat(all_gt_q2, dim=1)

        # --- 3. Compute relative poses ---
        # The helper functions receive (B, S * k_neighbors, ...) tensors
        gt_t_rel, gt_q_rel = relative_pose_absT_quatR(gt_trans1_pairs, gt_q1_pairs, gt_trans2_pairs, gt_q2_pairs)
        pred_t_rel, pred_q_rel = relative_pose_absT_quatR(pred_trans1_pairs, pred_q1_pairs, pred_trans2_pairs, pred_q2_pairs)

        # --- 4. Compute loss ---
        # The errors are shape (B, S * k_neighbors)
        rel_trans_err = self._trans_error(pred_t_rel, gt_t_rel)
        rel_rot_err_rad = CameraPoseLoss._geodesic_distance_from_quaternions(pred_q_rel, gt_q_rel)

        # --- 5. Mean of errors ---
        # .mean() averages over all B * S * k_neighbors pairs.
        rel_trans_err_mean = rel_trans_err.mean()
        rel_rot_err_mean = rel_rot_err_rad.mean()

        return rel_trans_err_mean, rel_rot_err_mean
    def compute_relative_pose_loss_circular_k(self, pred_trans, pred_rot, gt_trans, gt_rot, k_neighbors=1):
        """
        Computes the relative pose loss between each camera 'i' and its 'k'
        next neighbors (i+1, i+2, ..., i+k) in a circular fashion.

        This is an efficient O(S*k) implementation.
        """
        B, S, _ = pred_trans.shape

        # --- 1. Convert rotations to quaternions ---
        pred_q = matrix_to_quaternion(pred_rot) # (B, S, 4)
        gt_q = matrix_to_quaternion(gt_rot)     # (B, S, 4)

        # --- 2. Create generalized pairs (i, i+k) ---

        # --- "From" Tensors (pose 'i') ---
        # We just repeat the original tensor k_neighbors times.
        # Shape will be (B, S * k_neighbors, ...)
        pred_trans1_pairs = torch.cat([pred_trans] * k_neighbors, dim=1)
        gt_trans1_pairs = torch.cat([gt_trans] * k_neighbors, dim=1)
        pred_q1_pairs = torch.cat([pred_q] * k_neighbors, dim=1)
        gt_q1_pairs = torch.cat([gt_q] * k_neighbors, dim=1)

        # --- "To" Tensors (poses 'i+1' through 'i+k') ---
        # We build lists of the k shifted tensors, then concatenate once.
        
        # Use list comprehensions for a clean build
        all_pred_trans2 = [torch.roll(pred_trans, shifts=-k, dims=1) for k in range(1, k_neighbors + 1)]
        all_gt_trans2 = [torch.roll(gt_trans, shifts=-k, dims=1) for k in range(1, k_neighbors + 1)]
        all_pred_q2 = [torch.roll(pred_q, shifts=-k, dims=1) for k in range(1, k_neighbors + 1)]
        all_gt_q2 = [torch.roll(gt_q, shifts=-k, dims=1) for k in range(1, k_neighbors + 1)]

        # Concatenate the list of shifted tensors
        # Shape will also be (B, S * k_neighbors, ...)
        pred_trans2_pairs = torch.cat(all_pred_trans2, dim=1)
        gt_trans2_pairs = torch.cat(all_gt_trans2, dim=1)
        pred_q2_pairs = torch.cat(all_pred_q2, dim=1)
        gt_q2_pairs = torch.cat(all_gt_q2, dim=1)

        # --- 3. Compute relative poses ---
        # The helper functions receive (B, S * k_neighbors, ...) tensors
        gt_t_rel, gt_q_rel = relative_pose_absT_quatR(gt_trans1_pairs, gt_q1_pairs, gt_trans2_pairs, gt_q2_pairs)
        pred_t_rel, pred_q_rel = relative_pose_absT_quatR(pred_trans1_pairs, pred_q1_pairs, pred_trans2_pairs, pred_q2_pairs)

        # --- 4. Compute loss ---
        # The errors are shape (B, S * k_neighbors)
        rel_trans_err = self._trans_error(pred_t_rel, gt_t_rel)
        rel_rot_err_rad = CameraPoseLoss._geodesic_distance_from_quaternions(pred_q_rel, gt_q_rel)

        # --- 5. Mean of errors ---
        # .mean() averages over all B * S * k_neighbors pairs.
        rel_trans_err_mean = rel_trans_err.mean()
        rel_rot_err_mean = rel_rot_err_rad.mean()

        return rel_trans_err_mean, rel_rot_err_mean

class PoseEncodingLoss(nn.Module):
    """
    Computes loss for camera intrinsics (e.g., focal length).

    This module is designed to replicate and replace the 'loss_FL' component
    from your original 'camera_loss_single' function, but in a modular
    nn.Module format.

    It operates on the raw pose encoding vector.
    """
    def __init__(self, loss_type: str = "l1", beta: float = 1.0):
        """
        Initializes the loss module.

        Args:
            loss_type (str): "l1" (Mean Absolute Error), 
                             "l2" (Euclidean Norm), 
                             or "smooth_l1".
        """
        super().__init__()
        self.loss_type = loss_type

        self.beta = beta
        if loss_type == "smooth_l1":
            # Use reduction='none' to get per-element loss,
            # which we will then average.
            self.loss_fn = nn.SmoothL1Loss(reduction='none', beta=beta)
        elif loss_type not in ["l1", "l2"]:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, pred_pose_enc: torch.Tensor, gt_pose_enc: torch.Tensor, max_val: Optional[float] = None) -> torch.Tensor:
        """
        Calculates the intrinsics loss.

        Args:
            pred_pose_enc (torch.Tensor): (..., D) predicted pose encoding.
            gt_pose_enc (torch.Tensor): (..., D) ground truth pose encoding.
            max_val (float, optional): Maximum value to clamp the loss to. Defaults to None.

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        
        # 2. Handle empty tensor case
        if pred_pose_enc.numel() == 0:
            return (pred_pose_enc.sum() * 0)

        # 3. Calculate the loss based on the specified type
        if self.loss_type == "l1":
            # Mean Absolute Error over all individual parameters.
            # Shape (..., K) -> (..., K)
            loss = (pred_pose_enc - gt_pose_enc).abs()
        
        elif self.loss_type == "l2":
            # Mean Euclidean distance of the intrinsic vectors.
            # Shape (..., K) -> (...)
            loss = torch.norm(pred_pose_enc - gt_pose_enc, p=2, dim=-1)
        
        elif self.loss_type == "smooth_l1":
            # Smooth L1 loss over all individual parameters.
            # Shape (..., K) -> (..., K)
            loss = self.loss_fn(pred_pose_enc, gt_pose_enc)

        loss = check_and_fix_inf_nan(loss, "loss_FL")
        if max_val is not None:
            loss = loss.clamp(max=max_val)

        # 5. Return the final mean scalar loss
        return loss.mean()
