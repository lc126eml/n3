
import torch
import logging
from typing import Optional, Tuple
from vggt.utils.geometry import closed_form_inverse_se3
from train_utils.general import check_and_fix_inf_nan

def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")

def _pad_se3_to_4x4_efficient(se3_tensor: torch.Tensor) -> torch.Tensor:
    """
    Pads a batch of 3x4 SE(3) matrices to 4x4 using the 'create and copy' method.
    This is generally the most efficient approach.
    """
    if se3_tensor.shape[-2:] == (4, 4):
        return se3_tensor
    if se3_tensor.shape[-2:] != (3, 4):
        raise ValueError(f"Expected (*, 3, 4) or (*, 4, 4), got {se3_tensor.shape}")

    # 1. Create the final 4x4 tensor with uninitialized memory
    # This is faster than torch.zeros or torch.eye
    target_shape = se3_tensor.shape[:-2] + (4, 4)
    padded_se3 = torch.empty(target_shape, dtype=se3_tensor.dtype, device=se3_tensor.device)

    # 2. Copy the existing 3x4 data into the top rows
    padded_se3[..., :3, :] = se3_tensor

    # 3. Fill the last row with [0, 0, 0, 1]
    # Use with torch.no_grad() to avoid tracking this in autograd
    with torch.no_grad():
        padded_se3[..., 3, :] = 0.0
        padded_se3[..., 3, 3] = 1.0
    
    return padded_se3

def _pad_se3_to_4x4(se3_tensor: torch.Tensor) -> torch.Tensor:
    if se3_tensor.shape[-2:] == (4, 4):
        return se3_tensor
    if se3_tensor.shape[-2:] != (3, 4):
        raise ValueError(f"Expected (*, 3, 4) or (*, 4, 4), got {se3_tensor.shape}")

    device, dtype = se3_tensor.device, se3_tensor.dtype
    bottom_row = torch.zeros(se3_tensor.shape[:-2] + (1, 4), device=device, dtype=dtype)
    bottom_row[..., 0, 3] = 1.0
    return torch.cat([se3_tensor, bottom_row], dim=-2)


def align_camera_and_points_batch_ext(
    poses: torch.Tensor,
    world_points: Optional[torch.Tensor] = None,
    cam_to_world_origin: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Normalize camera poses and 3D points relative to a new origin pose.
    This function handles poses in both (B, S, 3, 4) and (B, S, 4, 4) format.
    
    Args:
        poses: Camera poses (cam_to_world) of shape (B, S, 4, 4) or (B, S, 3, 4).
        world_points: 3D points in original world coords, shape (B, S, H, W, 3) or (*,3).
        cam_to_world_origin: Pose defining the new world origin, shape (B, 4, 4) or (B, 3, 4).
                             If None, the pose of the first camera (poses[:, 0]) is used.
    
    Returns:
        Tuple containing:
        - Aligned camera poses in the same shape as the input `poses`.
        - Aligned world points in the new coordinate system.
    """
    # --- Input Standardization ---
    # Store the original format to return the same shape
    # is_3x4_input = (poses.shape[-2] == 3)
    
    # Pad poses to 4x4 for calculations
    poses_4x4 = _pad_se3_to_4x4_efficient(poses)
    
    # Determine the origin pose and ensure it is also 4x4
    if cam_to_world_origin is None:
        origin_pose_4x4 = poses_4x4[:, 0, :, :]
    else:
        origin_pose_4x4 = _pad_se3_to_4x4_efficient(cam_to_world_origin)
        
    # --- Transformation Logic (in 4x4) ---
    # The transformation is the inverse of the new origin's pose.
    world_to_new_origin = closed_form_inverse_se3(origin_pose_4x4)

    # Align poses: P'_i = (P_origin)^-1 @ P_i
    aligned_poses_4x4 = torch.matmul(world_to_new_origin.unsqueeze(1), poses_4x4)

    # Align points: p' = (P_origin)^-1 @ p
    aligned_world_points = None
    if world_points is not None:
        pts_shape = world_points.shape
        B = pts_shape[0]
        pts_reshaped = world_points.view(B, -1, 3)
        
        ones = torch.ones((B, pts_reshaped.shape[1], 1), device=world_points.device, dtype=world_points.dtype)
        world_points_homog = torch.cat([pts_reshaped, ones], dim=-1)

        transformed_pts_homog = torch.bmm(
            world_to_new_origin, 
            world_points_homog.transpose(1, 2)
        )
        
        transformed_pts_cartesian = transformed_pts_homog.transpose(1, 2)[:, :, :3]
        aligned_world_points = transformed_pts_cartesian.view(*pts_shape)
        
    # --- Output Formatting ---
    # If the original input was 3x4, slice the output back to 3x4
    # if is_3x4_input:
    #     aligned_poses = aligned_poses_4x4[..., :3, :]
    # else:
    #     aligned_poses = aligned_poses_4x4

    return aligned_poses_4x4, aligned_world_points