import enum
import logging
import math
import sys
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

inf = float("inf")



def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def camera_to_pose_encoding(
    camera,
    pose_encoding_type="absT_quaR",
):
    """
    Inverse to pose_encoding_to_camera
    camera: opencv, cam2world
    """
    if pose_encoding_type == "absT_quaR":

        quaternion_R = matrix_to_quaternion(camera[:, :3, :3])

        pose_encoding = torch.cat([camera[:, :3, 3], quaternion_R], dim=-1)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    return pose_encoding


def quaternion_to_matrix(quaternions: torch.Tensor, epsilon = 1e-12) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)

    # Calculate the squared norm
    sum_sq = (quaternions * quaternions).sum(-1)

    # --- Robustness Fix ---
    # Identify near-zero quaternions
    mask = sum_sq < epsilon
    if mask.any():
        logging.warning("Near-zero norm quaternion detected. Clamping to prevent division by zero.")
        # Use torch.clamp to ensure the denominator is not too small, only for problematic entries.
        sum_sq = torch.clamp(sum_sq, min=epsilon)
    # ----------------------

    # The conversion formula
    two_s = 2.0 / sum_sq

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def is_valid_camera_pose_torch(P_batch, atol=3e-3, encoding=''):
    """
    Checks if a batch of 4x4 PyTorch tensors P_batch represents valid camera poses.
    P_batch is expected to have shape (B, 4, 4).

    Args:
        P_batch (torch.Tensor): Batch of poses, shape (B, 4, 4).
        atol (float): Absolute tolerance for floating point comparisons.
                      The relative tolerance (rtol) for comparisons is set to PyTorch's
                      default for `isclose` (1e-5).

    Returns:
        torch.Tensor: A boolean tensor of shape (B,) indicating validity for each pose.

    Raises:
        TypeError: If P_batch is not a PyTorch tensor.
        ValueError: If P_batch does not have ndim=3 or shape[1:] != (4,4).
    """
    if not isinstance(P_batch, torch.Tensor):
        raise TypeError(f"Input P_batch must be a PyTorch tensor, got {type(P_batch)}")

    if P_batch.ndim != 3 or P_batch.shape[1:] != (4, 4):
        raise ValueError(
            f"Input P_batch has incorrect shape: {P_batch.shape}. Expected (B, 4, 4) where B is batch size."
        )

    B = P_batch.shape[0]    

    device = P_batch.device
    dtype = P_batch.dtype
    # PyTorch's default rtol for functions like isclose/allclose
    default_rtol = 1e-5  # PyTorch's default rtol for float32 in isclose

    # Start with all poses presumed valid
    validity_mask = torch.ones(B, dtype=torch.bool, device=device)

    # Check 1: NaNs or Infs
    # For each item in batch, check if it has any NaNs or Infs.
    # Reshape to (B, 16) then check any along the second dimension.
    has_nan_or_inf = torch.logical_or(torch.isnan(P_batch), torch.isinf(P_batch)).reshape(B, -1).any(dim=1)
    validity_mask &= ~has_nan_or_inf  # Valid if NOT (has_nan OR has_inf)
    if has_nan_or_inf.any():
        print(f"{P_batch[has_nan_or_inf][0]} has_nan_or_inf")
        raise ValueError(f"{P_batch[has_nan_or_inf][0]} has_nan_or_inf: {encoding}")
        

    # Check 2: Last row should be [0, 0, 0, 1]
    # expected_last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device)  # Shape (4,)
    # P_batch[:, 3, :] has shape (B, 4)
    # Compare each of the B last rows with expected_last_row
    # last_row_correct = torch.all(
    #     torch.isclose(P_batch[:, 3, :], expected_last_row.unsqueeze(0), atol=atol, rtol=default_rtol),
    #     dim=1  # Check if all 4 elements in the last row are close for each batch item
    # )  # Shape (B,)
    # validity_mask &= last_row_correct
    # if not last_row_correct.all():
    #     raise ValueError(f"{P_batch[~last_row_correct][0]} not last_row_correct")

    # Extract rotation part for all poses in the batch
    R_batch = P_batch[:, :3, :3]  # Shape (B, 3, 3)
    # Check 3a: Orthogonality of Rotation Matrix R (R @ R.T should be close to identity)
    I = torch.eye(3, dtype=dtype, device=device)  # Shape (3, 3)
    # R_batch @ R_batch.transpose(1, 2) computes (R @ R.T) for each pose. Shape (B, 3, 3)
    # Using bmm for explicit batch matrix multiplication is also an option:
    # RR_T = torch.bmm(R_batch, R_batch.transpose(1, 2))
    RR_T = torch.matmul(R_batch, R_batch.transpose(1, 2)) # matmul also handles batches correctly
    # Check if each (3,3) matrix in RR_T is close to I (broadcasted)
    rotation_orthogonal = torch.all(
        torch.isclose(RR_T, I.unsqueeze(0), atol=atol, rtol=default_rtol),
        dim=(1, 2)  # Check if all 3x3 elements are close for each batch item
    )  # Shape (B,)
    validity_mask &= rotation_orthogonal
    if not rotation_orthogonal.all():
        raise ValueError(f"{R_batch[~rotation_orthogonal][0]} -> {RR_T[~rotation_orthogonal][0]} not rotation_orthogonal")

    # Check 3b: Determinant of Rotation Matrix R (det(R) should be close to +1)
    det_R_batch = torch.linalg.det(R_batch)  # Shape (B,)
    # Compare with a tensor of ones, or a scalar 1.0 which will broadcast.
    determinant_correct = torch.isclose(
        det_R_batch, torch.tensor(1.0, dtype=dtype, device=device), atol=atol, rtol=default_rtol
    )  # Shape (B,)
    validity_mask &= determinant_correct
    if not determinant_correct.all():
        raise ValueError(f"{det_R_batch[~determinant_correct][0]} not determinant_correct")

    return validity_mask
    
def get_center_of_views(
    poses_views_in_world1_c2w: torch.Tensor # Input: (B, S, 4, 4) C2W poses (views in world1)
) -> torch.Tensor:                          # Output: (B, 4, 4) pose of central world2 in world1
    """
    Calculates the pose of a 'central' world2 for each sample in a batch,
    relative to world1. The central world2's origin is the centroid of the
    input view positions (camera origins), and its orientation is the average
    orientation of the views.

    Args:
        poses_views_in_world1_c2w: A PyTorch tensor of shape (B, S, 4, 4) representing
                                   Camera-to-World (C2W) poses. These are the poses of 'S'
                                   views expressed in 'world1' coordinates, for 'B' batch
                                   samples. It's assumed these are valid SE(3) matrices
                                   (i.e., the rotation part is orthogonal with determinant +1,
                                   and the last row is [0,0,0,1]).

    Returns:
        pose_world2_to_world1: A PyTorch tensor of shape (B, 4, 4).
                               This matrix represents the pose of the calculated 'central world2'
                               expressed in 'world1' coordinates for each batch sample.
                               By convention, this pose matrix transforms points from
                               'world2' coordinates to 'world1' coordinates.
                               (i.e., X_world1 = pose_world2_to_world1 @ X_world2)
    """
    B = poses_views_in_world1_c2w.shape[0]
    device = poses_views_in_world1_c2w.device
    dtype = poses_views_in_world1_c2w.dtype

    # 1. Extract Rotations and Translations from the input C2W poses
    # R_views_in_world1: Orientations of view_S's axes in world1 coordinates
    R_views_in_world1 = poses_views_in_world1_c2w[:, :, :3, :3]  # Shape: (B, S, 3, 3)
    # t_view_origins_in_world1: Positions of view_S's origins in world1 coordinates
    t_view_origins_in_world1 = poses_views_in_world1_c2w[:, :, :3, 3]   # Shape: (B, S, 3)
    
    # 2. Calculate Average Camera Position (Centroid) for World2's origin
    # This defines the translational component of world2's pose in world1.
    t_origin_world2_in_world1 = torch.mean(t_view_origins_in_world1, dim=1)  # Shape: (B, 3)
    # t_origin_world2_in_world1 = check_and_fix_inf_nan(t_origin_world2_in_world1, loss_name='t_origin_world2_in_world1')

    # 3. Calculate Average Rotation for World2's orientation in world1
    # Sum of rotation matrices for each batch sample
    M_rot_sum = torch.sum(R_views_in_world1, dim=1)  # Shape: (B, 3, 3)

    # Batch SVD: M_rot_sum = U @ diag(S_singular_values) @ Vh
    # U: (B, 3, 3), S_singular_values: (B, 3), Vh: (B, 3, 3) (Vh is V.transpose for real matrices)
    U, S_singular_values, Vh = torch.linalg.svd(M_rot_sum)

    # Determinant correction to ensure the average rotation is in SO(3)
    # The target average rotation is R_avg = U @ diag(1, 1, det(U@Vh)) @ Vh.
    # det(U@Vh) is equivalent to det(U) * det(Vh).
    det_U = torch.linalg.det(U)    # Shape: (B)
    det_Vh = torch.linalg.det(Vh)  # Shape: (B)
    # det_product_uvh is det(U @ Vh)
    det_product_uvh = det_U * det_Vh # Shape: (B)

    # Create a corrected U (U_corrected) by flipping the sign of the last column
    # of U for batch items where det_product_uvh is negative (det(U@Vh) = -1).
    U_corrected = U.clone() # Use .clone() for safety before in-place modification
    correction_needed_mask = det_product_uvh < 0.0
    U_corrected[correction_needed_mask, :, 2] *= -1.0
    
    # Calculate the corrected average rotation for world2's orientation in world1
    R_orientation_world2_in_world1 = U_corrected @ Vh  # Shape: (B, 3, 3)
    # R_orientation_world2_in_world1 = check_and_fix_inf_nan(R_orientation_world2_in_world1, loss_name='R_orientation_world2_in_world1')
    
    # 4. Construct the Pose of World2 in World1
    # Initialize a batch of 4x4 matrices (zeros, then set diagonal)
    pose_world2_to_world1 = torch.zeros((B, 4, 4), device=device, dtype=dtype)
    
    # Assign the calculated rotation and translation parts
    pose_world2_to_world1[:, :3, :3] = R_orientation_world2_in_world1
    pose_world2_to_world1[:, :3, 3] = t_origin_world2_in_world1
    # Set the homogenous coordinate
    pose_world2_to_world1[:, 3, 3] = 1.0
    
    return pose_world2_to_world1

def get_pose_and_index_with_median_rotation_angle(
    poses_views_in_world1_c2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Identifies the view whose rotation component has a median angle of rotation
    (scalar "distance" from the identity rotation), for each sample in the batch.

    The selection is based on the index returned by torch.median when applied
    to the rotation angles. If the number of views S is even, torch.median typically
    returns the value and index of the lower of the two middle elements after sorting.

    Args:
        poses_views_in_world1_c2w: A PyTorch tensor of shape (B, S, 4, 4) representing
                                   Camera-to-World (C2W) poses. These are the poses of 'S'
                                   views expressed in 'world1' coordinates, for 'B' batch
                                   samples. Assumed to be valid SE(3) matrices.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - selected_poses: Tensor of shape (B, 4, 4). For each batch item,
                              this is the full pose of the view whose rotation matrix's
                              angle (representing its deviation from identity) is the
                              median among all such angles in that batch item's set of views.
            - selected_indices: Tensor of shape (B), dtype=torch.long. For each batch
                                item, this is the index (0 to S-1) of the selected
                                pose within its original S-dimension.
    """
    B, S, _, _ = poses_views_in_world1_c2w.shape
    device = poses_views_in_world1_c2w.device
    dtype = poses_views_in_world1_c2w.dtype # Input dtype for new tensors

    # 1. Extract all rotation matrices from the C2W poses
    Rs_views_in_world1 = poses_views_in_world1_c2w[:, :, :3, :3]   # Shape: (B, S, 3, 3)

    # 2. Calculate the angle of rotation for each rotation matrix.
    # This angle measures the rotation's "distance" from the Identity rotation.
    # Formula: angle = acos((trace(R) - 1) / 2)
    
    # Calculate trace for each rotation matrix in the batch
    # R.diagonal(offset=0, dim1=-2, dim2=-1) extracts diagonals for matrices in last 2 dims
    # .sum(dim=-1) sums these diagonal elements to get the trace.
    trace_of_Rs = Rs_views_in_world1.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1) # Shape: (B, S)
    
    # Clamp the argument of acos to the valid range [-1, 1] for numerical stability.
    # Trace(R) for R in SO(3) is in [-1, 3], so (Trace(R)-1)/2 is in [-1, 1].
    # Epsilon helps guard against floating point inaccuracies pushing values slightly out of bounds.
    epsilon_acos = 1e-7 if dtype == torch.float32 else 1e-15
    cos_theta_vals = torch.clamp(
        (trace_of_Rs - 1.0) / 2.0, 
        min=-1.0 + epsilon_acos, 
        max=1.0 - epsilon_acos
    ) # Shape: (B, S)
    
    angles_of_rotation = torch.acos(cos_theta_vals) # Shape: (B, S)

    # 3. Find the median of these rotation angles for each batch sample.
    # torch.median along dim=1 (the S dimension) returns two tensors:
    #   - median_angle_vals: The actual median angle for each batch item. (Shape: B)
    #   - median_angle_indices: The indices (0 to S-1) into the S dimension
    #                           that correspond to these median angles. (Shape: B, dtype=torch.long)
    _ , median_angle_indices = torch.median(angles_of_rotation, dim=1)
    # We only need the indices to select the corresponding pose.

    # 4. Gather the full poses corresponding to these selected indices
    # Create batch indices [0, 1, ..., B-1] for advanced indexing
    batch_indices = torch.arange(B, device=device)
    selected_poses = poses_views_in_world1_c2w[batch_indices, median_angle_indices] # Shape: (B, 4, 4)
    
    return selected_poses, median_angle_indices

def get_pose_and_index_with_median_distance_to_origin(
    poses_views_in_world1_c2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Identifies the view whose camera position (translation component) has a
    median L2 distance to the origin of world1, for each sample in the batch.

    The selection is based on the index returned by torch.median when applied
    to the distances. If the number of views S is even, torch.median typically
    returns the value and index of the lower of the two middle elements after sorting.

    Args:
        poses_views_in_world1_c2w: A PyTorch tensor of shape (B, S, 4, 4) representing
                                   Camera-to-World (C2W) poses. These are the poses of 'S'
                                   views expressed in 'world1' coordinates, for 'B' batch
                                   samples. Assumed to be valid SE(3) matrices.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - selected_poses: Tensor of shape (B, 4, 4). For each batch item,
                              this is the full pose of the view whose translation's
                              L2 distance to the world1 origin is the median among
                              all such distances in that batch item's set of views.
            - selected_indices: Tensor of shape (B), dtype=torch.long. For each batch
                                item, this is the index (0 to S-1) of the selected
                                pose within its original S-dimension, corresponding to
                                the view with the median distance to origin.
    """
    B, S, _, _ = poses_views_in_world1_c2w.shape
    device = poses_views_in_world1_c2w.device
    # dtype of the input tensor will be preserved for the output pose
    # selected_indices will be torch.long

    # 1. Extract all translation vectors (positions of view origins in world1)
    ts_view_origins_in_world1 = poses_views_in_world1_c2w[:, :, :3, 3]   # Shape: (B, S, 3)

    # 2. Calculate the L2 norm (Euclidean distance to world1 origin (0,0,0))
    # for each translation vector.
    dists_to_world1_origin = torch.norm(ts_view_origins_in_world1, p=2, dim=2) # Shape: (B, S)

    # 3. Find the median of these distances for each batch sample.
    # torch.median along dim=1 (the S dimension) returns two tensors:
    #   - median_distance_values: The actual median distance for each batch item. (Shape: B)
    #   - median_distance_indices: The indices (0 to S-1) into the S dimension
    #                              that correspond to these median distances. If S is
    #                              even, PyTorch's median returns the index of the
    #                              lower of the two middle elements. (Shape: B, dtype=torch.long)
    # _ , dist_indices = torch.median(dists_to_world1_origin, dim=1)
    dist_indices = torch.argmin(dists_to_world1_origin, dim=1)
    # We only need the indices to select the corresponding pose.

    # 4. Gather the full poses corresponding to these selected indices
    # Create batch indices [0, 1, ..., B-1] for advanced indexing
    batch_indices = torch.arange(B, device=device)
    selected_poses = poses_views_in_world1_c2w[batch_indices, dist_indices] # Shape: (B, 4, 4)
    
    return selected_poses, dist_indices

def get_local_pts3d_from_depth(depth, fuv, fuv_scaler=1.0):
    '''
    depth (B, H, W)
    fuv (B, 2) 
    '''
    # Get the batch size, height, and width from the input tensors
    B, H, W = depth.shape
    device = depth.device
    # print(B, H, W, fuv.shape)

    # Create a meshgrid of pixel coordinates (u, v)
    # Note that torch.meshgrid expects indexing in (x, y) which corresponds to (H, W)
    v_coords, u_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    # print(u_coords.shape, v_coords.shape)

    # Reshape the coordinates to be broadcastable with the depth map
    # (H, W) -> (1, H, W)
    u_coords = u_coords.unsqueeze(0).float()
    v_coords = v_coords.unsqueeze(0).float()
    # print(u_coords.shape, v_coords.shape)

    # Extract the focal lengths (f_u, f_v) for each item in the batch
    # and reshape for broadcasting: (B, 2) -> (B, 1, 1)
    fuv = fuv * fuv_scaler
    f_u = fuv[:, 0].view(B, 1, 1) 
    f_v = fuv[:, 1].view(B, 1, 1)

    # Assume the principal point (c_u, c_v) is the center of the image
    c_u = (W - 1) / 2.0
    c_v = (H - 1) / 2.0

    # The Z coordinate in the camera view is the depth value
    # Z has shape (B, H, W)
    Z = depth

    # Calculate the X and Y coordinates using the pinhole camera model equations
    X = (u_coords - c_u) * Z / f_u
    Y = (v_coords - c_v) * Z / f_v

    # Stack the X, Y, and Z coordinates to form the 3D point cloud
    # The result will be a tensor of shape (B, H, W, 3)
    pts_3d_camera = torch.stack([X, Y, Z], dim=-1)
    
    return pts_3d_camera

def center_c2w_poses_batch(c2w_poses: torch.Tensor, return_poses: bool=True) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
    """
    Computes new world coordinate systems to center a batch of c2w pose sets.

    This function operates on a batch of scenes. For each scene, it finds the
    average translation and rotation of the input camera poses and calculates a
    transformation to move the origin of that scene's world coordinate system
    to its average pose.

    Args:
        c2w_poses (torch.Tensor): A tensor of shape (B, N, 4, 4) containing B scenes,
                                  each with N camera-to-world transformation matrices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        - mean_pose_in_old_world (torch.Tensor): (B, 4, 4) The transformations from the new
                                       centered worlds to the original worlds.
        - old_world_to_mean_pose (torch.Tensor): (B, 4, 4) The transformations from the original
                                       worlds to the new centered worlds.
        - new_c2w_poses (torch.Tensor): (B, N, 4, 4) The new, centered c2w poses for each scene.
    """
    # Ensure the input tensor is a float tensor
    c2w_poses = c2w_poses.float()
    B, N, _, _ = c2w_poses.shape

    # --- Step 1: Decompose the Poses ---
    rotations = c2w_poses[:, :, :3, :3]      # Shape: (B, N, 3, 3)
    translations = c2w_poses[:, :, :3, 3]    # Shape: (B, N, 3)

    # --- Step 2: Calculate the Mean Translation for each scene ---
    # We average over the N poses (dim=1)
    mean_translation = torch.mean(translations, dim=1) # Shape: (B, 3)

    # --- Step 3: Calculate the Mean Rotation for each scene ---
    # Sum the rotation matrices over the N poses (dim=1)
    summed_rotations = torch.sum(rotations, dim=1) # Shape: (B, 3, 3)

    # Perform SVD on the batch of summed rotation matrices
    try:
        U, _, Vh = torch.linalg.svd(summed_rotations) # U, Vh shape: (B, 3, 3)
        # Tentatively calculate the mean rotation for the whole batch
        mean_rotation = U @ Vh
    except torch.linalg.LinAlgError:
        print("SVD computation failed for one or more items in the batch.")
        mean_rotation = torch.eye(3, dtype=c2w_poses.dtype, device=c2w_poses.device).unsqueeze(0).repeat(B, 1, 1)
        raise

    # Check for reflections (det < 0) in the batch
    det = torch.linalg.det(mean_rotation.to(torch.float32)) # Shape: (B,)
    reflection_mask = det < 0
    
    # Correct only the matrices that are reflections
    if torch.any(reflection_mask):
        # Create a clone of Vh to modify safely
        Vh_prime = Vh.clone()
        
        # For each matrix needing correction, flip the sign of the last row of Vh
        Vh_prime[reflection_mask, -1, :] *= -1
        
        # Recompute the mean rotation ONLY for the items that were reflections
        corrected_rotations = U[reflection_mask] @ Vh_prime[reflection_mask]
        mean_rotation[reflection_mask] = corrected_rotations

    # --- Step 4: Construct the New World Transformations for each scene ---
    # This matrix represents the pose of the "mean camera" in the original world frame.
    # It transforms points from the new centered frame to the old original frame.
    mean_pose_in_old_world = torch.eye(4, dtype=c2w_poses.dtype, device=c2w_poses.device).unsqueeze(0).repeat(B, 1, 1)
    mean_pose_in_old_world[:, :3, :3] = mean_rotation
    mean_pose_in_old_world[:, :3, 3] = mean_translation

    # Transformation from old worlds to new worlds (batched analytical inverse)
    old_world_to_mean_pose = torch.eye(4, dtype=c2w_poses.dtype, device=c2w_poses.device).unsqueeze(0).repeat(B, 1, 1)
    R_inv = mean_rotation.transpose(-2, -1) # Batched transpose
    t_inv = -R_inv @ mean_translation.unsqueeze(-1) # Batched matrix-vector product
    old_world_to_mean_pose[:, :3, :3] = R_inv
    old_world_to_mean_pose[:, :3, 3] = t_inv.squeeze(-1)
    
    if return_poses:
        # ...
        new_c2w_poses = torch.matmul(old_world_to_mean_pose[:, None], c2w_poses)
        return mean_pose_in_old_world, old_world_to_mean_pose, new_c2w_poses
    else:
        return mean_pose_in_old_world, old_world_to_mean_pose, None

def get_pred_world_to_gt_world_transforms(
    gt_c2w_poses: torch.Tensor,
    pred_c2w_poses: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the transformations from the predicted world to the GT world.

    This function assumes that for each batch and item index (b, n),
    gt_c2w_poses[b, n] and pred_c2w_poses[b, n] represent poses from the
    *same* camera coordinate system, but into two different world
    coordinate systems (the "ground truth" world and the "predicted" world).

    - gt_c2w_poses (T_c_gt): Transforms points from camera to GT world.
    - pred_c2w_poses (T_c_pred): Transforms points from camera to Predicted world.

    We want to find the transformation (T_pred_gt) that maps points from
    the Predicted world to the GT world:
        P_gt = T_pred_gt @ P_pred

    The relationship is:
        T_c_gt @ P_cam = T_pred_gt @ (T_c_pred @ P_cam)
        T_c_gt = T_pred_gt @ T_c_pred

    Solving for T_pred_gt:
        T_pred_gt = T_c_gt @ torch.linalg.inv(T_c_pred)

    This function implements this by first analytically computing the
    inverse of pred_c2w_poses (which is the w2c pose T_pred_c).

    Args:
        gt_c2w_poses (torch.Tensor): (B, N, 4, 4) Poses from camera to GT world.
        pred_c2w_poses (torch.Tensor): (B, N, 4, 4) Poses from camera to Predicted world.

    Returns:
        torch.Tensor: (B, N, 4, 4) The transformation matrices from the
                      Predicted world to the GT world for each pair.
    """
    
    # --- 1. Analytically compute the inverse of pred_c2w_poses (T_c_pred) ---
    # The inverse is the world-to-camera pose (T_pred_c)
    
    # Decompose the predicted poses
    R_pred = pred_c2w_poses[..., :3, :3]  # (B, N, 3, 3)
    t_pred = pred_c2w_poses[..., :3, 3:4] # (B, N, 3, 1)

    # Calculate inverse rotation (transpose)
    R_pred_inv = R_pred.transpose(-2, -1) # (B, N, 3, 3)
    
    # Calculate inverse translation (-R^T * t)
    t_pred_inv = -R_pred_inv @ t_pred     # (B, N, 3, 1)

    # Build the (B, N, 4, 4) inverse matrices
    # Start with identity matrices
    batch_shape = R_pred_inv.shape[:-2]

    # 2. Allocate memory for the final tensor ONCE, without initializing it
    inv_pred_c2w_poses = torch.empty(*batch_shape, 4, 4, dtype=pred_c2w_poses.dtype, device=pred_c2w_poses.device)

    # 3. Fill the components directly into the allocated memory
    inv_pred_c2w_poses[..., :3, :3] = R_pred_inv
    inv_pred_c2w_poses[..., :3, 3:4] = t_pred_inv
    inv_pred_c2w_poses[..., 3, :3] = 0.0
    inv_pred_c2w_poses[..., 3, 3] = 1.0

    # --- 2. Compute the final transformation T_pred_gt = T_c_gt @ T_pred_c ---
    
    # (B, N, 4, 4) @ (B, N, 4, 4) -> (B, N, 4, 4)
    # This performs a batched matrix multiplication for each (b, n) pair.
    pred_to_gt_transform = gt_c2w_poses @ inv_pred_c2w_poses

    return pred_to_gt_transform    
   
def pose_encoding_to_camera(
    pose_encoding,
    pose_encoding_type="absT_quaR",
):
    """
    Args:
        pose_encoding: A tensor of shape `BxC`, containing a batch of
                        `B` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
    """

    if pose_encoding_type == "absT_quaR":

        abs_T = pose_encoding[:, :3]
        quaternion_R = pose_encoding[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    c2w_mats = torch.eye(4, 4).to(R.dtype).to(R.device)
    c2w_mats = c2w_mats[None].repeat(len(R), 1, 1)
    c2w_mats[:, :3, :3] = R
    c2w_mats[:, :3, 3] = abs_T

    return c2w_mats


def quaternion_conjugate(q):
    """Compute the conjugate of quaternion q (w, x, y, z)."""

    q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return q_conj


def quaternion_multiply(q1, q2):
    """Multiply two quaternions q1 and q2."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def rotate_vector(q, v):
    """Rotate vector v by quaternion q."""
    q_vec = q[..., 1:]
    q_w = q[..., :1]

    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    v_rot = v + q_w * t + torch.cross(q_vec, t, dim=-1)
    return v_rot


def relative_pose_absT_quatR(t1, q1, t2, q2):
    """Compute the relative translation and quaternion between two poses."""

    q1_inv = quaternion_conjugate(q1)

    q_rel = quaternion_multiply(q1_inv, q2)

    delta_t = t2 - t1
    t_rel = rotate_vector(q1_inv, delta_t)
    return t_rel, q_rel

def _combine_R_t_to_pose(R_batch: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
    """
    Combines a batch of rotation matrices and translation vectors into a batch
    of 4x4 homogeneous transformation matrices (poses).

    Args:
        R_batch (torch.Tensor): Batch of rotation matrices, shape (B, 3, 3).
        t_batch (torch.Tensor): Batch of translation vectors, shape (B, 1, 3) or (B, 3).

    Returns:
        torch.Tensor: Batch of 4x4 poses, shape (B, 4, 4).
    """
    # Ensure t_batch is shape (B, 3) for easy assignment
    if t_batch.ndim == 3:
        t_batch = t_batch.squeeze(1)

    B = R_batch.shape[0]
    device = R_batch.device
    dtype = R_batch.dtype

    # Initialize a batch of 4x4 identity matrices
    poses = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)

    # Place the rotation and translation into the matrix
    poses[:, :3, :3] = R_batch
    poses[:, :3, 3] = t_batch

    return poses

def transform_gt_poses_to_pr_world(
    R_batch: torch.Tensor, t_batch: torch.Tensor, gt_pose_list: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Transforms a list of ground truth camera poses into the predicted world coordinate system.

    Args:
        R_batch (torch.Tensor): Batch of rotation matrices from the registration,
                                shape (B, 3, 3). This is the rotation part of the
                                transformation from the GT world to the PR world.
        t_batch (torch.Tensor): Batch of translation vectors from the registration,
                                shape (B, 1, 3) or (B, 3). This is the translation part.
        gt_pose_list (List[torch.Tensor]): A list of ground truth camera poses. Each element
                                           is a tensor of shape (B, 4, 4) representing the
                                           poses for a specific view across the batch.

    Returns:
        List[torch.Tensor]: A list of the transformed ground truth poses, where each
                            element is now expressed in the predicted world coordinate system.
    """
    T_gt_to_pr = _combine_R_t_to_pose(R_batch, t_batch)

    new_gt_pose_list = [torch.matmul(T_gt_to_pr, gt_poses) for gt_poses in gt_pose_list]


    return new_gt_pose_list
