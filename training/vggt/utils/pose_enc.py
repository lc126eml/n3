# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Tuple
from .rotation import quat_to_mat, mat_to_quat, safe_quat_to_mat

def intri_to_fov_encoding(
    intrinsics: torch.Tensor,
    image_size_hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Converts camera intrinsics to a 2D field of view (FoV) encoding.

    Args:
        intrinsics (torch.Tensor): Camera intrinsic parameters with shape BxSx3x3.
            Defined in pixels, with format:
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
            where fx, fy are focal lengths and (cx, cy) is the principal point.
        image_size_hw (Tuple[int, int]): Tuple of (height, width) of the image
            in pixels. This is required to compute the field of view.

    Returns:
        torch.Tensor: Encoded field of view parameters with shape BxSx2.
            The 2 dimensions are:
            - [:, :, 0] = fov_h (vertical field of view in radians)
            - [:, :, 1] = fov_w (horizontal field of view in radians)
    """
    if image_size_hw is None:
        raise ValueError("image_size_hw must be provided (e.g., (height, width)) to calculate field of view.")

    # Unpack image dimensions
    H, W = image_size_hw

    # Extract focal lengths from the BxSx3x3 intrinsics matrix
    # fx is at (0, 0), fy is at (1, 1)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]

    # Calculate FoV in radians using the pinhole camera model formula
    fov_h = 2 * torch.atan((H / 2) / fy)
    fov_w = 2 * torch.atan((W / 2) / fx)

    # Combine into a BxSx2 tensor
    # fov_h has shape (B, S), fov_h[..., None] makes it (B, S, 1)
    # torch.cat stacks them along the last dimension
    fov_encoding = torch.cat([fov_h[..., None], fov_w[..., None]], dim=-1).float()

    return fov_encoding

def intri_to_logk_encoding(
    intrinsics: torch.Tensor,
    image_size_hw: Tuple[int, int],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Converts camera intrinsics to a log-normalized K encoding.

    The 4 dimensions are:
    - [:, :, 0] = log(fx / W)
    - [:, :, 1] = log(fy / H)
    - [:, :, 2] = log(cx / W)
    - [:, :, 3] = log(cy / H)
    """
    if image_size_hw is None:
        raise ValueError("image_size_hw must be provided (e.g., (height, width)) to calculate logK.")

    H, W = image_size_hw

    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]

    log_fx_w = torch.log(torch.clamp(fx / W, min=eps))
    log_fy_h = torch.log(torch.clamp(fy / H, min=eps))
    log_cx_w = torch.log(torch.clamp(cx / W, min=eps))
    log_cy_h = torch.log(torch.clamp(cy / H, min=eps))

    return torch.stack([log_fx_w, log_fy_h, log_cx_w, log_cy_h], dim=-1).float()

def extri_to_pose_encoding(
    extrinsics, pose_encoding_type="absT_quaR" 
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    This function transforms camera parameters into a unified pose encoding format,
    which can be used for various downstream tasks like pose prediction or representation.

    Args:
        extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4,
            where B is batch size and S is sequence length.
            In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world transformation.
            The format is [R|t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        pose_encoding_type (str): Type of pose encoding to use. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).

    Returns:
        torch.Tensor: Encoded camera pose parameters with shape BxSx9.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
    """

    # extrinsics: BxSx3x4

    if pose_encoding_type == "absT_quaR":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3

        quat = mat_to_quat(R)
        pose_encoding = torch.cat([T, quat], dim=-1).float()
    else:
        raise NotImplementedError

    return pose_encoding

def extri_intri_to_pose_encoding(
    extrinsics, intrinsics, image_size_hw=None, pose_encoding_type="absT_quaR_FoV"  # e.g., (256, 512)
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    This function transforms camera parameters into a unified pose encoding format,
    which can be used for various downstream tasks like pose prediction or representation.

    Args:
        extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4,
            where B is batch size and S is sequence length.
            In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world transformation.
            The format is [R|t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        intrinsics (torch.Tensor): Camera intrinsic parameters with shape BxSx3x3.
            Defined in pixels, with format:
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
            where fx, fy are focal lengths and (cx, cy) is the principal point
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for computing field of view values. For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding to use. Currently
            supports "absT_quaR_FoV" and "absT_quaR_logK".

    Returns:
        torch.Tensor: Encoded camera pose parameters with shape BxSx9.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
    """

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3

    if pose_encoding_type == "absT_quaR_FoV":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3

        quat = mat_to_quat(R)
        # Note the order of h and w here
        H, W = image_size_hw
        fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    elif pose_encoding_type == "absT_quaR_logK":
        R = extrinsics[:, :, :3, :3]
        T = extrinsics[:, :, :3, 3]

        quat = mat_to_quat(R)
        logk = intri_to_logk_encoding(intrinsics, image_size_hw)
        pose_encoding = torch.cat([T, quat, logk], dim=-1).float()
    else:
        raise NotImplementedError

    return pose_encoding

def make_extrinsics(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Convert R (..., 3, 3) and T (..., 3) to extrinsics (..., 4, 4).
    """
    batch_shape = R.shape[:-2]
    device, dtype = R.device, R.dtype
    out = torch.empty(*batch_shape, 4, 4, device=device, dtype=dtype)
    
    out[..., :3, :3] = R
    out[..., :3, 3]  = T
    out[..., 3, :3]  = 0.0
    out[..., 3, 3]   = 1.0
    
    return out

def pose_encoding_to_extri_intri(
    pose_encoding, image_size_hw=None, pose_encoding_type="absT_quaR_FoV", build_intrinsics=True  # e.g., (256, 512)
):
    """Convert a pose encoding back to camera extrinsics and intrinsics.

    This function performs the inverse operation of extri_intri_to_pose_encoding,
    reconstructing the full camera parameters from the compact encoding.

    Args:
        pose_encoding (torch.Tensor): Encoded camera pose parameters with shape BxSx9,
            where B is batch size and S is sequence length.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for reconstructing intrinsics from field of view values.
            For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding used. Currently
            supports "absT_quaR_FoV" and "absT_quaR_logK".
        build_intrinsics (bool): Whether to reconstruct the intrinsics matrix.
            If False, only extrinsics are returned and intrinsics will be None.

    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4.
              In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world
              transformation. The format is [R|t] where R is a 3x3 rotation matrix and t is
              a 3x1 translation vector.
            - intrinsics (torch.Tensor or None): Camera intrinsic parameters with shape BxSx3x3,
              or None if build_intrinsics is False. Defined in pixels, with format:
              [[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]]
              where fx, fy are focal lengths and (cx, cy) is the principal point,
              assumed to be at the center of the image (W/2, H/2).
    """

    intrinsics = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = safe_quat_to_mat(quat)
        extrinsics = make_extrinsics(R, T)

        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
    elif pose_encoding_type == "absT_quaR_logK":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        log_fx_w = pose_encoding[..., 7]
        log_fy_h = pose_encoding[..., 8]
        log_cx_w = pose_encoding[..., 9]
        log_cy_h = pose_encoding[..., 10]

        R = safe_quat_to_mat(quat)
        extrinsics = make_extrinsics(R, T)

        if build_intrinsics:
            H, W = image_size_hw
            fx = torch.exp(log_fx_w) * W
            fy = torch.exp(log_fy_h) * H
            cx = torch.exp(log_cx_w) * W
            cy = torch.exp(log_cy_h) * H
            intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = cx
            intrinsics[..., 1, 2] = cy
            intrinsics[..., 2, 2] = 1.0
    else:
        raise NotImplementedError
    
    return extrinsics, intrinsics

def is_extrinsics_valid(extrinsics):
    """
    Checks if the extrinsics matrix contains any NaN or infinite values.

    An extrinsics matrix is considered invalid if any of its elements are not finite numbers.
    This can happen due to issues in the pose estimation process, such as numerical instability.

    Args:
        extrinsics (torch.Tensor): Camera extrinsic parameters, typically with shape BxSx3x4.

    Returns:
        bool: True if the extrinsics matrix is valid (contains only finite numbers),
              False otherwise.
    """
    if not isinstance(extrinsics, torch.Tensor):
        raise TypeError("Input extrinsics must be a torch.Tensor.")
        
    return torch.all(torch.isfinite(extrinsics)).item()
