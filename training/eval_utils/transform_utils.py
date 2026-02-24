import torch

def global_points_from_cam(extrinsics: torch.Tensor, pts3d_cam: torch.Tensor) -> torch.Tensor:
    """
    Transforms 3D points from camera coordinates to world coordinates.
    
    Args:
        extrinsics (torch.Tensor): Camera-to-world transformation matrices. 
                                   Shape: (B, S, 3, 4) or (B, S, 4, 4).
        pts3d_cam (torch.Tensor): 3D points in camera coordinates. 
                                  Shape: (B, S, H, W, 3).
                                  
    Returns:
        torch.Tensor: 3D points in world coordinates. Shape: (B, S, H, W, 3).
    """
    R = extrinsics[..., :3, :3]  # Shape: (B, S, 3, 3)
    T = extrinsics[..., :3, 3]   # Shape: (B, S, 3)
    B, S, H, W, _ = pts3d_cam.shape
    pts3d_cam_flat = pts3d_cam.view(B, S, H * W, 3)

    # The transformation for row vectors is p_world = p_cam @ R.T + T
    points_world_flat = torch.matmul(pts3d_cam_flat, R.transpose(-1, -2)) + T[..., None, :]
    points_world = points_world_flat.view(B, S, H, W, 3)

    return points_world


def cam_points_from_depth(intrinsics: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    """
    Generates 3D points in camera coordinates from a depth map with high efficiency.

    This optimized version performs the unprojection by:
    1. Pre-calculating the inverse of the intrinsic matrix.
    2. Creating a static grid of homogeneous pixel coordinates.
    3. Applying the transformation in a single, fused `einsum` operation.
    This avoids creating multiple large intermediate tensors for X and Y coordinates.

    Args:
        intrinsics (torch.Tensor): The camera intrinsic matrix of shape (B, S, 3, 3).
        depths (torch.Tensor): A depth map of shape (B, S, H, W, 1).

    Returns:
        torch.Tensor: A tensor of 3D points in the camera's coordinate system,
                      with shape (B, S, H, W, 3).
    """
    B, S, H, W, _ = depths.shape
    device = depths.device
    dtype = depths.dtype
    
    # 1. Pre-calculate inverse intrinsics (B, S, 3, 3)
    inv_intrinsics = torch.inverse(intrinsics)
    
    # 2. Create static grid of homogeneous pixel coordinates (H, W, 3)
    # This is memory efficient as we only create it once
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    
    # Stack into homogeneous coordinates [x, y, 1] with shape (H, W, 3)
    pixel_coords = torch.stack([
        x_coords,
        y_coords,
        torch.ones_like(x_coords)
    ], dim=-1)
    
    # 3. Apply inverse intrinsics to get normalized camera coordinates
    # einsum: (B, S, 3, 3) × (H, W, 3) -> (B, S, H, W, 3)
    # This fuses the matrix multiplication across spatial dimensions
    normalized_coords = torch.einsum('bsij,hwj->bshwi', inv_intrinsics, pixel_coords)
    
    # 4. Scale by depth to get 3D camera points
    # Broadcasting: (B, S, H, W, 3) * (B, S, H, W, 1) -> (B, S, H, W, 3)
    cam_points = normalized_coords * depths
    
    return cam_points
