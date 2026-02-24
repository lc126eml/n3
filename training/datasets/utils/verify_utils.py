import os
import os.path as osp
import sys
import glob
import logging
import argparse # Import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
os.environ['OPEN3D_CPU_RENDERING'] = 'false'

open3d = False
if open3d:
    ### OPEN3D CHANGE: Replaced PyTorch3D with Open3D ###
    try:
        import open3d as o3d
        # Set Open3D to use less verbose logging
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)        
    except ImportError:
        print("FATAL ERROR: Open3D is not installed.")
        print("Please install it by following the instructions at http://www.open3d.org/docs/latest/getting_started.html")
        sys.exit(1)        

# --- 1. Define Success Criteria (Thresholds) ---
MAE_THRESHOLD = 0.01          # meters (1 cm)
RMSE_THRESHOLD = 0.02         # meters (2 cm)
INLIER_METRIC_ERROR_THRESH = 0.05 # meters (5 cm)
INLIER_METRIC_PERCENT_THRESH = 99.0 # % of pixels

# --- 2. User-Provided Helper Functions & Data Loading ---

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    valid_mask = depthmap > 0.0
    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(
    depthmap, camera_intrinsics, camera_pose, **kw
):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam  # default
    if camera_pose is not None:

        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        X_world = (
            np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]
        )

    return X_world, valid_mask, X_cam

# --- 3. GPU-Accelerated Projection and Verification ---
### OPEN3D CHANGE: New rendering function using Open3D's OffscreenRenderer ###
def render_depth_from_pointcloud_open3d(points_np, camera_pose, intrinsics, image_size, device):
    """Renders a depth map from a point cloud using Open3D's headless renderer."""
    H, W = image_size
    
    # 1. Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    
    # 2. Set up the headless renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    
    # Use a simple unlit material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    renderer.scene.add_geometry("pcd", pcd, material)
    
    # 3. Set up camera
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    
    # Camera pose is cam_to_world, Open3D wants world_to_cam (extrinsics)
    world_to_cam = np.linalg.inv(camera_pose)
    renderer.setup_camera(o3d_intrinsics, world_to_cam)
    
    # 4. Render the depth image
    # z_in_view_space=True gives depth along the Z-axis, matching the original depth format.
    depth_o3d = renderer.render_to_depth_image(z_in_view_space=True)
    
    # 5. Convert to NumPy array and return
    rendered_depth = np.asarray(depth_o3d)
    
    # Clean up the scene
    renderer.scene.remove_geometry("pcd")
    
    return rendered_depth

def render_depth_from_pointcloud_enhanced(points, camera_pose, intrinsics, image_size, device, epsilon=1e-6):
    H, W = image_size
    
    # Convert inputs to tensors with float64 precision
    camera_pose = torch.tensor(camera_pose, dtype=torch.float64, device=device)
    intrinsics = torch.tensor(intrinsics, dtype=torch.float64, device=device)
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float64, device=device)

    # Compute stable inverse of camera pose
    R = camera_pose[:3, :3].T
    t = -R @ camera_pose[:3, 3]
    world_to_cam = torch.eye(4, dtype=torch.float64, device=device)
    world_to_cam[:3, :3] = R
    world_to_cam[:3, 3] = t

    # Convert points to homogeneous coordinates
    ones = torch.ones((points.shape[0], 1), dtype=torch.float64, device=device)
    points_h = torch.cat([points, ones], dim=1)

    # Transform points into camera coordinate system
    points_cam = (world_to_cam @ points_h.T).T[:, :3]
    x, y, z = points_cam.unbind(-1)

    # Replace near-zero depth values with epsilon to avoid instability
    z_safe = torch.where(z > epsilon, z, torch.full_like(z, epsilon))

    # Project to image plane using intrinsics (assuming no distortion)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = (fx * x / z_safe) + cx
    v = (fy * y / z_safe) + cy

    # Convert to integer pixel indices
    u_pixel = u.long()
    v_pixel = v.long()

    # Valid pixel mask
    valid_mask = (u_pixel >= 0) & (u_pixel < W) & \
                 (v_pixel >= 0) & (v_pixel < H) & \
                 (z > 0)

    u_valid = u_pixel[valid_mask]
    v_valid = v_pixel[valid_mask]
    z_valid = z[valid_mask]

    if u_valid.numel() == 0:
        return torch.zeros((H, W), dtype=torch.float32, device='cpu').numpy()

    # Compute flattened pixel indices
    pixel_indices = v_valid * W + u_valid

    # Mask out-of-bound pixel indices (should not happen, but ensures safety)
    valid_pixel_mask = pixel_indices < (H * W)
    pixel_indices = pixel_indices[valid_pixel_mask]
    z_valid = z_valid[valid_pixel_mask]

    # Initialize depth map with inf for min-depth rendering
    rendered_depth = torch.full((H * W,), float('inf'), dtype=torch.float32, device=device)

    # Use scatter_reduce_ if available (PyTorch ≥1.10)
    # if hasattr(torch.Tensor, 'scatter_reduce_'):
    rendered_depth.scatter_reduce_(0, pixel_indices, z_valid.float(), reduce='amin')
    # else:
    #     # Fallback: sort by ascending depth and use scatter_ (overwrite logic)
    #     sorted_indices = torch.argsort(z_valid)
    #     pixel_indices = pixel_indices[sorted_indices]
    #     z_sorted = z_valid[sorted_indices].float()
    #     rendered_depth.scatter_(0, pixel_indices, z_sorted)

    # Replace inf with 0.0 (no valid depth)
    rendered_depth[torch.isinf(rendered_depth)] = 0.0

    # Reshape to (H, W)
    rendered_depth = rendered_depth.view(H, W)

    return rendered_depth.cpu().numpy()

def render_depth_from_pointcloud(points, camera_pose, intrinsics, image_size, device):
    H, W = image_size
    world_to_cam = torch.inverse(torch.tensor(camera_pose, dtype=torch.float32, device=device))
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32, device=device)
    points_h = torch.cat([points, torch.ones(points.shape[0], 1, device=device)], dim=1)
    points_cam = (world_to_cam @ points_h.T).T[:, :3]
    points_proj = (torch.tensor(intrinsics, dtype=torch.float32, device=device) @ points_cam.T).T
    u = (points_proj[:, 0] / points_proj[:, 2]).long()
    v = (points_proj[:, 1] / points_proj[:, 2]).long()
    d = points_proj[:, 2]
    valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (d > 0)
    u, v, d = u[valid_mask], v[valid_mask], d[valid_mask]
    rendered_depth = torch.zeros((H, W), dtype=torch.float32, device=device)
    if u.numel() > 0:
        depth_sorted_indices = torch.argsort(d, descending=True)
        u_sorted, v_sorted, d_sorted = u[depth_sorted_indices], v[depth_sorted_indices], d[depth_sorted_indices]
        pixel_indices = v_sorted * W + u_sorted
        rendered_depth.view(-1).scatter_(0, pixel_indices, d_sorted)
    return rendered_depth.cpu().numpy()

### OPEN3D CHANGE: New voxel downsampling function using Open3D ###
def voxel_downsample_open3d(points_np, voxel_size):
    """Performs voxel downsampling using Open3D."""
    if points_np.size == 0:
        return points_np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downsampled_pcd.points)

def voxel_downsample_pytorch(points_tensor, voxel_size=0.001):
    """
    Performs voxel downsampling on a point cloud tensor.

    Args:
        points_tensor (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
        voxel_size (float): The size of a voxel cube.

    Returns:
        torch.Tensor: The downsampled point cloud tensor.
    """
    # Check if the tensor is empty
    if points_tensor.numel() == 0:
        return points_tensor

    # Create integer voxel indices
    voxel_indices = torch.floor(points_tensor / voxel_size).long()

    # Create a unique 1D hash for each 3D voxel index
    # This is a standard way to uniquely identify voxels in a sparse grid
    # The hash values are chosen to be large primes to minimize collisions
    # This assumes the coordinate space isn't pathologically large
    hasher = torch.tensor([1, 41, 16777619], device=points_tensor.device, dtype=torch.long)
    voxel_hash = torch.sum(voxel_indices * hasher, dim=1)

    # Get the unique voxel hashes and map original points to the unique indices
    unique_hashes, inverse_indices = torch.unique(voxel_hash, return_inverse=True)

    # Prepare tensors for summing and counting points in each voxel
    num_unique_voxels = unique_hashes.shape[0]
    summed_points = torch.zeros((num_unique_voxels, 3), dtype=points_tensor.dtype, device=points_tensor.device)
    point_counts = torch.zeros(num_unique_voxels, dtype=torch.int64, device=points_tensor.device)

    # Use scatter_add_ to efficiently sum points and count points per voxel
    summed_points.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), points_tensor)
    point_counts.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.int64))
    
    # Calculate the centroid by dividing summed points by the count
    downsampled_points = summed_points / point_counts.unsqueeze(1).float()

    return downsampled_points

if open3d:
    render_f = render_depth_from_pointcloud_open3d
    downsample_f = voxel_downsample_open3d
else:
    render_f = render_depth_from_pointcloud_enhanced
    downsample_f = voxel_downsample_pytorch

def calculate_metrics(rendered_depth, original_depth, valid_mask):
    valid_mask = (original_depth > 0) & (rendered_depth > 0) & valid_mask
    if np.sum(valid_mask) == 0:
        return float('inf'), float('inf'), 0.0
    diff = np.abs(rendered_depth[valid_mask] - original_depth[valid_mask])
    mae = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    inliers = np.sum(diff < INLIER_METRIC_ERROR_THRESH)
    inlier_percent = (inliers / np.sum(valid_mask)) * 100.0 if np.sum(valid_mask) > 0 else 0.0
    return mae, rmse, inlier_percent

def setup_logging(log_file='verification.log'):
    """Configures logging to write to a file and the console."""
    logger = logging.getLogger('VerificationLogger')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger


def verify_subscene(subscene_path, device, logger, load_view_data, ext='_rgb.png', voxel_size=0.0):
    """Runs the full verification protocol on a single subscene directory."""
    rgb_paths = sorted(glob.glob(osp.join(subscene_path, f"*{ext}")))
    logger.info(f"\nVerifying subscene: {subscene_path}, files: {len(rgb_paths)}")
    if len(rgb_paths) < 2:
        logger.warning("Not enough views to verify. Skipping.")
        return True
    all_world_points, all_views_data = [], []
    logger.info("Step 1/3: Loading views and building global point cloud...")
    for rgb_path in tqdm(rgb_paths, file=sys.stdout):
        view_data = load_view_data(rgb_path, logger)
        if view_data is None: continue
        world_points, valid_mask, _ = depthmap_to_absolute_camera_coordinates(
            view_data["depth"], view_data["intrinsics"], view_data["pose"]
        )
        view_data["valid_mask"] = valid_mask
        all_views_data.append(view_data)
        all_world_points.append(world_points[valid_mask])
    if not all_world_points:
        logger.error("No valid data found in subscene.")
        return False
    global_points = np.concatenate(all_world_points, axis=0)

    if voxel_size > 0:
        logger.info(f"Original {len(global_points)} points.")
        # 1. Move raw points to a GPU tensor
        points_tensor_raw = torch.tensor(global_points, dtype=torch.float32, device=device)        
        # 2. Define the voxel size.
        # voxel_size = 0.01         
        # 3. Apply voxel downsampling
        global_points = downsample_f(points_tensor_raw, voxel_size)
    
    # The downsampled points are already a tensor on the correct device
    logger.info(f"Global model constructed with {len(global_points)} points.")
    logger.info("Step 2/3: Projecting global model to each view for verification...")
    is_consistent, failed_views = True, []
    for view_data in tqdm(all_views_data, file=sys.stdout):
        image_size = view_data["depth"].shape
        rendered_depth = render_f(
            global_points, view_data["pose"], view_data["intrinsics"], image_size, device
        )
        original_depth = view_data["depth"]
        mask = view_data["valid_mask"]
        mae, rmse, inlier_percent = calculate_metrics(rendered_depth, original_depth, mask)
        pass_mae, pass_rmse, pass_inliers = mae <= MAE_THRESHOLD, rmse <= RMSE_THRESHOLD, inlier_percent >= INLIER_METRIC_PERCENT_THRESH
        if not (pass_mae and pass_rmse and pass_inliers):
            is_consistent = False
            failed_views.append({
                "path": osp.basename(view_data["path"]), "mae": mae, "rmse": rmse, "inliers": inlier_percent,
                "pass_mae": pass_mae, "pass_rmse": pass_rmse, "pass_inliers": pass_inliers
            })
        else:
            print(osp.basename(view_data["path"]), {"mae": mae, "rmse": rmse, "inliers": inlier_percent})
    logger.info("Step 3/3: Final Verdict.")
    if is_consistent:
        logger.info(f"\n--- VERDICT: PASS ---\nSubscene '{osp.basename(subscene_path)}' is consistent and correct.")
    else:
        log_message = [f"\n--- VERDICT: FAIL ---\nSubscene '{osp.basename(subscene_path)}' is NOT consistent."]
        log_message.append("The following views failed to meet the thresholds:")
        for fail in failed_views:
            log_message.append(f"  - View: {fail['path']}")
            log_message.append(f"    MAE: {fail['mae']:.4f}m (Threshold: {MAE_THRESHOLD}, Pass: {fail['pass_mae']})")
            log_message.append(f"    RMSE: {fail['rmse']:.4f}m (Threshold: {RMSE_THRESHOLD}, Pass: {fail['pass_rmse']})")
            log_message.append(f"    Inliers (< {INLIER_METRIC_ERROR_THRESH}m): {fail['inliers']:.2f}% (Threshold: {INLIER_METRIC_PERCENT_THRESH}%, Pass: {fail['pass_inliers']})")
        logger.warning('\n'.join(log_message))
    return is_consistent

# NEW SEPARATE VALIDATION FUNCTION
def verify_depth_and_world_positions(subscene_path, device, logger, all_views_data):
    """
    Verifies the consistency between depth_meters.hdf5 and position.hdf5
    for each view, using the provided camera pose.
    """
    logger.info(f"\nVerifying subscene (Depth vs. World Position Consistency): {subscene_path}")
    is_consistent_wp = True
    failed_views_wp = []

    if not all_views_data:
        logger.warning("No view data available for Depth vs. World Position Consistency check.")
        return True, "SKIPPED_WP" # Treat as passed if no data to check

    for view_data in tqdm(all_views_data, file=sys.stdout):
        if "world_positions" not in view_data or view_data["world_positions"] is None:
            logger.warning(f"Skipping Depth vs. World Position check for {osp.basename(view_data['path'])}: No 'world_positions' found.")
            failed_views_wp.append({
                "path": osp.basename(view_data["path"]),
                "reason": "missing_world_positions",
                "type": "depth_vs_world_pos"
            })
            continue

        original_depth = view_data["depth"] # (H, W)
        world_points_frame = view_data["world_positions"] # (H, W, 3) in meters
        intrinsics = view_data["intrinsics"]
        pose = view_data["pose"] # cam2world
        H, W = original_depth.shape

        # Filter out (0,0,0) world points which often indicate invalid data
        valid_world_points_mask = (world_points_frame[:, :, 0] != 0) | \
                                  (world_points_frame[:, :, 1] != 0) | \
                                  (world_points_frame[:, :, 2] != 0)

        # Flatten and convert to tensor for projection
        valid_world_points_flat = world_points_frame[valid_world_points_mask].reshape(-1, 3)

        if valid_world_points_flat.size == 0:
            logger.warning(f"No valid world points for {osp.basename(view_data['path'])} for direct verification.")
            failed_views_wp.append({
                "path": osp.basename(view_data["path"]),
                "reason": "no_valid_world_points",
                "type": "depth_vs_world_pos"
            })
            continue

        # Convert world points to camera coordinates
        # world_to_cam is inverse of cam2world (pose)
        world_to_cam = torch.tensor(np.linalg.inv(pose), dtype=torch.float64, device=device)

        # Convert valid_world_points to homogeneous coordinates and transform
        points_h_tensor = torch.cat([
            torch.tensor(valid_world_points_flat, dtype=torch.float64, device=device),
            torch.ones(valid_world_points_flat.shape[0], 1, dtype=torch.float64, device=device)
        ], dim=1)
        points_cam_tensor = (world_to_cam @ points_h_tensor.T).T[:, :3] # X_cam, Y_cam, Z_cam

        # The Z-component of points_cam_tensor is the planar depth
        projected_z_depth_flat = points_cam_tensor[:, 2].cpu().numpy()

        # Reconstruct a full depth map from the projected Z-depth values
        # This requires mapping the valid_world_points_flat back to their original (H,W) pixel locations.
        rendered_depth_from_world_pos = np.zeros_like(original_depth, dtype=np.float32)
        rows_valid, cols_valid = np.where(valid_world_points_mask)

        # Populate the rendered_depth_from_world_pos map
        # Ensure that only points with positive depth in camera space are considered
        positive_depth_mask_in_cam = projected_z_depth_flat > 0

        # Filter rows/cols and projected_z_depth based on positive_depth_mask_in_cam
        rows_valid_filtered = rows_valid[positive_depth_mask_in_cam]
        cols_valid_filtered = cols_valid[positive_depth_mask_in_cam]
        projected_z_depth_filtered = projected_z_depth_flat[positive_depth_mask_in_cam]

        # Use an index array to efficiently scatter values
        linear_indices = rows_valid_filtered * W + cols_valid_filtered
        
        # Initialize a temporary flat array with inf to find minimum depth for overlapping points
        temp_flat_depth = np.full(H * W, np.inf, dtype=np.float32)

        # Scatter the depths. If multiple points map to the same pixel, min() keeps the closest one.
        # This can be vectorized or done with a loop if vectorization is complex.
        # For simple assignment, direct indexing is fine. For min, a loop or more advanced numpy ops.
        # A simple loop for clarity:
        for idx_val, depth_val in zip(linear_indices, projected_z_depth_filtered):
            temp_flat_depth[idx_val] = min(temp_flat_depth[idx_val], depth_val)

        rendered_depth_from_world_pos = temp_flat_depth.reshape(H, W)
        rendered_depth_from_world_pos[np.isinf(rendered_depth_from_world_pos)] = 0.0 # No depth where inf

        # Combine masks for metric calculation
        # The original_depth's valid mask is already considered when loading it.
        # Here, we need to ensure both rendered and original have valid, positive depths.
        comparison_mask = (original_depth > 0) & (rendered_depth_from_world_pos > 0)

        mae_wp, rmse_wp, inlier_percent_wp = calculate_metrics(
            rendered_depth_from_world_pos, original_depth, comparison_mask
        )

        pass_mae_wp, pass_rmse_wp, pass_inliers_wp = \
            mae_wp <= MAE_THRESHOLD, rmse_wp <= RMSE_THRESHOLD, inlier_percent_wp >= INLIER_METRIC_PERCENT_THRESH

        if not (pass_mae_wp and pass_rmse_wp and pass_inliers_wp):
            is_consistent_wp = False
            failed_views_wp.append({
                "path": osp.basename(view_data["path"]),
                "mae": mae_wp, "rmse": rmse_wp, "inliers": inlier_percent_wp,
                "pass_mae": pass_mae_wp, "pass_rmse": pass_rmse_wp, "pass_inliers": pass_inliers_wp,
                "type": "depth_vs_world_pos"
            })
        else: 
            print(f"Depth vs. World Pos. {osp.basename(view_data['path'])} pass",
                  {"mae": mae_wp, "rmse": rmse_wp, "inliers": inlier_percent_wp})

    return is_consistent_wp, failed_views_wp
