#%%
import hydra
import sys
import os
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, "/lc/code/3D/vggt-training")
from training.datasets.base.standalone_multiview_datamodule import StandaloneMultiViewDataModule

def python_eval_resolver(code: str):
    """A resolver for OmegaConf to evaluate a python expression."""
    return eval(code)

if not OmegaConf.has_resolver("python_eval"):
    OmegaConf.register_new_resolver("python_eval", python_eval_resolver)

#%%
# Initialize Hydra and load the training configuration.
# This looks for the config file at /lc/code/3D/vggt-training/training/config/data/standalone_multiview_train.yaml
with hydra.initialize(version_base="1.3", config_path="../../configs", job_name="test_data"):
    # The config name is relative to the config_path.
    # e.g., 'data/standalone_multiview_train' will load 'data/standalone_multiview_train.yaml'
    cfg = hydra.compose(config_name="default.yaml")

print("--- Configuration Loaded ---")
print(OmegaConf.to_yaml(cfg))
#%%
print((cfg.data.data_module.train_config.datasets))
#%%
print((cfg.data.data_module.train_config.datasets[0]))


#%%
# Instantiate the DataModule using the loaded configuration
print("\n--- Instantiating DataModule ---")
datamodule = hydra.utils.instantiate(cfg.data.data_module)
print(f"DataModule of type '{type(datamodule).__name__}' created.")

#%%
# Get the training dataloader
print("\n--- Getting Train Dataloader ---")
train_loader = datamodule.train_dataloader()
print("Train dataloader created.")

#%%
# Get a single batch from the dataloader
print("\n--- Fetching a Batch ---")
batch = next(iter(train_loader))
print("A batch has been fetched from the train_loader.")

#%%
# Print the shape of each tensor in the batch
print("\n--- Inspecting Batch Shapes ---")
for key, value in batch.items():
    if hasattr(value, 'shape'):
        print(f"  '{key}': {value.shape}")
    else:
        # For non-tensor data like lists of strings (labels)
        print(f"  '{key}': type={type(value)}, len={len(value) if hasattr(value, '__len__') else 'N/A'}")

# %%

import torch
from training.eval_utils.transform_utils import global_points_from_cam

# 1. Transform 'pts3d_cam' to world coordinates
print("\n--- Verifying Coordinate Transformation ---")
pts3d_cam = batch['pts3d_cam']
camera_pose = batch['camera_pose']
transformed_pts3d = global_points_from_cam(camera_pose, pts3d_cam)

# 2. Compare with the ground truth 'pts3d'
pts3d_gt = batch['pts3d']

# Using torch.allclose for robust floating-point comparison
are_close = torch.allclose(transformed_pts3d, pts3d_gt, atol=1e-6)

print(f"Transformation Correctness Verification: {'SUCCESS' if are_close else 'FAILURE'}")
if not are_close:
    # Optional: print difference for debugging
    difference = torch.abs(transformed_pts3d - pts3d_gt)
    avg_diff = difference.mean()
    max_diff = difference.max()
    print(f"  - Average difference: {avg_diff.item()}")
    print(f"  - Maximum difference: {max_diff.item()}")


# %%
# --- Import or Paste your functions here ---
# from your_module import align_pred_to_gt_torch_batch_roma, align_c2w_poses_points_torch
from training.eval_utils.align_utils.umeyama_alignment import align_pred_to_gt_torch_batch_roma, align_c2w_poses_points_torch
def test_validation_scheme(c2w_camera_pose: torch.Tensor, pts3d: torch.Tensor):
    """
    Validates Sim3 alignment methods using the provided c2w poses and 3D points
    as the 'Source' data.
    
    Args:
        c2w_camera_pose: (B, S, 4, 4) Camera to World matrices
        pts3d: (B, S, W, H, 3) World coordinate 3D points
    """
    print(f"\n--- Starting Validation with Inputs: {c2w_camera_pose.shape} & {pts3d.shape} ---")
    
    # 0. Setup & Standardization
    device = c2w_camera_pose.device
    dtype = torch.float32 # Use float32 for numerical stability during validation
    
    # Ensure inputs are float32
    c2w_src = c2w_camera_pose.to(dtype=dtype)
    pts_src = pts3d.to(dtype=dtype)
    
    B, S, _, _ = c2w_src.shape
    _, _, H, W, _ = pts_src.shape

    # ==========================================================================
    # STEP 1: Generate Synthetic "Ground Truth" (Target)
    # We apply a KNOWN random Sim3 transformation to your inputs.
    # If your methods are correct, they must be able to reverse this process.
    # ==========================================================================
    
    # 1.1 Create Random Sim3 Parameters
    print("1. Generating random Sim3 transformation (Ground Truth)...")
    
    # Scale: Random value between 0.5 and 2.5
    sim3_s = (torch.rand(B, device=device, dtype=dtype) * 2.0 + 0.5)
    
    # Translation: Random large offsets
    sim3_t = torch.randn(B, 3, device=device, dtype=dtype) * 10.0
    
    # Rotation: Random rotation matrices via QR decomposition
    rand_mat = torch.randn(B, 3, 3, device=device, dtype=dtype)
    sim3_R, _ = torch.linalg.qr(rand_mat)

    # 1.2 Apply Sim3 to Points to get pts_tgt (Target Points)
    # Formula: P_tgt = s * (P_src @ R^T) + t
    flat_pts = pts_src.reshape(B, -1, 3)
    # (B, N, 3) @ (B, 3, 3).T -> (B, N, 3)
    rotated_pts = flat_pts @ sim3_R.transpose(-2, -1) 
    scaled_pts = rotated_pts * sim3_s.view(B, 1, 1)
    pts_tgt_flat = scaled_pts + sim3_t.unsqueeze(1)
    pts_tgt = pts_tgt_flat.reshape(B, S, H, W, 3)

    # 1.3 Apply Sim3 to Cameras to get c2w_tgt (Target Poses)
    # Extract rotation and translation from source poses
    R_cam_src = c2w_src[..., :3, :3] # (B, S, 3, 3)
    t_cam_src = c2w_src[..., :3, 3]  # (B, S, 3)

    # Calculate Target Rotation: R_cam_tgt = R_sim3 @ R_cam_src
    R_cam_tgt = sim3_R.unsqueeze(1) @ R_cam_src
    
    # Calculate Target Translation: t_cam_tgt = s * (R_sim3 @ t_cam_src) + t_sim3
    # Rotate camera centers
    # (B, 1, 3, 3) @ (B, S, 3, 1) -> (B, S, 3)
    t_cam_rotated = (sim3_R.unsqueeze(1) @ t_cam_src.unsqueeze(-1)).squeeze(-1)
    t_cam_tgt = sim3_s.view(B, 1, 1) * t_cam_rotated + sim3_t.unsqueeze(1)

    # Assemble Target C2W
    c2w_tgt = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(B, S, 1, 1)
    c2w_tgt[..., :3, :3] = R_cam_tgt
    c2w_tgt[..., :3, 3] = t_cam_tgt

    # ==========================================================================
    # STEP 2: Run Method 1 (Solver)
    # Can we align the original 'pts_src' to the generated 'pts_tgt' 
    # and recover the 'sim3' params we just created?
    # ==========================================================================
    print("2. Testing `align_pred_to_gt_torch_batch_roma` (Solver)...")
    
    _, pred_params = align_pred_to_gt_torch_batch_roma(
        pred_points=pts_src,
        gt_points=pts_tgt,
        with_scale=True,
        return_points=False # We'll check points in Step 3
    )

    # Check Parameter Accuracy
    diff_s = (pred_params['scale'] - sim3_s).abs().max().item()
    diff_R = (pred_params['rotation'] - sim3_R).abs().max().item()
    diff_t = (pred_params['translation'] - sim3_t).abs().max().item()

    print(f"   > Scale Error:       {diff_s:.2e}")
    print(f"   > Rotation Error:    {diff_R:.2e}")
    print(f"   > Translation Error: {diff_t:.2e}")

    assert diff_s < 1e-4, "❌ Scale recovery failed!"
    assert diff_R < 1e-4, "❌ Rotation recovery failed!"
    assert diff_t < 1e-4, "❌ Translation recovery failed!"
    print("   ✅ Solver correctly recovered the ground truth Sim3 parameters.")

    # ==========================================================================
    # STEP 3: Run Method 2 (Application)
    # Apply the recovered parameters to the SOURCE cameras and points.
    # The result should match the TARGET cameras and points exactly.
    # ==========================================================================
    print("3. Testing `align_c2w_poses_points_torch` (Application)...")
    
    aligned_c2w, aligned_pts = align_c2w_poses_points_torch(
        c2w_poses=c2w_src,
        transform_params=pred_params,
        points3D=pts_src,
        with_scale=True
    )

    # Check Points Alignment
    pts_error = (aligned_pts - pts_tgt).abs().mean().item()
    print(f"   > Points Alignment Mean Error: {pts_error:.2e}")
    assert pts_error < 1e-4, "❌ Point cloud alignment failed!"

    # Check Camera Rotation Alignment
    cam_R_error = (aligned_c2w[..., :3, :3] - c2w_tgt[..., :3, :3]).abs().mean().item()
    print(f"   > Camera Rotation Mean Error:  {cam_R_error:.2e}")
    assert cam_R_error < 1e-4, "❌ Camera rotation alignment failed!"

    # Check Camera Translation Alignment
    cam_t_error = (aligned_c2w[..., :3, 3] - c2w_tgt[..., :3, 3]).abs().mean().item()
    print(f"   > Camera Center Mean Error:    {cam_t_error:.2e}")
    assert cam_t_error < 1e-4, "❌ Camera translation alignment failed!"
    
    print("   ✅ Camera poses and points were transformed correctly.")
    print("\n🎉 VALIDATION SUCCESS: Both methods are verified correct and consistent.")

test_validation_scheme(batch['camera_pose'], batch['pts3d'])
# %%
def test_strict_geometric_consistency(c2w_camera_pose: torch.Tensor, pts3d: torch.Tensor):
    print(f"\n--- Starting STRICT Geometric Validation ---")
    
    device = c2w_camera_pose.device
    dtype = torch.float32
    
    # Ensure inputs are float32
    c2w_src = c2w_camera_pose.to(dtype=dtype)
    pts_src = pts3d.to(dtype=dtype)
    
    B, S, _, _ = c2w_src.shape
    _, _, H, W, _ = pts_src.shape

    # --- Setup: Generate the same synthetic transformation as before ---
    sim3_s = (torch.rand(B, device=device, dtype=dtype) * 2.0 + 0.5)
    sim3_t = torch.randn(B, 3, device=device, dtype=dtype) * 10.0
    rand_mat = torch.randn(B, 3, 3, device=device, dtype=dtype)
    sim3_R, _ = torch.linalg.qr(rand_mat)

    # Create Target (Ground Truth) using Manual Sim3 application
    # (Same logic as previous script, condensed)
    flat_pts = pts_src.reshape(B, -1, 3)
    rotated_pts = flat_pts @ sim3_R.transpose(-2, -1) 
    pts_tgt = (rotated_pts * sim3_s.view(B, 1, 1) + sim3_t.unsqueeze(1)).reshape(B, S, H, W, 3)

    # Run Solver
    _, pred_params = align_pred_to_gt_torch_batch_roma(
        pred_points=pts_src, gt_points=pts_tgt, with_scale=True, return_points=False
    )
    
    # Run Application
    aligned_c2w, aligned_pts = align_c2w_poses_points_torch(
        c2w_poses=c2w_src, transform_params=pred_params, points3D=pts_src, with_scale=True
    )
    
    recovered_s = pred_params['scale'].view(B, 1, 1)

    # ==========================================================================
    # STRICT TEST 1: Relative Pose Invariance
    # Calculate relative transform between View 0 and View 1
    # T_rel = inv(T_0) @ T_1
    # ==========================================================================
    print("1. Strict Test: Relative Pose Invariance...")
    
    # Source Relative Poses
    pose_0_src = c2w_src[:, 0] # (B, 4, 4)
    pose_1_src = c2w_src[:, 1] # (B, 4, 4)
    # T_rel = T0_inv @ T1
    rel_pose_src = torch.linalg.inv(pose_0_src) @ pose_1_src
    
    # Aligned Relative Poses
    pose_0_aligned = aligned_c2w[:, 0]
    pose_1_aligned = aligned_c2w[:, 1]
    rel_pose_aligned = torch.linalg.inv(pose_0_aligned) @ pose_1_aligned
    
    # CHECK ROTATION: Relative rotation should be IDENTICAL (Sim3 preserves angles)
    R_rel_src = rel_pose_src[:, :3, :3]
    R_rel_aln = rel_pose_aligned[:, :3, :3]
    
    diff_rel_R = (R_rel_src - R_rel_aln).abs().mean().item()
    print(f"   > Relative Rotation Delta: {diff_rel_R:.2e}")
    assert diff_rel_R < 1e-5, "❌ Relative Rotation is NOT preserved! Scene is distorted."
    
    # CHECK TRANSLATION: Relative translation should be scaled by s
    t_rel_src = rel_pose_src[:, :3, 3]
    t_rel_aln = rel_pose_aligned[:, :3, 3]
    
    # t_rel_aligned should equal t_rel_src * s
    expected_t_rel = t_rel_src * recovered_s.squeeze(-1)
    diff_rel_t = (t_rel_aln - expected_t_rel).abs().mean().item()
    
    print(f"   > Relative Translation Delta (scaled): {diff_rel_t:.2e}")
    assert diff_rel_t < 1e-5, "❌ Relative Translation does not match scale factor!"
    print("   ✅ Relative Poses preserved correctly.")

    # ==========================================================================
    # STRICT TEST 2: Local Camera Coordinate Consistency
    # Select a point in World Space. Transform it to Camera Space.
    # Verify: P_cam_aligned == s * P_cam_source
    # This proves the points and cameras are moving *together*.
    # ==========================================================================
    print("2. Strict Test: Local Camera Coordinate Consistency...")

    # Pick the center point of the image in View 0 for every batch
    # pts_src: (B, S, H, W, 3)
    test_pt_src = pts_src[:, 0, H//2, W//2, :] # (B, 3)
    test_pt_aln = aligned_pts[:, 0, H//2, W//2, :] # (B, 3)
    
    # Camera 0 Poses
    cam0_src = c2w_src[:, 0]
    cam0_aln = aligned_c2w[:, 0]
    
    # Transform World Point to Camera Space: P_cam = R^T (P_world - t)
    # Source Frame
    R_src = cam0_src[:, :3, :3]
    t_src = cam0_src[:, :3, 3]
    p_local_src = (R_src.transpose(1,2) @ (test_pt_src - t_src).unsqueeze(-1)).squeeze(-1)
    
    # Aligned Frame
    R_aln = cam0_aln[:, :3, :3]
    t_aln = cam0_aln[:, :3, 3]
    p_local_aln = (R_aln.transpose(1,2) @ (test_pt_aln - t_aln).unsqueeze(-1)).squeeze(-1)
    
    # Check: p_local_aln should be exactly s * p_local_src
    expected_p_local = p_local_src * recovered_s.squeeze(-1)
    
    diff_local = (p_local_aln - expected_p_local).abs().mean().item()
    print(f"   > Local Coordinate Delta: {diff_local:.2e}")
    
    assert diff_local < 1e-4, "❌ Points and Cameras have drifted relative to each other!"
    print("   ✅ Points are locked to cameras correctly.")
    
    print("\n🎉 STRICT VALIDATION PASSED.")

test_strict_geometric_consistency(batch['camera_pose'], batch['pts3d'])
# %%
import torch
from training.eval_utils.align_utils.umeyama_alignment import align_pred_to_gt_torch_batch_roma, align_c2w_poses_points_torch

def test_inverse_consistency(c2w_camera_pose: torch.Tensor, pts3d: torch.Tensor):
    """
    Validates that `align_c2w_poses_points_torch` and `align_pred_to_gt_torch_batch_roma`
    are consistent inverses of each other.

    Logic:
    1. Define random transform T.
    2. Target = align_c2w(Source, T)
    3. T_recovered = align_pred_to_gt(pred=Target, gt=Source)
    4. Verify T_recovered == Inverse(T)
    5. Reconstructed_Source = align_c2w(Target, T_recovered)
    6. Verify Reconstructed_Source == Source
    """
    print(f"\n--- Starting Inverse Consistency (Round-Trip) Validation ---")
    
    device = c2w_camera_pose.device
    dtype = torch.float32
    
    # Force float32 inputs
    c2w_src = c2w_camera_pose.to(dtype=dtype)
    pts_src = pts3d.to(dtype=dtype)
    B = c2w_src.shape[0]

    # ==========================================================================
    # STEP 1: Create Random Transform T (The "Forward" Transform)
    # ==========================================================================
    print("1. Generating random Sim3 transform T...")
    
    # Scale
    s_fwd = (torch.rand(B, device=device, dtype=dtype) * 2.0 + 0.5)
    # Translation
    t_fwd = torch.randn(B, 3, device=device, dtype=dtype) * 5.0
    # Rotation
    rand_mat = torch.randn(B, 3, 3, device=device, dtype=dtype)
    R_fwd, _ = torch.linalg.qr(rand_mat)

    params_fwd = {
        'scale': s_fwd,
        'rotation': R_fwd,
        'translation': t_fwd
    }

    # ==========================================================================
    # STEP 2: Apply T to Source -> Generate Target
    # P_tgt = s * (P_src @ R^T) + t
    # ==========================================================================
    print("2. Applying T to Source to generate Target (using align_c2w)...")
    
    c2w_tgt, pts_tgt = align_c2w_poses_points_torch(
        c2w_poses=c2w_src,
        transform_params=params_fwd,
        points3D=pts_src,
        with_scale=True
    )

    # ==========================================================================
    # STEP 3: Recover Transform from Target -> Source
    # We ask the solver: "How do I move Target back to Source?"
    # ==========================================================================
    print("3. Recovering transform T_inv from Target->Source (using align_pred_to_gt)...")
    
    # Important: pred=Target, gt=Source
    # This means we want to find T_inv such that: T_inv(Target) ~= Source
    _, params_rec = align_pred_to_gt_torch_batch_roma(
        pred_points=pts_tgt,
        gt_points=pts_src,
        with_scale=True,
        return_points=False
    )

    # ==========================================================================
    # STEP 4: Verify T_rec is mathematically the Inverse of T
    # ==========================================================================
    print("4. Verifying Mathematical Inverse Properties...")
    
    # 4a. Check Scale: s_rec should be 1 / s_fwd
    s_rec = params_rec['scale']
    s_expected = 1.0 / s_fwd
    diff_s = (s_rec - s_expected).abs().max().item()
    print(f"   > Scale Inverse Error:       {diff_s:.2e}")
    assert diff_s < 1e-4, "❌ Scale inverse check failed"

    # 4b. Check Rotation: R_rec should be R_fwd.T (Inverse Rotation)
    R_rec = params_rec['rotation']
    R_expected = R_fwd.transpose(-2, -1)
    diff_R = (R_rec - R_expected).abs().max().item()
    print(f"   > Rotation Inverse Error:    {diff_R:.2e}")
    assert diff_R < 1e-4, "❌ Rotation inverse check failed"

    # 4c. Check Translation: 
    # Forward: P_tgt = s * P_src * R^T + t
    # Inverse: P_src = (1/s) * P_tgt * R - (1/s) * t * R
    # So t_rec should be: - (t_fwd @ R_fwd) / s_fwd
    t_rec = params_rec['translation']
    
    # Careful with dimensions for t @ R
    # t_fwd: (B, 3) -> (B, 1, 3)
    # R_fwd: (B, 3, 3)
    t_rotated = (t_fwd.unsqueeze(1) @ R_fwd).squeeze(1)
    t_expected = -t_rotated / s_fwd.unsqueeze(1)
    
    diff_t = (t_rec - t_expected).abs().max().item()
    print(f"   > Translation Inverse Error: {diff_t:.2e}")
    assert diff_t < 1e-4, "❌ Translation inverse check failed"

    print("   ✅ Parameter Inverse Check Passed.")

    # ==========================================================================
    # STEP 5: Data Round-Trip Check
    # Apply T_rec to Target. We should land EXACTLY back on Source.
    # ==========================================================================
    print("5. Verifying Data Round-Trip (Target + T_inv == Source?)...")

    c2w_reconstructed, pts_reconstructed = align_c2w_poses_points_torch(
        c2w_poses=c2w_tgt,
        transform_params=params_rec,
        points3D=pts_tgt,
        with_scale=True
    )

    # Check Points
    pts_roundtrip_error = (pts_reconstructed - pts_src).abs().mean().item()
    print(f"   > Points Round-Trip Mean Error: {pts_roundtrip_error:.2e}")
    
    # Check Poses
    pose_t_error = (c2w_reconstructed[..., :3, 3] - c2w_src[..., :3, 3]).abs().mean().item()
    print(f"   > Pose Center Round-Trip Error: {pose_t_error:.2e}")

    assert pts_roundtrip_error < 1e-4, "❌ Points did not return to original position"
    assert pose_t_error < 1e-4, "❌ Poses did not return to original position"

    print("   ✅ Data Round-Trip Passed.")
    print("\n🎉 SUCCESS: The two functions are consistent inverses of each other.")

# Run with your batch data
test_inverse_consistency(batch['camera_pose'], batch['pts3d'])
# %%
import torch
import math
from training.eval_utils.align_utils.umeyama_alignment import align_pred_to_gt_torch_batch_roma, align_c2w_poses_points_torch

def test_manual_geometric_truth():
    print("\n--- Starting 'Manual Oracle' Truth Test ---")
    print("(Validating against manual mental math, not random tensors)")
    
    # B=1 for simplicity, we will broadcast later
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # =========================================================
    # SCENARIO SETUP: "The Shifted Cube"
    # =========================================================
    
    # 1. Source Point: (1, 0, 0)
    # Shape: (B=1, S=1, W=1, H=1, 3)
    pts_src = torch.tensor([[[ [[1.0, 0.0, 0.0]] ]]], device=device, dtype=dtype)
    
    # 2. Source Camera: At Origin (0,0,0), Looking down Z
    # Identity Matrix
    c2w_src = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4)

    # 3. Define Transform MANUALLY
    # Scale = 2.0
    s = torch.tensor([2.0], device=device, dtype=dtype)
    
    # Translation = (0, 0, 5)
    t = torch.tensor([[0.0, 0.0, 5.0]], device=device, dtype=dtype)
    
    # Rotation = 90 degrees around Z-axis
    # [ cos -sin  0]   [ 0 -1  0]
    # [ sin  cos  0] -> [ 1  0  0]
    # [  0    0   1]   [ 0  0  1]
    R = torch.tensor([[
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ]], device=device, dtype=dtype)

    params = {'scale': s, 'rotation': R, 'translation': t}

    # =========================================================
    # TEST 1: FORWARD APPLICATION (align_c2w)
    # Does the code move the point to (0, 2, 5)?
    # =========================================================
    print("\n1. Testing Forward Application (align_c2w)...")
    
    # We expect:
    # (1,0,0) --RotZ(90)--> (0,1,0) --Scale(2)--> (0,2,0) --Trans(0,0,5)--> (0,2,5)
    EXPECTED_PT = torch.tensor([0.0, 2.0, 5.0], device=device)
    
    _, pts_res = align_c2w_poses_points_torch(
        c2w_src, params, points3D=pts_src, with_scale=True
    )
    
    pt_result = pts_res.view(-1)
    print(f"   > Source Point:   (1.0, 0.0, 0.0)")
    print(f"   > Transform:      RotZ(90) + Scale(2) + TransZ(5)")
    print(f"   > Expected Point: {EXPECTED_PT.cpu().numpy()}")
    print(f"   > Actual Point:   {pt_result.cpu().numpy()}")
    
    err = (pt_result - EXPECTED_PT).abs().sum().item()
    if err < 1e-5:
        print("   ✅ FORWARD PASS IS CORRECT (Matches Mental Math)")
    else:
        print("   ❌ FORWARD PASS FAILED. Math is wrong.")
        return # Stop if forward is wrong

    # =========================================================
    # TEST 2: SOLVER RECOVERY (align_pred)
    # Can the solver take (0,2,5) and (1,0,0) and tell us "Scale=2"?
    # =========================================================
    print("\n2. Testing Solver Recovery (align_pred_to_gt)...")
    
    # We treat pts_src (1,0,0) as GT, and pts_res (0,2,5) as PRED
    # WAIT! The function is align_PRED_to_GT.
    # If we want to recover the transform that turned SRC -> RES...
    # We input PRED=SRC, GT=RES.
    # Solver finds T such that T(SRC) = RES.
    
    _, params_rec = align_pred_to_gt_torch_batch_roma(
        pred_points=pts_src, # (1,0,0)
        gt_points=pts_res,   # (0,2,5)
        with_scale=True,
        return_points=False
    )
    
    rec_s = params_rec['scale'].item()
    rec_t = params_rec['translation'].view(-1)
    rec_R = params_rec['rotation'].squeeze()
    
    print(f"   > Recovered Scale (Exp 2.0): {rec_s:.5f}")
    print(f"   > Recovered Trans (Exp 0,0,5): {rec_t.cpu().numpy()}")
    
    # Check Scale
    if abs(rec_s - 2.0) < 1e-4:
        print("   ✅ Scale Recovered Correctly")
    else:
        print("   ❌ Scale Recovery Failed")

    # Check Translation
    if (rec_t - t.view(-1)).abs().sum() < 1e-4:
        print("   ✅ Translation Recovered Correctly")
    else:
        print("   ❌ Translation Recovery Failed")

    # Check Rotation
    if (rec_R - R.squeeze()).abs().sum() < 1e-4:
        print("   ✅ Rotation Recovered Correctly")
    else:
        print("   ❌ Rotation Recovery Failed")

    # =========================================================
    # TEST 3: CAMERA CENTER CHECK
    # Camera was at (0,0,0). 
    # RotZ(90) on (0,0,0) is (0,0,0). Scale(2) is (0,0,0).
    # Translation (0,0,5) -> Final Pos should be (0,0,5).
    # =========================================================
    print("\n3. Testing Camera Center Movement...")
    c2w_res, _ = align_c2w_poses_points_torch(
        c2w_src, params, points3D=None, with_scale=True
    )
    
    cam_pos = c2w_res[0, 0, :3, 3]
    print(f"   > Orig Cam Pos: (0,0,0)")
    print(f"   > New Cam Pos:  {cam_pos.cpu().numpy()}")
    
    if (cam_pos - torch.tensor([0.0, 0.0, 5.0], device=device)).abs().sum() < 1e-4:
        print("   ✅ Camera moved correctly to (0,0,5)")
    else:
        print("   ❌ Camera movement failed.")

    print("\n🎉 FINAL VERDICT: If you see all Green Checks, the code is ABSOLUTELY CORRECT.")

test_manual_geometric_truth()
# %%
