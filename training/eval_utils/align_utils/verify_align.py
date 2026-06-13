#%%
import torch
from umeyama_alignment import (
    align_pred_to_gt_torch_batch_roma,
    reverse_transform,
)

def test_strict_robustness():
    print("\n--- Starting STRICT Robustness & Isolation Tests ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # ==========================================================================
    # TEST 1: The "Poison Pill" (Weighted Outlier Rejection)
    # Can we explicitly ignore a point that is completely wrong?
    # ==========================================================================
    print("\n1. Testing Weighted Outlier Rejection ('Poison Pill')...")

    # Setup: 4 Points. 
    # First 3 form a perfect unit triangle (Valid).
    # 4th point is "Poison" - massive coordinate error.
    
    # Source (Predict): Triangle + Poison
    pts_pred_raw = [
        [0.0, 0.0, 0.0], # Pt 0
        [1.0, 0.0, 0.0], # Pt 1
        [0.0, 1.0, 0.0], # Pt 2
        [1000.0, 1000.0, 1000.0] # Pt 3 (POISON PILL)
    ]
    
    # Target (GT): Triangle (Shifted by +5Z) + [0,0,0] for Poison
    # If the solver tries to align Pt 3 (1000,1000,1000) to (0,0,0), 
    # it will skew the whole result.
    pts_gt_raw = [
        [0.0, 0.0, 5.0], # Pt 0 + 5z
        [1.0, 0.0, 5.0], # Pt 1 + 5z
        [0.0, 1.0, 5.0], # Pt 2 + 5z
        [0.0, 0.0, 0.0]  # Pt 3 (Arbitrary GT pos)
    ]
    
    # Shape: (B=1, S=1, W=4, H=1, 3)
    pts_pred = torch.tensor([[ [pts_pred_raw] ]], device=device, dtype=dtype)
    pts_gt   = torch.tensor([[ [pts_gt_raw] ]], device=device, dtype=dtype)

    # CONFIDENCE MASKS (The Critical Part)
    # 1.0 for Triangle, 0.0 for Poison
    conf_raw = [1.0, 1.0, 1.0, 0.0]
    # Actually func expects (B, S, W, H)
    pred_conf = torch.tensor([[ [conf_raw] ]], device=device, dtype=dtype) # (1,1,4,1)

    # Run Solver
    _, params = align_pred_to_gt_torch_batch_roma(
        pred_points=pts_pred,
        gt_points=pts_gt,
        pred_conf=pred_conf,
        conf_threshold=0.5, # Should keep 1.0s, drop 0.0
        with_scale=True,
        return_points=False
    )

    # Verification
    # We know the transform should be: Scale=1.0, Trans=[0,0,5]
    rec_s = params['scale'].item()
    rec_t = params['translation'].view(-1)
    
    print(f"   > Input: 3 Valid Pts + 1 Massive Outlier (Weight=0)")
    print(f"   > Expected: Scale=1.0, Trans=[0,0,5]")
    print(f"   > Actual:   Scale={rec_s:.4f}, Trans={rec_t.cpu().numpy()}")

    if abs(rec_s - 1.0) < 1e-4 and (rec_t - torch.tensor([0.,0.,5.], device=device)).abs().sum() < 1e-4:
        print("   ✅ PASSED: Solver successfully ignored the poison pill.")
    else:
        print("   ❌ FAILED: The outlier corrupted the alignment.")
        
    # ==========================================================================
    # TEST 2: The "Batch Crosstalk" Test (Isolation)
    # Does a huge scale in Batch 1 affect Batch 0?
    # ==========================================================================
    print("\n2. Testing Batch Isolation (Crosstalk Check)...")
    
    B_test = 2
    # Batch 0: Identity Transform (Scale=1)
    # Batch 1: Massive Expansion (Scale=1000)
    
    # Points: We MUST use 3 points (Triangle) to pass the N>=3 safety check in the solver.
    # (B=2, S=1, W=3, H=1, 3)
    pts_base = [
        [0.0, 0.0, 0.0], 
        [1.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0]
    ]
    pts_input = torch.tensor([ [pts_base] ], device=device, dtype=dtype).repeat(2, 1, 1, 1, 1)
    
    # GT Construction
    pts_gt_list = []
    
    # Batch 0 GT: Same as input (Identity)
    pts_gt_list.append([ [pts_base] ])
    
    # Batch 1 GT: Scaled by 1000
    pts_scaled = [
        [0.0, 0.0, 0.0], 
        [1000.0, 0.0, 0.0], 
        [0.0, 1000.0, 0.0]
    ]
    pts_gt_list.append([ [pts_scaled] ])
    
    pts_gt = torch.tensor(pts_gt_list, device=device, dtype=dtype).squeeze(1) # Handle list nesting
    
    # Run Solver
    _, params_batch = align_pred_to_gt_torch_batch_roma(
        pred_points=pts_input,
        gt_points=pts_gt,
        with_scale=True,
        return_points=False
    )
    
    s_b0 = params_batch['scale'][0].item()
    s_b1 = params_batch['scale'][1].item()
    
    print(f"   > Batch 0 Expected Scale: 1.0")
    print(f"   > Batch 0 Actual Scale:   {s_b0:.5f}")
    print(f"   > Batch 1 Expected Scale: 1000.0")
    print(f"   > Batch 1 Actual Scale:   {s_b1:.5f}")
    
    if abs(s_b0 - 1.0) < 1e-4 and abs(s_b1 - 1000.0) < 1e-1:
        print("   ✅ PASSED: Batches are perfectly isolated.")
    else:
        print("   ❌ FAILED: Batch data leaked (or solver defaulted to Identity due to insufficient points).")

    print("\n🎉 Robustness Tests Complete.")

def test_reverse():
    print("\n--- Testing Forward and Reverse Similarity Transforms ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Asymmetric, non-coplanar points avoid rotational ambiguity and exercise all
    # three coordinates. The solver applies y = s * (x @ R.T) + t.
    pred_flat = torch.tensor([
        [-1.2, 0.3, 2.1],
        [0.7, -2.0, 1.4],
        [2.3, 1.1, -0.8],
        [-0.4, 2.7, 0.5],
        [1.6, -0.9, -2.2],
        [-2.1, -1.3, 1.0],
        [0.2, 1.8, 3.1],
        [3.0, -1.5, 0.4],
    ], device=device, dtype=dtype)

    angle_x = torch.deg2rad(torch.tensor(-23.0, device=device, dtype=dtype))
    angle_z = torch.deg2rad(torch.tensor(37.0, device=device, dtype=dtype))
    one = torch.ones((), device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    rotation_x = torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, torch.cos(angle_x), -torch.sin(angle_x)]),
        torch.stack([zero, torch.sin(angle_x), torch.cos(angle_x)]),
    ])
    rotation_z = torch.stack([
        torch.stack([torch.cos(angle_z), -torch.sin(angle_z), zero]),
        torch.stack([torch.sin(angle_z), torch.cos(angle_z), zero]),
        torch.stack([zero, zero, one]),
    ])

    expected_scale = torch.tensor(2.4, device=device, dtype=dtype)
    expected_rotation = rotation_z @ rotation_x
    expected_translation = torch.tensor(
        [1.7, -3.2, 0.9], device=device, dtype=dtype
    )
    gt_flat = (
        expected_scale * (pred_flat @ expected_rotation.T)
        + expected_translation
    )

    # Shape required by align_pred_to_gt_torch_batch_roma: (B, S, W, H, 3).
    pts_pred = pred_flat.view(1, 1, -1, 1, 3)
    pts_gt = gt_flat.view(1, 1, -1, 1, 3)

    pred_aligned, pred_to_gt = align_pred_to_gt_torch_batch_roma(
        pred_points=pts_pred,
        gt_points=pts_gt,
        with_scale=True,
        return_points=True,
    )
    gt_aligned, gt_to_pred = align_pred_to_gt_torch_batch_roma(
        pred_points=pts_gt,
        gt_points=pts_pred,
        with_scale=True,
        return_points=True,
    )
    reversed_from_forward = reverse_transform(pred_to_gt)
    double_reversed = reverse_transform(reversed_from_forward)

    s_forward = pred_to_gt["scale"][0]
    R_forward = pred_to_gt["rotation"][0]
    t_forward = pred_to_gt["translation"][0]
    s_reverse = gt_to_pred["scale"][0]
    R_reverse = gt_to_pred["rotation"][0]
    t_reverse = gt_to_pred["translation"][0]
    s_helper = reversed_from_forward["scale"][0]
    R_helper = reversed_from_forward["rotation"][0]
    t_helper = reversed_from_forward["translation"][0]
    helper_aligned_flat = s_helper * (gt_flat @ R_helper.T) + t_helper

    # If gt = s * R * pred + t in column-vector notation, then:
    # pred = (1/s) * R.T * gt - (1/s) * R.T * t.
    expected_reverse_scale = 1.0 / s_forward
    expected_reverse_rotation = R_forward.T
    expected_reverse_translation = -(t_forward @ R_forward) / s_forward

    print("\nPred -> GT: gt = s_f * (pred @ R_f.T) + t_f")
    print(f"  s_f = {s_forward.item():.8f}")
    print(f"  R_f =\n{R_forward.cpu()}")
    print(f"  t_f = {t_forward.cpu()}")
    print("\nGT -> Pred: pred = s_r * (gt @ R_r.T) + t_r")
    print(f"  s_r = {s_reverse.item():.8f}")
    print(f"  R_r =\n{R_reverse.cpu()}")
    print(f"  t_r = {t_reverse.cpu()}")
    print("\nreverse_transform(pred_to_gt):")
    print(f"  s_helper = {s_helper.item():.8f}")
    print(f"  R_helper =\n{R_helper.cpu()}")
    print(f"  t_helper = {t_helper.cpu()}")
    print("\nExpected inverse relationships:")
    print(f"  s_r = 1 / s_f = {expected_reverse_scale.item():.8f}")
    print("  R_r = R_f.T")
    print("  t_r = -(t_f @ R_f) / s_f")
    print(f"      = {expected_reverse_translation.cpu()}")
    print(f"  s_f * s_r = {(s_forward * s_reverse).item():.8f}")
    print(f"  max forward point error = {(pred_aligned - pts_gt).abs().max().item():.3e}")
    print(f"  max reverse fit error = {(gt_aligned - pts_pred).abs().max().item():.3e}")
    print(f"  max helper point error = {(helper_aligned_flat - pred_flat).abs().max().item():.3e}")

    atol = 2e-5
    rtol = 2e-5
    torch.testing.assert_close(s_forward, expected_scale, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        R_forward, expected_rotation, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        t_forward, expected_translation, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        s_reverse, expected_reverse_scale, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        R_reverse, expected_reverse_rotation, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        t_reverse, expected_reverse_translation, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        s_helper, expected_reverse_scale, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        R_helper, expected_reverse_rotation, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        t_helper, expected_reverse_translation, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        reversed_from_forward["scale"], gt_to_pred["scale"], atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        reversed_from_forward["rotation"], gt_to_pred["rotation"],
        atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        reversed_from_forward["translation"], gt_to_pred["translation"],
        atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        double_reversed["scale"], pred_to_gt["scale"], atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        double_reversed["rotation"], pred_to_gt["rotation"],
        atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        double_reversed["translation"], pred_to_gt["translation"],
        atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        helper_aligned_flat, pred_flat, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(pred_aligned, pts_gt, atol=atol, rtol=rtol)
    torch.testing.assert_close(gt_aligned, pts_pred, atol=atol, rtol=rtol)

    rigid_gt_flat = pred_flat @ expected_rotation.T + expected_translation
    rigid_gt = rigid_gt_flat.view_as(pts_pred)
    _, rigid_forward = align_pred_to_gt_torch_batch_roma(
        pred_points=pts_pred,
        gt_points=rigid_gt,
        with_scale=False,
        return_points=False,
    )
    rigid_reverse = reverse_transform(rigid_forward)
    rigid_restored = (
        rigid_gt_flat @ rigid_reverse["rotation"][0].T
        + rigid_reverse["translation"][0]
    )
    assert rigid_reverse["scale"] is None
    torch.testing.assert_close(
        rigid_reverse["rotation"][0], rigid_forward["rotation"][0].T,
        atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        rigid_restored, pred_flat, atol=atol, rtol=rtol
    )
    print("  rigid-only scale remains None and its inverse also restores points")
    print("\nPASSED: forward and reverse parameters are consistent inverses.")

if __name__ == "__main__":
    test_reverse()
    # test_strict_robustness()
# %%
# import torch
# from training.eval_utils.align_utils.umeyama_alignment import align_pred_to_gt_torch_batch_roma

def test_reflection_and_degeneracy():
    print("\n--- Starting STRICT Corner Case Tests (Reflection & Degeneracy) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # ==========================================================================
    # TEST 1: The "Mirror World" Trap
    # Scenario: Target is a perfect mirror image of Source.
    # Strict Rule: A Rigid Body solver MUST NOT flip the world (Det must be +1).
    #              It should accept high error rather than breaking physics.
    # ==========================================================================
    print("\n1. Testing Reflection Handling (The 'Mirror World' Trap)...")

    # Source: Right-Handed Triangle
    # 0->(0,0), 1->(1,0), 2->(0,1)
    pts_src_raw = [[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]]
    
    # Target: Left-Handed (Mirrored) Triangle (X is negated)
    # 0->(0,0), 1->(-1,0), 2->(0,1)
    # No amount of rotation can make these match perfectly.
    pts_tgt_raw = [[0.,0.,0.], [-1.,0.,0.], [0.,1.,0.]]

    pts_src = torch.tensor([[ [pts_src_raw] ]], device=device, dtype=dtype)
    pts_tgt = torch.tensor([[ [pts_tgt_raw] ]], device=device, dtype=dtype)

    _, params = align_pred_to_gt_torch_batch_roma(
        pred_points=pts_src, gt_points=pts_tgt, with_scale=True, return_points=False
    )

    R_rec = params['rotation'].squeeze()
    det_R = torch.det(R_rec)
    
    print(f"   > Determinant of Recovered Rotation: {det_R.item():.6f}")

    # Strict Check: Determinant MUST be +1.0 (Rotation), not -1.0 (Reflection)
    if abs(det_R - 1.0) < 1e-4:
        print("   ✅ PASSED: Solver refused to flip the world (Det=+1).")
        print("      (Note: High alignment error is expected here, which is correct behavior).")
    elif abs(det_R + 1.0) < 1e-4:
        print("   ❌ FAILED: Solver cheated by creating a Reflection (Det=-1). Physics broken.")
    else:
        print(f"   ❌ FAILED: Invalid rotation matrix (Det={det_R}).")

    # ==========================================================================
    # TEST 2: The "Flatland" Trap (Planar Degeneracy)
    # Scenario: All points lie on a 2D plane (Z=0). 
    #           We rotate this plane into 3D.
    # Strict Rule: Solver must handle Rank-Deficient Covariance without crashing/NaNs.
    # ==========================================================================
    print("\n2. Testing Planar Degeneracy (The 'Flatland' Trap)...")

    # Source: A 2D grid on Z=0 (Rank 2 covariance)
    pts_plane = [
        [0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [1.,1.,0.]
    ]
    
    # Transform: Rotate 90 deg around X axis (Plane moves from XY to XZ)
    # (x, y, 0) -> (x, 0, y)
    pts_rotated = [
        [0.,0.,0.], [1.,0.,0.], [0.,0.,1.], [1.,0.,1.]
    ]
    
    pts_src_plane = torch.tensor([[ [pts_plane] ]], device=device, dtype=dtype)
    pts_tgt_plane = torch.tensor([[ [pts_rotated] ]], device=device, dtype=dtype)

    _, params_plane = align_pred_to_gt_torch_batch_roma(
        pred_points=pts_src_plane, gt_points=pts_tgt_plane, with_scale=True, return_points=False
    )
    
    # Verification: 
    # It should find the 90 deg rotation perfectly, even though Z-variance was 0.
    
    # Check Rotation (Expected: 90 deg around X)
    # [1  0  0]
    # [0  0 -1]
    # [0  1  0]
    R_plane = params_plane['rotation'].squeeze()
    
    # We check if R_plane maps (0,1,0) -> (0,0,1)
    test_vec = torch.tensor([0., 1., 0.], device=device)
    res_vec = R_plane @ test_vec # Should be [0, 0, 1]
    
    print(f"   > Planar Rotation Test Vector: {res_vec.cpu().numpy()}")
    
    if (res_vec - torch.tensor([0.,0.,1.], device=device)).abs().sum() < 1e-4:
        print("   ✅ PASSED: Solver handled planar degeneracy perfectly.")
    else:
        print("   ❌ FAILED: Solver failed on planar data (Rank Deficient).")

    print("\n🎉 Corner Case Tests Complete.")

# if __name__ == "__main__":
#     test_reflection_and_degeneracy()

# %%
