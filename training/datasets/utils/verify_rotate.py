import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from rotate import rotate_k90_aug
from verify_utils import depthmap_to_absolute_camera_coordinates


def verify_one_k(image, depth, intrinsics, pose, k, atol=1e-6, rtol=0.0):
    """
    Verify that rotating (image, depth, K, c2w) by k*90 degrees preserves the world-space
    pointmap up to a pixel-space undo-rotation.

    k can be: 1 (CCW 90), -1 (CW 90), 2 (180), 3 (CW 90).
    """
    # 1) Unproject BEFORE rotation
    points_before, _, _ = depthmap_to_absolute_camera_coordinates(depth, intrinsics, pose)

    # 2) Apply rotation augmentation
    image_rot, depth_rot, intrinsics_new, pose_new, _ = rotate_k90_aug(
        image, depth, intrinsics, pose, k=k
    )

    # 3) Unproject AFTER rotation
    points_after, _, _ = depthmap_to_absolute_camera_coordinates(depth_rot, intrinsics_new, pose_new)

    # 4) Undo the pixel rotation on the AFTER pointmap so we can compare pixel-to-pixel
    # If we rotated by k, undo by -k.
    points_after_realigned = np.rot90(points_after, k=-k)

    # 5) Compare
    ok = np.allclose(points_before, points_after_realigned, atol=atol, rtol=rtol)

    report = {
        "k": k,
        "before_shape": points_before.shape,
        "after_shape": points_after.shape,
        "realigned_shape": points_after_realigned.shape,
        "ok": ok,
    }

    if not ok:
        diff = np.abs(points_before - points_after_realigned)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        report.update(
            {
                "max_diff_idx": max_idx,
                "max_diff": float(diff[max_idx]),
                "point_before": points_before[max_idx[0], max_idx[1]].copy(),
                "point_after_realigned": points_after_realigned[max_idx[0], max_idx[1]].copy(),
            }
        )

    return report


def verify():
    print("--- Starting Verification (Pixel-to-Pixel Method) ---")
    np.random.seed(22)

    # Mock data
    H, W = 640, 480
    image = np.random.rand(H, W, 3).astype(np.float32)
    print(image)

    # Depth with unique gradient values (stable for debugging)
    depth = np.fromfunction(lambda j, i: (j / H) + (i / W) + 1.0, (H, W), dtype=np.float32).astype(np.float32)

    intrinsics_cases = [
        (
            "centerish",
            np.array(
                [
                    [436.11008, 0.0, 240.91835],
                    [0.0, 436.11008, 322.66653],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        ),
        (
            "near_corner",
            np.array(
                [
                    [436.11008, 0.0, 5.0],
                    [0.0, 436.11008, 5.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        ),
        (
            "outside",
            np.array(
                [
                    [436.11008, 0.0, -30.0],
                    [0.0, 436.11008, H + 10.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        ),
    ]

    # Camera-to-world pose (c2w)
    pose = np.array(
        [
            [9.8153263e-01, -7.7415816e-02, 1.7492986e-01, -6.1428569e-02],
            [7.7405863e-02, 9.9697584e-01, 6.8903100e-03, -4.0515000e-04],
            [-1.7493425e-01, 6.7775301e-03, 9.9845568e-01, 1.0289070e-02],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    print(f"Original Dimensions: H={H}, W={W}")
    print(f"Top-left RGB pixel (seeded): {image[0, 0]}")

    # Verify k in {+90, -90, 180}
    ks = [1, -1, 2]
    atol = 1e-6
    rtol = 0.0

    print("\n--- Running checks ---")
    all_ok = True
    for label, intrinsics in intrinsics_cases:
        print(f"\n[Intrinsics] {label} (cx={intrinsics[0,2]:.2f}, cy={intrinsics[1,2]:.2f})")
        for k in ks:
            print(f"\n[Check] k={k} (rotation = {k*90} degrees)")
            rep = verify_one_k(image, depth, intrinsics, pose, k=k, atol=atol, rtol=rtol)

            print(f"  Pointmap shapes: before {rep['before_shape']}, after {rep['after_shape']}, realigned {rep['realigned_shape']}")
            if rep["ok"]:
                print("  SUCCESS: pointmaps match after undo-rotation.")
            else:
                all_ok = False
                print("  FAILURE: pointmaps do NOT match.")
                print(f"  Max diff at pixel {rep['max_diff_idx'][:2]} (channel {rep['max_diff_idx'][2]}): {rep['max_diff']}")
                ij = rep["max_diff_idx"][:2]
                print(f"  Point Before @ {ij}:            {rep['point_before']}")
                print(f"  Point After (realigned) @ {ij}: {rep['point_after_realigned']}")
    print("\n--- Summary ---")
    if all_ok:
        print("ALL SUCCESS: rotate_k90_aug is verified for 90, -90, and 180 degrees (pixel-to-pixel pointmap invariance).")
    else:
        print("SOME FAILURES: at least one rotation case failed invariance; inspect the max-diff diagnostics above.")


if __name__ == "__main__":
    verify()
