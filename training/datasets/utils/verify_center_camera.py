import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from cropping import get_center_camera
from verify_utils import depthmap_to_absolute_camera_coordinates


def random_rotation(rng):
    # Deterministic random rotation matrix via SVD.
    a = rng.normal(size=(3, 3))
    u, _, vt = np.linalg.svd(a)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r.astype(np.float32)


def rotation_is_orthonormal(r, atol=1e-6):
    if r.shape != (3, 3):
        return False
    should_be_identity = r.T @ r
    return np.allclose(should_be_identity, np.eye(3), atol=atol) and np.isclose(
        np.linalg.det(r), 1.0, atol=atol
    )


def verify_pp_centering(image, depth, intrinsics, pose, atol=1e-6, rtol=0.0):
    points_before, _, _ = depthmap_to_absolute_camera_coordinates(
        depth, intrinsics, pose
    )

    intrinsics_new, pose_new = get_center_camera(
        intrinsics, pose, depthmap=depth
    )
    print(f"{pose_new=}")
    print(f"{intrinsics_new=}")

    points_after, _, _ = depthmap_to_absolute_camera_coordinates(
        depth, intrinsics_new, pose_new
    )

    ok = np.allclose(points_before, points_after, atol=atol, rtol=rtol)
    report = {
        "ok": ok,
        "before_shape": points_before.shape,
        "after_shape": points_after.shape,
        "cxcy_before": (float(intrinsics[0, 2]), float(intrinsics[1, 2])),
        "cxcy_after": (float(intrinsics_new[0, 2]), float(intrinsics_new[1, 2])),
        "pose_rot_orthonormal": rotation_is_orthonormal(pose[:3, :3], atol=atol),
        "pose_new_rot_orthonormal": rotation_is_orthonormal(
            pose_new[:3, :3], atol=atol
        ),
    }

    if not ok:
        diff = np.abs(points_before - points_after)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        report.update(
            {
                "max_diff_idx": max_idx,
                "max_diff": float(diff[max_idx]),
                "point_before": points_before[max_idx[0], max_idx[1]].copy(),
                "point_after": points_after[max_idx[0], max_idx[1]].copy(),
            }
        )

    return report


def verify():
    print("--- Starting get_center_camera Verification ---")
    rng = np.random.default_rng(22)

    # Mock data
    H, W = 32, 16
    image = rng.random((H, W, 3), dtype=np.float32)
    depth = rng.random((H, W), dtype=np.float32) + 0.5

    # Principal point at (5, 6)
    fx, fy = 50.0, 55.0
    intrinsics = np.array(
        [
            [fx, 0.0, 5.0],
            [0.0, fy, 6.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    r = random_rotation(rng)
    t = rng.normal(size=(3,)).astype(np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = r
    pose[:3, 3] = t

    print(f"Image shape: {image.shape}, Depth shape: {depth.shape}")
    print(f"PP before: ({intrinsics[0, 2]}, {intrinsics[1, 2]})")

    rep = verify_pp_centering(image, depth, intrinsics, pose, atol=1e-6, rtol=0.0)

    print(f"PP after: ({rep['cxcy_after'][0]}, {rep['cxcy_after'][1]})")
    print(f"Pointmap shapes: before {rep['before_shape']}, after {rep['after_shape']}")
    print(f"Pose R orthonormal: {rep['pose_rot_orthonormal']}")
    print(f"Pose_new R orthonormal: {rep['pose_new_rot_orthonormal']}")
    if rep["ok"]:
        print("SUCCESS: pointmaps match after principal point centering.")
    else:
        print("FAILURE: pointmaps do NOT match after principal point centering.")
        ij = rep["max_diff_idx"][:2]
        print(f"Max diff at pixel {ij}: {rep['max_diff']}")
        print(f"Point before @ {ij}: {rep['point_before']}")
        print(f"Point after @ {ij}:  {rep['point_after']}")


if __name__ == "__main__":
    verify()
