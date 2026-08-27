import unittest
from unittest.mock import patch

import numpy as np
import torch

from eval_utils.eval_wrapper import _select_pred_world_points, eval_batch
from eval_utils.pose_from_points import estimate_poses_from_world_points


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    @classmethod
    def nested(cls, value):
        if isinstance(value, dict):
            return cls({key: cls.nested(item) for key, item in value.items()})
        return value


class PoseFromPointsTest(unittest.TestCase):
    @staticmethod
    def _synthetic_point_maps():
        height, width = 32, 40
        focal_candidates = np.geomspace(
            max(height, width) * 0.5, max(height, width) * 3.0, 41
        )
        focal = float(focal_candidates[20])
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        depth = (
            2.0 + 0.2 * torch.sin(x.float() * 0.37) + 0.15 * torch.cos(y.float() * 0.29)
        )
        camera_points = torch.stack(
            (
                (x.float() - width / 2.0) * depth / focal,
                (y.float() - height / 2.0) * depth / focal,
                depth,
            ),
            dim=-1,
        )

        poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
        angle = 0.12
        poses[0, 1, :3, :3] = torch.tensor(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=torch.float32,
        )
        poses[0, 1, :3, 3] = torch.tensor([0.35, -0.08, 0.12])

        point_maps = []
        for pose in poses[0]:
            point_maps.append(
                camera_points @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]
            )
        return torch.stack(point_maps).unsqueeze(0), poses, focal

    def test_recovers_c2w_pose_with_focal_search(self):
        points, expected_poses, expected_focal = self._synthetic_point_maps()
        confidence = torch.full(points.shape[:-1], 2.0)
        poses, focals, failures = estimate_poses_from_world_points(
            points,
            confidence,
            {"mode": "threshold", "threshold": 1.0},
            {
                "iterations": 100,
                "reprojection_error": 0.05,
                "focal_candidates": 41,
                "focal_min_factor": 0.5,
                "focal_max_factor": 3.0,
            },
            pose_convention="c2w",
        )

        self.assertFalse(failures.any())
        torch.testing.assert_close(
            focals, torch.full_like(focals, expected_focal), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(poses, expected_poses, rtol=2e-3, atol=2e-3)

    def test_threshold_failure_returns_identity(self):
        points, _, _ = self._synthetic_point_maps()
        confidence = torch.ones(points.shape[:-1])
        poses, focals, failures = estimate_poses_from_world_points(
            points,
            confidence,
            {"mode": "threshold", "threshold": 1.0},
            {"focal_candidates": 5},
        )

        self.assertTrue(failures.all())
        self.assertTrue(torch.isnan(focals).all())
        expected = torch.eye(4).view(1, 1, 4, 4).expand_as(poses)
        torch.testing.assert_close(poses, expected)

    def test_point_selection_precedence_and_missing(self):
        world = torch.tensor(1.0)
        aligned = torch.tensor(2.0)
        keys = {"world_points": "pts3d", "aligned_world_points": "aligned_pts3d"}

        key, value = _select_pred_world_points(
            {"pts3d": world, "aligned_pts3d": aligned}, keys
        )
        self.assertEqual(key, "aligned_pts3d")
        self.assertIs(value, aligned)

        key, value = _select_pred_world_points({"pts3d": world}, keys)
        self.assertEqual(key, "pts3d")
        self.assertIs(value, world)
        self.assertEqual(_select_pred_world_points({}, keys), (None, None))

    def test_eval_silently_skips_when_points_are_missing(self):
        poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
        metrics_conf = AttrDict.nested(
            {
                "camera": {
                    "enabled": True,
                    "abs_err": True,
                    "rel_err": False,
                    "auc": False,
                    "pts3d_pose": {"enabled": True},
                },
                "intrinsics": {"enabled": False},
                "depth": {"enabled": False},
                "recon": {"pts_err": False},
            }
        )
        metrics = eval_batch(
            {"camera_pose": poses.clone()},
            {"camera_pose": poses},
            metrics_conf,
            {"extrinsics": "camera_pose"},
            {"extrinsics": "camera_pose", "world_points": "pts3d"},
        )
        self.assertFalse(any(key.startswith("pts3d_") for key in metrics))

    def test_eval_prefixes_point_pose_metrics(self):
        poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
        poses[0, 1, 0, 3] = 1.0
        points = torch.zeros(1, 2, 2, 2, 3)
        metrics_conf = AttrDict.nested(
            {
                "camera": {
                    "enabled": True,
                    "abs_err": True,
                    "rel_err": True,
                    "auc": True,
                    "pts3d_pose": {"enabled": True, "normalize_translation": True},
                },
                "intrinsics": {"enabled": False},
                "depth": {"enabled": False},
                "recon": {"pts_err": False},
            }
        )
        pnp_result = (
            poses.cpu(),
            torch.ones(1, 2),
            torch.zeros(1, 2, dtype=torch.bool),
        )
        with patch(
            "eval_utils.eval_wrapper.estimate_poses_from_world_points",
            return_value=pnp_result,
        ):
            metrics = eval_batch(
                {"camera_pose": poses.clone(), "pts3d": points},
                {"camera_pose": poses},
                metrics_conf,
                {"extrinsics": "camera_pose"},
                {"extrinsics": "camera_pose", "world_points": "pts3d"},
            )

        self.assertIn("pts3d_rot_error_mean_deg", metrics)
        self.assertIn("pts3d_all_pairs_rot_error_deg", metrics)
        self.assertIn("pts3d_auc", metrics)
        self.assertEqual(metrics["pts3d_pnp_failure_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
