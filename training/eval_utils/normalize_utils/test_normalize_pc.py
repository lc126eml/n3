import unittest

import torch

from eval_utils.normalize_utils.normalize_pc import normalize_pose_translation
from eval_utils.transform_utils import invert_poses


class NormalizePoseTranslationTest(unittest.TestCase):
    def _c2w_poses(self) -> torch.Tensor:
        poses = torch.eye(4).view(1, 1, 4, 4).repeat(2, 3, 1, 1)
        poses[0, :, :3, 3] = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 6.0, 0.0]]
        )
        poses[1, :, :3, 3] = torch.tensor(
            [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
        )
        return poses

    def test_c2w_uses_mean_camera_center_norm(self):
        poses = self._c2w_poses()
        original = poses.clone()

        normalized, scale = normalize_pose_translation(poses)

        torch.testing.assert_close(scale, torch.tensor([3.0, 2.0]))
        mean_norm = torch.linalg.vector_norm(normalized[..., :3, 3], dim=-1).mean(dim=1)
        torch.testing.assert_close(mean_norm, torch.ones(2))
        torch.testing.assert_close(normalized[..., :3, :3], original[..., :3, :3])
        torch.testing.assert_close(normalized[..., 3, :], original[..., 3, :])
        torch.testing.assert_close(poses, original)

    def test_w2c_and_3x4_preserve_shape_and_rotation(self):
        c2w = self._c2w_poses()[:1]
        w2c = invert_poses(c2w)[..., :3, :]

        normalized, scale = normalize_pose_translation(w2c, pose_convention="w2c")

        self.assertEqual(normalized.shape, w2c.shape)
        torch.testing.assert_close(scale, torch.tensor([3.0]))
        torch.testing.assert_close(normalized[..., :3, :3], w2c[..., :3, :3])
        normalized_centers = invert_poses(normalized)[..., :3, 3]
        mean_norm = torch.linalg.vector_norm(normalized_centers, dim=-1).mean(dim=1)
        torch.testing.assert_close(mean_norm, torch.ones(1))

    def test_degenerate_translation_uses_unit_scale(self):
        poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)

        normalized, scale = normalize_pose_translation(poses)

        torch.testing.assert_close(scale, torch.ones(1))
        torch.testing.assert_close(normalized, poses)


if __name__ == "__main__":
    unittest.main()
