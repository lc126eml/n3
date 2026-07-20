import unittest

import torch

from eval_utils.align_utils.depth_median_scaling import median_scale_depth_torch_batch
from eval_utils.normalize_utils.normalize_pc import calculate_depth_scale


class DepthScalingModeTest(unittest.TestCase):
    def test_gt_depth_scale_supports_mean_and_median(self):
        depth = torch.tensor([[[[1.0, 1.0, 10.0]]]])
        valid = torch.ones_like(depth, dtype=torch.bool)

        mean_scale = calculate_depth_scale(depth, valid, mode="mean")
        median_scale = calculate_depth_scale(depth, valid, mode="median")

        torch.testing.assert_close(mean_scale, torch.tensor([4.0]))
        torch.testing.assert_close(median_scale, torch.tensor([1.0]))

    def test_depth_alignment_supports_mean_and_median(self):
        pred = torch.tensor([[[[[1.0], [1.0], [10.0]]]]])
        gt = torch.tensor([[[[2.0, 2.0, 2.0]]]])
        valid = torch.ones_like(gt, dtype=torch.bool)

        _, mean_scale, mean_valid = median_scale_depth_torch_batch(
            pred, gt, valid_mask=valid, mode="mean"
        )
        _, median_scale, median_valid = median_scale_depth_torch_batch(
            pred, gt, valid_mask=valid, mode="median"
        )

        torch.testing.assert_close(mean_scale, torch.tensor([0.5]))
        torch.testing.assert_close(median_scale, torch.tensor([2.0]))
        self.assertTrue(mean_valid.item())
        self.assertTrue(median_valid.item())

    def test_invalid_mode_is_rejected(self):
        depth = torch.ones(1, 1, 1, 1)
        with self.assertRaises(ValueError):
            calculate_depth_scale(depth, torch.ones_like(depth), mode="maximum")


if __name__ == "__main__":
    unittest.main()
