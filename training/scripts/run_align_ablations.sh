#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  experiment/depth_loss_ablation_first_frame
  # experiment/align_ablation_first_frame_share_dpt
  # experiment/align_ablation_first_frame
  # experiment/align_ablation_center_world
  # experiment/align_ablation_pred_center
  # experiment/align_ablation_gt_align_to_pts
  # experiment/align_ablation_pts_align_to_gt
  # experiment/align_ablation_pts_align_to_gt_rot
)

for cfg in "${CONFIGS[@]}"; do
  echo "=== Running ${cfg} ==="
  python launch.py --config_name "${cfg}"
done
