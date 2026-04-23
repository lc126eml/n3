# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn.functional as F
import copy

from dataclasses import dataclass
from train_utils.general import check_and_fix_inf_nan # as _check_and_fix_inf_nan
from math import ceil, floor

LOSS_CHECKS_ENABLED = os.getenv("LOSS_CHECKS_ENABLED", "0") != "0"


# def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
#     if not LOSS_CHECKS_ENABLED:
#         return input_tensor
#     return _check_and_fix_inf_nan(input_tensor, loss_name=loss_name, hard_max=hard_max)


def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    del loss_name
    if input_tensor is None:
        return input_tensor
    if not LOSS_CHECKS_ENABLED:
        return input_tensor
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)
    return torch.nan_to_num(input_tensor, nan=0.0, posinf=hard_max or 0.0, neginf=-(hard_max or 0.0))


def _tpu_masked_filtered_mean(values, mask, valid_range, hard_max: float = 100.0):
    values = values.clamp(max=hard_max)
    del valid_range
    weights = mask.to(dtype=values.dtype)
    denom = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denom


def _masked_batch_mean(values, batch_mask):
    weights = batch_mask.to(dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0)
    values_per_batch = values.reshape(values.shape[0], -1).mean(dim=1)
    return (values_per_batch * batch_mask.to(dtype=values.dtype)).sum() / denom


def _tpu_standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def _tpu_sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=0.0))


def _tpu_mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _tpu_sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    floor = q_abs.new_tensor(0.1)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp_min(floor))
    gather_idx = q_abs.argmax(dim=-1)[..., None, None].expand(batch_dim + (1, 4))
    out = torch.gather(quat_candidates, dim=-2, index=gather_idx).squeeze(-2)
    out = out[..., [1, 2, 3, 0]]
    return _tpu_standardize_quaternion(out)


def intri_to_fov_encoding(intrinsics: torch.Tensor, image_size_hw) -> torch.Tensor:
    if image_size_hw is None:
        raise ValueError("image_size_hw must be provided to calculate field of view.")
    height, width = image_size_hw
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    fov_h = 2 * torch.atan((height / 2) / fy)
    fov_w = 2 * torch.atan((width / 2) / fx)
    return torch.stack([fov_h, fov_w], dim=-1).float()


def intri_to_logk_encoding(intrinsics: torch.Tensor, image_size_hw, eps: float = 1e-8) -> torch.Tensor:
    if image_size_hw is None:
        raise ValueError("image_size_hw must be provided to calculate logK.")
    height, width = image_size_hw
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    return torch.stack(
        [
            torch.log(torch.clamp(fx / width, min=eps)),
            torch.log(torch.clamp(fy / height, min=eps)),
            torch.log(torch.clamp(cx / width, min=eps)),
            torch.log(torch.clamp(cy / height, min=eps)),
        ],
        dim=-1,
    ).float()


def extri_to_pose_encoding(extrinsics, pose_encoding_type="absT_quaR"):
    if pose_encoding_type != "absT_quaR":
        raise NotImplementedError
    rot = extrinsics[:, :, :3, :3]
    trans = extrinsics[:, :, :3, 3]
    quat = _tpu_mat_to_quat(rot)
    return torch.cat([trans, quat], dim=-1).float()


def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw=None,
    pose_encoding_type="absT_quaR_FoV",
):
    rot = extrinsics[:, :, :3, :3]
    trans = extrinsics[:, :, :3, 3]
    quat = _tpu_mat_to_quat(rot)
    if pose_encoding_type == "absT_quaR_FoV":
        intri_enc = intri_to_fov_encoding(intrinsics, image_size_hw)
    elif pose_encoding_type == "absT_quaR_logK":
        intri_enc = intri_to_logk_encoding(intrinsics, image_size_hw)
    else:
        raise NotImplementedError
    return torch.cat([trans, quat, intri_enc], dim=-1).float()


class PoseEncodingLoss(torch.nn.Module):
    """Dense TPU-local pose encoding loss; avoids importing GPU camera_loss.py."""

    def __init__(self, loss_type: str = "l1", beta: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta
        if loss_type not in ("l1", "l2", "smooth_l1"):
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, pred_pose_enc: torch.Tensor, gt_pose_enc: torch.Tensor, max_val=None) -> torch.Tensor:
        zero = (0.0 * pred_pose_enc).sum()
        if pred_pose_enc.numel() == 0:
            return zero

        if self.loss_type == "l1":
            loss = (pred_pose_enc - gt_pose_enc).abs()
        elif self.loss_type == "l2":
            loss = torch.norm(pred_pose_enc - gt_pose_enc, p=2, dim=-1)
        else:
            loss = F.smooth_l1_loss(
                pred_pose_enc,
                gt_pose_enc,
                reduction="none",
                beta=self.beta,
            )

        loss = check_and_fix_inf_nan(loss, "loss_pose_enc")
        if max_val is not None:
            loss = loss.clamp(max=max_val)
        return loss.mean()


class _TPUCameraPoseLoss(torch.nn.Module):
    """
    TPU/XLA-friendly camera pose loss.

    This intentionally avoids boolean indexing such as rel[:, mask], which lowers
    through aten::nonzero and can create dynamic-shape graph fragments on XLA.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        compute_relative: bool = True,
        compute_absolute: bool = False,
        relative_neighbors: int = -1,
        loss_type: str = "l2",
        beta: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.compute_relative = compute_relative
        self.compute_absolute = compute_absolute
        self.relative_neighbors = relative_neighbors
        self.loss_type = loss_type
        self.beta = beta

    def forward(self, pred_poses, gt_poses, compute_relative=None, compute_absolute=None, relative_neighbors=None):
        if compute_relative is None:
            compute_relative = self.compute_relative
        if compute_absolute is None:
            compute_absolute = self.compute_absolute
        if relative_neighbors is None:
            relative_neighbors = self.relative_neighbors

        zero = (0.0 * pred_poses).mean()
        if pred_poses.numel() == 0 or (not compute_absolute and not compute_relative):
            return zero, zero, zero, zero

        pred_trans = pred_poses[..., :3, 3]
        gt_trans = gt_poses[..., :3, 3]
        pred_rot = pred_poses[..., :3, :3]
        gt_rot = gt_poses[..., :3, :3]

        abs_trans_err = zero
        abs_rot_err = zero
        if compute_absolute:
            abs_trans_err = self._trans_error(pred_trans, gt_trans).mean()
            abs_rot_err = self._geodesic_distance_from_matrices(pred_rot, gt_rot).mean()

        rel_trans_err = zero
        rel_rot_err = zero
        if compute_relative and pred_poses.shape[1] > 1:
            if relative_neighbors <= 0:
                rel_trans_err, rel_rot_err = self.compute_relative_pose_loss(
                    pred_trans, pred_rot, gt_trans, gt_rot
                )
            else:
                rel_trans_err, rel_rot_err = self.compute_relative_pose_loss_k1(
                    pred_trans, pred_rot, gt_trans, gt_rot
                )

        return abs_trans_err, abs_rot_err, rel_trans_err, rel_rot_err

    @staticmethod
    def _geodesic_distance_from_matrices(r1, r2):
        eps = 1e-6
        r_rel = torch.matmul(r1, r2.transpose(-1, -2))
        trace = r_rel.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        cos_theta = torch.clamp((trace - 1) / 2.0, -1.0 + eps, 1.0 - eps)
        return torch.acos(cos_theta)

    @staticmethod
    def _relative_pose_from_matrices(t1, r1, t2, r2):
        r1_inv = r1.transpose(-1, -2)
        t_rel = torch.matmul(r1_inv, (t2 - t1).unsqueeze(-1)).squeeze(-1)
        r_rel = torch.matmul(r1_inv, r2)
        return t_rel, r_rel

    def _trans_error(self, pred_trans, gt_trans):
        if self.loss_type == "l2":
            return torch.norm(pred_trans - gt_trans, dim=-1)
        if self.loss_type == "l1":
            return (pred_trans - gt_trans).abs().mean(dim=-1)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(
                pred_trans,
                gt_trans,
                reduction="none",
                beta=self.beta,
            ).mean(dim=-1)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def compute_relative_pose_loss(self, pred_trans, pred_rot, gt_trans, gt_rot):
        batch_size, num_views, _ = pred_trans.shape

        gt_trans1 = gt_trans.unsqueeze(2).expand(-1, -1, num_views, -1)
        gt_trans2 = gt_trans.unsqueeze(1).expand(-1, num_views, -1, -1)
        pred_trans1 = pred_trans.unsqueeze(2).expand(-1, -1, num_views, -1)
        pred_trans2 = pred_trans.unsqueeze(1).expand(-1, num_views, -1, -1)

        gt_rot1 = gt_rot.unsqueeze(2).expand(-1, -1, num_views, -1, -1)
        gt_rot2 = gt_rot.unsqueeze(1).expand(-1, num_views, -1, -1, -1)
        pred_rot1 = pred_rot.unsqueeze(2).expand(-1, -1, num_views, -1, -1)
        pred_rot2 = pred_rot.unsqueeze(1).expand(-1, num_views, -1, -1, -1)

        gt_t_rel, gt_r_rel = self._relative_pose_from_matrices(gt_trans1, gt_rot1, gt_trans2, gt_rot2)
        pred_t_rel, pred_r_rel = self._relative_pose_from_matrices(
            pred_trans1, pred_rot1, pred_trans2, pred_rot2
        )

        rel_trans_err = self._trans_error(pred_t_rel, gt_t_rel)
        rel_rot_err_rad = self._geodesic_distance_from_matrices(pred_r_rel, gt_r_rel)

        pair_weights = torch.triu(
            torch.ones(num_views, num_views, device=pred_trans.device, dtype=rel_trans_err.dtype),
            diagonal=1,
        )
        pair_weights = pair_weights.unsqueeze(0)
        denom = (pair_weights.sum() * rel_trans_err.new_tensor(batch_size)).clamp_min(1.0)
        rel_trans_err_mean = (rel_trans_err * pair_weights).sum() / denom
        rel_rot_err_mean = (rel_rot_err_rad * pair_weights).sum() / denom

        return rel_trans_err_mean, rel_rot_err_mean

    def compute_relative_pose_loss_k1(self, pred_trans, pred_rot, gt_trans, gt_rot):
        pred_trans2 = torch.roll(pred_trans, shifts=-1, dims=1)
        gt_trans2 = torch.roll(gt_trans, shifts=-1, dims=1)
        pred_rot2 = torch.roll(pred_rot, shifts=-1, dims=1)
        gt_rot2 = torch.roll(gt_rot, shifts=-1, dims=1)

        gt_t_rel, gt_r_rel = self._relative_pose_from_matrices(gt_trans, gt_rot, gt_trans2, gt_rot2)
        pred_t_rel, pred_r_rel = self._relative_pose_from_matrices(
            pred_trans, pred_rot, pred_trans2, pred_rot2
        )

        rel_trans_err = self._trans_error(pred_t_rel, gt_t_rel)
        rel_rot_err_rad = self._geodesic_distance_from_matrices(pred_r_rel, gt_r_rel)

        return rel_trans_err.mean(), rel_rot_err_rad.mean()


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(self, camera=None, angle_pose=None, depth=None, point=None, track=None, switch=None, vggt=True, regulize_scale=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.angle_pose = angle_pose
        self.depth = depth
        self.point = point
        self.track = track
        self.switch = switch
        self.vggt = vggt
        self.regulize_scale = regulize_scale

        self.pose_enc_loss = PoseEncodingLoss(loss_type="l1")

        if angle_pose is not None: # and (angle_pose.get("compute_absolute") or angle_pose.get("compute_relative")):
            self.pose_loss = _TPUCameraPoseLoss(
                alpha=angle_pose.get("alpha", 1.0),
                compute_relative=angle_pose.get("compute_relative", False),
                compute_absolute=angle_pose.get("compute_absolute", False),
                relative_neighbors=angle_pose.get("relative_neighbors", -1),
                loss_type=angle_pose.get("loss_type", "l2"),
                beta=angle_pose.get("beta", 1.0),
            )
            self.angle_pose = angle_pose
        else:
            self.angle_pose = None
            self.pose_loss = None      


        if self.point is not None:
            self.point_no_conf_percent = {k: v for k, v in self.point.items() if k != "conf_percentage"}
            self.aligned_point = copy.deepcopy(self.point)
            self.aligned_point.valid_range = self.aligned_point.aligned_valid_range
        else:
            self.point_no_conf_percent = None
            self.aligned_point = None

    def forward(self, predictions, batch, data_keys, pred_data_keys) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        first_prediction = next(iter(predictions.values()), None)
        if torch.is_tensor(first_prediction):
            total_loss = (0.0 * first_prediction).mean()
        else:
            total_loss = torch.zeros((), device=batch["img"].device, dtype=batch["img"].dtype)
        loss_dict = {}
        
        # Camera pose loss - if pose encodings are predicted
        if "pose_enc" in predictions:
            if self.vggt:
                camera_loss_dict = compute_camera_loss_one(predictions, batch, **self.camera)  
                camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]   
                total_loss = total_loss + camera_loss
                loss_dict.update(camera_loss_dict)
            else:
                gt_poses = batch[data_keys.extrinsics]
                pred_enc = predictions['pose_enc']
                image_hw = batch["img"].shape[-2:]
                pose_aligned = predictions.get('pose_aligned', False)
                pose_trans_aligned = predictions.get('pose_trans_aligned', False)
                weight_trans = self.camera.get("weight_trans")
                weight_rot = self.camera.get("weight_rot")
                if weight_trans > 0 or weight_rot > 0:
                    pose_encoding = extri_to_pose_encoding(batch[data_keys.extrinsics])
                    loss_T = None
                    loss_R = None
                    if weight_trans > 0 and (not pose_trans_aligned):
                        loss_T = self.pose_enc_loss(pred_enc[..., :3], pose_encoding[..., :3], max_val=100)
                        loss_dict["loss_T"] = loss_T
                        total_loss = total_loss + weight_trans * loss_T
                    if (not pose_aligned) and weight_rot > 0:
                        loss_R = self.pose_enc_loss(pred_enc[..., 3:7], pose_encoding[..., 3:7])
                        loss_dict["loss_R"] = loss_R
                        total_loss = total_loss + weight_rot * loss_R
                
                weight_focal = self.camera.get("weight_focal")
                if weight_focal > 0:
                    pose_encoding_type = self.camera.get("pose_encoding_type", "absT_quaR_FoV")
                    if pose_encoding_type == "absT_quaR_logK":
                        gt_enc = intri_to_logk_encoding(
                            intrinsics=batch[data_keys.intrinsics], image_size_hw=image_hw
                        )
                    else:
                        gt_enc = intri_to_fov_encoding(
                            intrinsics=batch[data_keys.intrinsics], image_size_hw=image_hw
                        )
                    loss_FL = self.pose_enc_loss(pred_pose_enc=pred_enc[..., 7:], gt_pose_enc=gt_enc)
                    loss_dict["loss_FL"] = loss_FL
                    total_loss = total_loss + weight_focal * loss_FL
                
                if self.pose_loss is not None and (self.pose_loss.compute_absolute or self.pose_loss.compute_relative):                    
                    pred_poses = predictions[pred_data_keys.extrinsics]
                    relative_weight = self.angle_pose.get("relative_weight", 0.8)
                    absolute_weight = self.angle_pose.get("absolute_weight", 0.2)
                    weight_trans = self.angle_pose.get("weight_trans")
                    weight_rot = self.angle_pose.get("weight_rot")

                    abs_trans_err, abs_rot_err, rel_trans_err, rel_rot_err = self.pose_loss(pred_poses, gt_poses)
                    
                    
                    trans_err = relative_weight * rel_trans_err + absolute_weight * abs_trans_err
                    rot_err = relative_weight * rel_rot_err + absolute_weight * abs_rot_err
                    pose_loss = trans_err * weight_trans + rot_err * weight_rot

                    loss_dict["loss_Pose_T"] = abs_trans_err
                    loss_dict["loss_Pose_R"] = abs_rot_err
                    loss_dict["loss_Rel_T"] = rel_trans_err
                    loss_dict["loss_Rel_R"] = rel_rot_err
                    total_loss = total_loss + pose_loss * self.angle_pose.get("weight", 1.0)
                    
                loss_dict["loss_camera"] = total_loss
                    
        # Depth estimation loss - if depth maps are predicted
        if "depth" in predictions:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            conf_weight = self.depth.get("conf_weight", 1.0)
            reg_weight = self.depth.get("reg_weight", 1.0)
            grad_weight = self.depth.get("grad_weight", 1.0)
            silog_weight = self.depth.get("silog_weight", 0.0)
            depth_loss = (
                conf_weight * depth_loss_dict["loss_conf_depth"]
                + reg_weight * depth_loss_dict["loss_reg_depth"]
                + grad_weight * depth_loss_dict["loss_grad_depth"]
            )
            if "loss_silog_depth" in depth_loss_dict:
                depth_loss = depth_loss + silog_weight * depth_loss_dict["loss_silog_depth"]
            if "loss_silog_conf_depth" in depth_loss_dict:
                silog_conf_weight = self.depth.get("silog_conf_weight", 0.0)
                depth_loss = depth_loss + silog_conf_weight * depth_loss_dict["loss_silog_conf_depth"]
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)

        # 3D point reconstruction loss - if world points are predicted
        if self.point is not None:
            # Create a version of the point config without 'conf_percentage' for specific loss calculations.
            point_key = pred_data_keys.world_points
            weight = self.point["weight"]
            if self.switch is not None:
                if self.switch.get("pts_align_to_gt") or self.switch.get("pts_align_to_gt_rot"):
                    point_key = pred_data_keys.aligned_world_points
                    weight = self.point["global_aligned_weight"]
                elif self.switch.get("pts_align_to_center"):
                    point_key = pred_data_keys.get("global_aligned_to_center", "global_aligned_to_center")
                    weight = self.point["global_aligned_weight"]                
            
            if point_key in predictions:
                point_loss_dict, point_loss = compute_point_loss(predictions, batch, pts3d_name=point_key, **self.point_no_conf_percent)
                total_loss = total_loss + point_loss * weight
                loss_dict.update(point_loss_dict)
            # if pred_data_keys.pts3d_cam in predictions:
            #     loss_name = pred_data_keys.pts3d_cam
            #     conf_name = pred_data_keys.cam_points_conf
            #     point_loss_dict = compute_point_loss(predictions, batch, pts3d_name=loss_name, gt_pts_name=pred_data_keys.pts3d_cam,  pts3d_conf_name=conf_name, **self.point_no_conf_percent)
            #     point_loss = point_loss_dict[f"{loss_name}_loss_conf_point"] + point_loss_dict[f"{loss_name}_loss_reg_point"] + point_loss_dict[f"{loss_name}_loss_grad_point"]
            #     point_loss = point_loss * self.point["weight"]
            #     total_loss = total_loss + point_loss
            #     loss_dict.update(point_loss_dict)

            # if pred_data_keys.cam_from_depth in predictions:
            #     loss_name = pred_data_keys.cam_from_depth
            #     conf_name = pred_data_keys.depth_conf
            #     point_loss_dict = compute_point_loss(predictions, batch, pts3d_name=loss_name, gt_pts_name=pred_data_keys.pts3d_cam, pts3d_conf_name=conf_name, **self.aligned_point)
            #     point_loss = point_loss_dict[f"{loss_name}_loss_conf_point"] + point_loss_dict[f"{loss_name}_loss_reg_point"] + point_loss_dict[f"{loss_name}_loss_grad_point"]
            #     point_loss = point_loss * self.point["cam_from_depth_weight"]
            #     total_loss = total_loss + point_loss
            #     loss_dict.update(point_loss_dict)
            # if self.pts_align_to_gt:
            #     point_key = pred_data_keys.aligned_global_from_cam
            # else:
            #     point_key = pred_data_keys.global_from_cam
            # if point_key in predictions:
            #     loss_name = pred_data_keys.global_from_cam
            #     conf_name = pred_data_keys.cam_points_conf
            #     point_loss_dict = compute_point_loss(predictions, batch, pts3d_name=loss_name, gt_pts_name=pred_data_keys.world_points, pts3d_conf_name=conf_name, **self.aligned_point)
            #     point_loss = point_loss_dict[f"{loss_name}_loss_conf_point"] + point_loss_dict[f"{loss_name}_loss_reg_point"] + point_loss_dict[f"{loss_name}_loss_grad_point"]
            #     point_loss = point_loss * self.point["global_from_cam_weight"] 
            #     total_loss = total_loss + point_loss
            #     loss_dict.update(point_loss_dict)

            # if pred_data_keys.global_from_cam_detach_pose in predictions:
            #     loss_name = pred_data_keys.global_from_cam_detach_pose
            #     conf_name = pred_data_keys.cam_points_conf
            #     point_loss_dict = compute_point_loss(predictions, batch, pts3d_name=loss_name, gt_pts_name=pred_data_keys.world_points, pts3d_conf_name=conf_name, **self.aligned_point)
            #     point_loss = point_loss_dict[f"{loss_name}_loss_conf_point"] + point_loss_dict[f"{loss_name}_loss_reg_point"] + point_loss_dict[f"{loss_name}_loss_grad_point"]
            #     point_loss = point_loss * self.point["global_from_cam_weight"] 
            #     total_loss = total_loss + point_loss
            #     loss_dict.update(point_loss_dict)

            # if self.pts_align_to_gt:
            #     point_key = pred_data_keys.aligned_global_from_depth
            # else:
            #     point_key = pred_data_keys.global_from_depth
            # if point_key in predictions:
            #     loss_name = pred_data_keys.global_from_depth
            #     conf_name = pred_data_keys.depth_conf
            #     point_loss_dict = compute_point_loss(predictions, batch, pts3d_name=loss_name, gt_pts_name=pred_data_keys.world_points, pts3d_conf_name=conf_name, **self.aligned_point)
            #     point_loss = point_loss_dict[f"{loss_name}_loss_conf_point"] + point_loss_dict[f"{loss_name}_loss_reg_point"] + point_loss_dict[f"{loss_name}_loss_grad_point"]
            #     point_loss = point_loss * self.point["global_from_depth_weight"]
            #     total_loss = total_loss + point_loss
            #     loss_dict.update(point_loss_dict)

        if self.regulize_scale is not None and self.regulize_scale.get("enabled"):
            scale = predictions.get("scale", None)
            translation = None #predictions.get("translation", None)
            loss_scale_reg, loss_trans_reg = regulize_scale_loss(
                translation,
                scale,
                reference=total_loss,
            )
            total_loss = total_loss + self.regulize_scale.get("scale_weight") * loss_scale_reg
            # + self.regulize_scale.get("translation_weight") * loss_trans_reg
            loss_dict["loss_scale"] = loss_scale_reg
            loss_dict["loss_trans"] = loss_trans_reg

        # Tracking loss - not cleaned yet, dirty code is at the bottom of this file
        if self.track is not None and "track" in predictions:
            raise NotImplementedError("Track loss is not cleaned up yet")
        
        loss_dict["objective"] = total_loss

        return loss_dict

def regulize_scale_loss(translation=None, scale=None, reference=None):
    if reference is not None:
        zero = reference * 0.0
    elif scale is not None:
        zero = (0.0 * scale).mean()
    elif translation is not None:
        zero = (0.0 * translation).mean()
    else:
        raise ValueError("regulize_scale_loss requires reference, scale, or translation for device-safe zero fallback")

    if translation is not None:
        target_trans = torch.zeros_like(translation)
        loss_trans_reg = torch.nn.functional.mse_loss(translation, target_trans)
    else:
        loss_trans_reg = zero
    
    # 3. Compute Loss directly on parameters
    if scale is not None:
        # log_val = torch.log(scale + 1e-12)
        # target_scale = torch.zeros_like(log_val)
        # loss_scale_reg = torch.nn.functional.mse_loss(log_val, target_scale)        
        target_scale = torch.ones_like(scale)
        loss_scale_reg = torch.nn.functional.mse_loss(scale, target_scale)
    else:
        loss_scale_reg = zero
    return loss_scale_reg, loss_trans_reg
def compute_camera_loss_one(
    pred_dict: dict,               # Predictions dict, contains the pose encoding tensor
    batch_data: dict,              # Ground truth and mask batch dict
    loss_type: str = "l1",         # "l1" or "l2" loss
    pose_encoding_type: str = "absT_quaR_FoV",
    weight_trans: float = 1.0,     # Weight for translation loss
    weight_rot: float = 1.0,       # Weight for rotation loss
    weight_focal: float = 0.5,     # Weight for focal length loss
    beta: float = 1.0,
    **kwargs
) -> dict:
    """
    Computes the camera pose loss for a single prediction stage.

    Args:
        pred_dict (dict): Dictionary containing the predicted pose encoding tensor
                          under the key 'pose_enc'.
        batch_data (dict): Dictionary with ground truth data, including 'camera_pose',
                           'camera_intrinsics', and 'valid_mask'.
        loss_type (str): The type of loss to use ('l1' or 'l2').
        pose_encoding_type (str): The format of the pose encoding.
        weight_trans (float): The weight for the translation component of the loss.
        weight_rot (float): The weight for the rotation component of the loss.
        weight_focal (float): The weight for the focal length component of the loss.

    Returns:
        dict: A dictionary containing the total camera loss and its individual components.
    """
    # Get the single predicted pose encoding tensor
    pred_pose_encoding = pred_dict['pose_enc']

    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['valid_mask']
    # A frame is considered valid if it has more than 100 valid points
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data["camera_pose"]
    gt_intrinsics = batch_data["camera_intrinsics"]
    image_hw = batch_data["img"].shape[-2:]

    # Encode ground truth pose to match the predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Compute loss on dense tensors and reduce by mask. Boolean gathers create
    # dynamic shapes that are expensive for XLA.
    if loss_type == "l1":
        loss_T_map = (pred_pose_encoding[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R_map = (pred_pose_encoding[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        loss_FL_map = (pred_pose_encoding[..., 7:] - gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T_map = (pred_pose_encoding[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
        loss_R_map = (pred_pose_encoding[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_FL_map = (pred_pose_encoding[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
    elif loss_type == "smooth_l1":
        loss_T_map = F.smooth_l1_loss(
            pred_pose_encoding[..., :3],
            gt_pose_encoding[..., :3],
            reduction="none",
            beta=beta,
        ).mean(dim=-1, keepdim=True)
        loss_R_map = F.smooth_l1_loss(
            pred_pose_encoding[..., 3:7],
            gt_pose_encoding[..., 3:7],
            reduction="none",
            beta=beta,
        ).mean(dim=-1)
        loss_FL_map = F.smooth_l1_loss(
            pred_pose_encoding[..., 7:],
            gt_pose_encoding[..., 7:],
            reduction="none",
            beta=beta,
        ).mean(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = _masked_batch_mean(check_and_fix_inf_nan(loss_T_map, "loss_T").clamp(max=100), valid_frame_mask)
    loss_R = _masked_batch_mean(check_and_fix_inf_nan(loss_R_map, "loss_R"), valid_frame_mask)
    loss_FL = _masked_batch_mean(check_and_fix_inf_nan(loss_FL_map, "loss_FL"), valid_frame_mask)

    # Compute the total weighted camera loss
    total_camera_loss = (
        loss_T * weight_trans +
        loss_R * weight_rot +
        loss_FL * weight_focal
    )

    # Return a dictionary with the total loss and its components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": loss_T,
        "loss_R": loss_R,
        "loss_FL": loss_FL,
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1", beta: float = 1.0):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    elif loss_type == "smooth_l1":
        loss_T = F.smooth_l1_loss(
            pred_pose_enc[..., :3],
            gt_pose_enc[..., :3],
            reduction="none",
            beta=beta,
        ).mean(dim=-1, keepdim=True)
        loss_R = F.smooth_l1_loss(
            pred_pose_enc[..., 3:7],
            gt_pose_enc[..., 3:7],
            reduction="none",
            beta=beta,
        ).mean(dim=-1)
        loss_FL = F.smooth_l1_loss(
            pred_pose_enc[..., 7:],
            gt_pose_enc[..., 7:],
            reduction="none",
            beta=beta,
        ).mean(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def compute_point_loss(
    predictions,
    batch,
    gamma=1.0,
    alpha=0.2,
    gradient_loss_fn=None,
    valid_range=-1,
    reg_loss_type="l2",
    reg_loss_beta: float = 1.0,
    conf_weight=1.0,
    reg_weight=1.0,
    grad_weight=1.0,
    pts3d_name="pts3d",
    gt_pts_name="pts3d",
    pts3d_conf_name="world_points_conf",
    conf_percentage=None,
    **kwargs,
):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing "pts3d" and 'world_points_conf'
        batch: Dict containing ground truth "pts3d" and 'valid_mask'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    prefix = ""
    # if pts3d_name != "pts3d":
    #     prefix = f"{pts3d_name}_"
    pred_points = check_and_fix_inf_nan(predictions[pts3d_name], loss_name=pts3d_name)
    pred_points_conf = predictions[pts3d_conf_name]
    gt_points = batch[gt_pts_name]
    gt_points_mask = batch['valid_mask']
    
    gt_points = check_and_fix_inf_nan(gt_points, loss_name=gt_pts_name)
    
    valid_enough = (gt_points_mask.to(dtype=pred_points.dtype).sum() >= 100).to(dtype=pred_points.dtype)
    
    need_conf_reg = (conf_weight > 0) or (reg_weight > 0)
    need_grad = (grad_weight > 0) and bool(gradient_loss_fn)
    if need_conf_reg or need_grad:
        # Compute confidence-weighted regression loss with optional gradient loss
        loss_conf, loss_grad, loss_reg = regression_loss(
            pred_points,
            gt_points,
            gt_points_mask,
            conf=pred_points_conf,
            gradient_loss_fn=gradient_loss_fn if need_grad else None,
            gamma=gamma,
            alpha=alpha,
            valid_range=valid_range,
            conf_percentage=conf_percentage,
            reg_loss_type=reg_loss_type,
            reg_loss_beta=reg_loss_beta,
            compute_conf_reg=need_conf_reg,
        )
    else:
        dummy_loss = (0.0 * pred_points).mean()
        loss_conf, loss_grad, loss_reg = dummy_loss, dummy_loss, dummy_loss

    loss_conf = loss_conf * valid_enough
    loss_grad = loss_grad * valid_enough
    loss_reg = loss_reg * valid_enough
    
    loss_dict = {
        f"{prefix}loss_conf_point": loss_conf,
        f"{prefix}loss_reg_point": loss_reg,
        f"{prefix}loss_grad_point": loss_grad,
    }
    point_loss = conf_weight * loss_conf + reg_weight * loss_reg + grad_weight * loss_grad
    return loss_dict, point_loss


def compute_depth_loss(
    predictions,
    batch,
    gamma=1.0,
    alpha=0.2,
    gradient_loss_fn=None,
    valid_range=-1,
    reg_loss_type="l2",
    reg_loss_beta: float = 1.0,
    scale_invariant=False,
    scale_invariant_mode="scale",
    conf_weight=1.0,
    reg_weight=1.0,
    grad_weight=1.0,
    silog_weight=0.0,
    silog_variance_focus=0.85,
    **kwargs,
):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth "depthmap" and 'valid_mask'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth']
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch["depthmap"]
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    if gt_depth.shape[-1] != 1:
        gt_depth = gt_depth[..., None]              # (B, S, H, W, 1)
    gt_depth_mask = batch['valid_mask']   # 3D points derived from depth map, so we use the same mask

    valid_enough = (gt_depth_mask.to(dtype=pred_depth.dtype).sum() >= 100).to(dtype=pred_depth.dtype)

    if scale_invariant:
        pred_depth = align_depth_scale(
            pred_depth,
            gt_depth,
            gt_depth_mask,
            mode=scale_invariant_mode,
        )

    need_conf_reg = (conf_weight > 0) or (reg_weight > 0)
    need_grad = (grad_weight > 0) and bool(gradient_loss_fn)
    if need_conf_reg or need_grad:
        # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
        # this is hacky, but very easier to implement
        loss_conf, loss_grad, loss_reg = regression_loss(
            pred_depth,
            gt_depth,
            gt_depth_mask,
            conf=pred_depth_conf,
            gradient_loss_fn=gradient_loss_fn if need_grad else None,
            gamma=gamma,
            alpha=alpha,
            valid_range=valid_range,
            reg_loss_type=reg_loss_type,
            reg_loss_beta=reg_loss_beta,
            compute_conf_reg=need_conf_reg,
        )
    else:
        dummy_loss = (0.0 * pred_depth).mean()
        loss_conf, loss_grad, loss_reg = dummy_loss, dummy_loss, dummy_loss

    loss_conf = loss_conf * valid_enough
    loss_grad = loss_grad * valid_enough
    loss_reg = loss_reg * valid_enough

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,    
        f"loss_grad_depth": loss_grad,
    }

    if silog_weight > 0:
        loss_silog, loss_silog_conf = silog_loss(
            pred_depth,
            gt_depth,
            gt_depth_mask,
            variance_focus=silog_variance_focus,
            conf=pred_depth_conf,
            gamma=gamma,
            alpha=alpha,
        )
        loss_dict["loss_silog_depth"] = loss_silog
        if loss_silog_conf is not None:
            loss_dict["loss_silog_conf_depth"] = loss_silog_conf

    return loss_dict


def align_depth_scale(pred, gt, mask, mode="scale_shift", eps=1e-6):
    """
    Align predicted depth to GT with a per-sample scale (and optional shift).

    Args:
        pred: (B, S, H, W, 1)
        gt: (B, S, H, W, 1)
        mask: (B, S, H, W)
        mode: "scale" or "scale_shift"
    """
    if mode not in {"scale", "scale_shift"}:
        raise ValueError(f"Unsupported scale_invariant_mode: {mode}")

    pred_vals = pred[..., 0]
    gt_vals = gt[..., 0]
    m = mask.float()

    num = m.sum(dim=(2, 3), keepdim=True)
    sum_p2 = (m * pred_vals * pred_vals).sum(dim=(2, 3), keepdim=True)
    sum_p = (m * pred_vals).sum(dim=(2, 3), keepdim=True)
    sum_g = (m * gt_vals).sum(dim=(2, 3), keepdim=True)
    sum_pg = (m * pred_vals * gt_vals).sum(dim=(2, 3), keepdim=True)

    if mode == "scale":
        scale = sum_pg / (sum_p2 + eps)
        shift = torch.zeros_like(scale)
    else:
        det = sum_p2 * num - sum_p * sum_p
        safe = det.abs() > eps
        scale = (sum_pg * num - sum_p * sum_g) / (det + eps)
        shift = (sum_p2 * sum_g - sum_p * sum_pg) / (det + eps)
        scale = torch.where(safe, scale, sum_pg / (sum_p2 + eps))
        shift = torch.where(safe, shift, torch.zeros_like(shift))

    pred_aligned = scale * pred_vals + shift
    return pred_aligned[..., None]


def silog_loss(pred, gt, mask, variance_focus=0.85, eps=1e-6, conf=None, gamma=1.0, alpha=0.2):
    """
    Scale-invariant log loss (SiLog).

    Args:
        pred: (B, S, H, W, 1)
        gt: (B, S, H, W, 1)
        mask: (B, S, H, W)
        variance_focus: weight on variance term (lambda in SiLog)
    """
    pred_vals = pred[..., 0].clamp_min(eps)
    gt_vals = gt[..., 0].clamp_min(eps)
    valid = (mask > 0).to(dtype=pred_vals.dtype)
    log_diff = torch.log(pred_vals) - torch.log(gt_vals)
    loss_conf = None
    denom = valid.sum().clamp_min(1.0)
    mean = (log_diff * valid).sum() / denom
    var = ((log_diff - mean) * (log_diff - mean) * valid).sum() / denom
    if conf is not None:
        safe_conf = conf.clamp_min(eps)
        loss_conf_map = gamma * log_diff.abs() * safe_conf - alpha * torch.log(safe_conf)
        loss_conf = (loss_conf_map * valid).sum() / denom
    silog = torch.sqrt(torch.clamp(var + variance_focus * mean * mean, min=0.0))
    return silog, loss_conf


def regression_loss(
    pred,
    gt,
    mask,
    conf=None,
    gradient_loss_fn=None,
    gamma=1.0,
    alpha=0.2,
    valid_range=-1,
    conf_percentage=None,
    reg_loss_type="l2",
    reg_loss_beta: float = 1.0,
    compute_conf_reg=True,
):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # XLA/TPU is sensitive to boolean gathers and quantile/kthvalue in the loss
    # graph. Keep this path dense and fixed-shape, then reduce with masks.
    if conf_percentage is not None and not (0 < conf_percentage <= 100):
        raise ValueError("conf_percentage must be between 0 and 100.")
    combined_mask = mask.bool()

    if compute_conf_reg:
        # Compute L1/L2/SmoothL1 distance between predicted and ground truth points
        if reg_loss_type == "l2":
            diff = gt - pred
            loss_reg_map = torch.linalg.vector_norm(diff, dim=-1)
        elif reg_loss_type == "l1":
            diff = gt - pred
            loss_reg_map = diff.abs().mean(dim=-1)
        elif reg_loss_type == "smooth_l1":
            loss_reg_map = F.smooth_l1_loss(
                pred,
                gt,
                reduction="none",
                beta=reg_loss_beta,
            ).mean(dim=-1)
        else:
            raise ValueError(f"Unsupported reg_loss_type: {reg_loss_type}")
        # NaNs unlikely for loss_reg; keep guard for safety.
        loss_reg_map = check_and_fix_inf_nan(loss_reg_map, "loss_reg")

        # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
        # This encourages the model to be confident on easy examples and less confident on hard ones
        safe_conf = conf.clamp_min(1e-6) if conf is not None else torch.ones_like(loss_reg_map)
        loss_conf_map = gamma * loss_reg_map * safe_conf - alpha * torch.log(safe_conf)
        # conf_activation="expp1" => conf >= 1, so log(conf) is safe; keep guard anyway.
        loss_conf_map = check_and_fix_inf_nan(loss_conf_map, "loss_conf")
    else:
        loss_conf_map = None
        loss_reg_map = None
        
    # Initialize gradient loss
    loss_grad = (0.0 * pred).mean()

    # Prepare confidence for gradient loss if needed
    if gradient_loss_fn and ("conf" in gradient_loss_fn):
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if gradient_loss_fn and ("normal" in gradient_loss_fn):
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            combined_mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif gradient_loss_fn and ("grad" in gradient_loss_fn):
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            combined_mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    if compute_conf_reg:
        loss_conf = _tpu_masked_filtered_mean(loss_conf_map, combined_mask, valid_range)
        loss_reg = _tpu_masked_filtered_mean(loss_reg_map, combined_mask, valid_range)
    else:
        loss_conf = (0.0 * pred).mean()
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = prediction.new_zeros(())
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot
    loss = check_and_fix_inf_nan(loss, "normal_loss")

    valid = all_valid.to(dtype=loss.dtype)
    if conf is not None:
        conf = conf[None, ...].expand_as(loss).clamp_min(1e-6)
        loss = gamma * loss * conf - alpha * torch.log(conf)

    denom = valid.sum().clamp_min(1.0)
    return (loss * valid).sum() / denom


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1]).clamp_min(1e-6)
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = (gamma * grad_x * conf_x - alpha * torch.log(conf_x)) * mask_x
        grad_y = (gamma * grad_y * conf_y - alpha * torch.log(conf_y)) * mask_y

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    return torch.sum(grad_loss) / divisor.clamp_min(1.0)


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    # Pad inputs to avoid boundary issues
    padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
    pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

    # Get neighboring points for each pixel
    center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
    up     = pts[:, :-2,  1:-1, :]
    left   = pts[:, 1:-1, :-2 , :]
    down   = pts[:, 2:,   1:-1, :]
    right  = pts[:, 1:-1, 2:,   :]

    # Compute direction vectors from center to neighbors
    up_dir    = up    - center
    left_dir  = left  - center
    down_dir  = down  - center
    right_dir = right - center

    # Compute four cross products for different normal directions
    n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
    n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
    n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
    n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

    # Validity masks - require both direction pixels to be valid
    v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
    v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
    v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
    v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

    # Stack normals and validity masks
    normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
    valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

    # Normalize normal vectors
    normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids
