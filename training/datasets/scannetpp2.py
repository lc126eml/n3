import os.path as osp
import os
import numpy as np
import cv2
from collections import defaultdict
from training.datasets.base.base_multiview_dataset import BaseMultiViewDataset, cropping
from dust3r.utils.image import imread_cv2

class ScanNetpp_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, samples_per_scene=10, max_interval=9, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = max_interval
        self.samples_per_scene = samples_per_scene
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data()

    def _load_data(self):
        self.all_scenes = sorted(
            [f for f in os.listdir(self.ROOT) if os.path.isdir(osp.join(self.ROOT, f))]
        )
        scenes = []
        images = []
        scene_img_list = []
        offset = 0

        for scene_idx, scene in enumerate(self.all_scenes):
            scene_dir = osp.join(self.ROOT, scene)
            rgb_paths = sorted([f for f in os.listdir(scene_dir) if f.endswith(".jpg")])
            if not rgb_paths:
                print(f"Skipping {scene_dir}: No .jpg files found.")
                continue

            num_imgs = len(rgb_paths)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue

            img_ids = list(np.arange(num_imgs) + offset)
            scenes.append(scene)
            scene_img_list.append(img_ids)
            images.extend(rgb_paths)
            offset += num_imgs

        self.scenes = scenes
        self.images = images
        self.scene_img_list = scene_img_list

    def __len__(self):
        return len(self.scenes) * self.samples_per_scene

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, scene_id, resolution, rng, num_views):
        scene_id = scene_id // self.samples_per_scene
        all_image_ids = self.scene_img_list[scene_id]

        pos, ordered_video = self.efficient_random_intervals_revised(
            0,
            len(all_image_ids),
            num_views,
            rng,
            min_interval=1,
            max_interval=self.max_interval,
        )
        image_idxs = np.array(all_image_ids)[pos]
        # 1. Initialize a dictionary of lists to collect batch data
        batched_views = defaultdict(list)

        for v, view_idx in enumerate(image_idxs):
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            rgb_path = self.images[view_idx]
            depth_path = rgb_path.replace("rgb.jpg", "depth.png")
            cam_path = rgb_path.replace("rgb.jpg", "cam.npz")

            rgb_image = imread_cv2(osp.join(scene_dir, rgb_path), cv2.IMREAD_COLOR)
            depthmap = imread_cv2(osp.join(scene_dir, depth_path), cv2.IMREAD_UNCHANGED)
            depthmap[depthmap == 65535] = 0  # Handle invalid depth values
            depthmap = depthmap.astype(np.float32)/1000.0
            cam_file = np.load(osp.join(scene_dir, cam_path))
            intrinsics = cam_file["intrinsics"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

            if self.split == "train" and rng.random() < self.random_crop_prob:
                rgb_image, depthmap, intrinsics, camera_pose = self._crop_resize(
                    rgb_image,
                    depthmap,
                    intrinsics,
                    resolution,
                    rng=rng,
                    prot=self.prot,
                    pcrop=self.pcrop,
                    pose=camera_pose,
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )

            # Generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            intrinsics, camera_pose = cropping.get_center_camera(intrinsics, camera_pose, depthmap=depthmap)
            # 2. Append each piece of data to its corresponding list
            batched_views['img'].append(rgb_image)
            batched_views['depthmap'].append(depthmap.astype(np.float32))
            batched_views['camera_pose'].append(camera_pose.astype(np.float32))
            batched_views['camera_intrinsics'].append(intrinsics.astype(np.float32))
            batched_views['dataset'].append("scannetpp")
            batched_views['label'].append(self.scenes[scene_id] + "_" + rgb_path)
            batched_views['instance'].append(f"{str(scene_id)}_{str(view_idx)}")
            batched_views['is_metric'].append(self.is_metric)
            batched_views['is_video'].append(ordered_video)
            batched_views['quantile'].append(np.array(0.9, dtype=np.float32))
            batched_views['img_mask'].append(img_mask)
            batched_views['ray_mask'].append(ray_mask)
            batched_views['camera_only'].append(False)
            batched_views['depth_only'].append(False)
            batched_views['single_view'].append(False)
            batched_views['reset'].append(False)
        
        return batched_views
