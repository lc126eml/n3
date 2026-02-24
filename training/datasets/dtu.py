import os
import sys
import PIL
import cv2
import re
import numpy as np
import os.path as osp
from collections import deque, defaultdict
import glob

from training.datasets.base.base_multiview_dataset import BaseMultiViewDataset, cropping
from dust3r.utils.image import imread_cv2


class DTU_Multi(BaseMultiViewDataset):
    """
    A PyTorch dataset for loading data from a PRE-PROCESSED DTU MVS dataset.
    This class is designed to read the output of our preprocessing script
    (aligned images, .exr depth maps, and .npz camera files) and is
    compatible with the DUST3R model.
    """

    def __init__(self, *args, ROOT, samples_per_scene=1, max_interval=5, **kwargs):
        """
        Initializes the DTU dataset for processed data.

        Args:
            ROOT (str): The root directory of the PROCESSED DTU dataset.
            samples_per_scene (int, optional): Number of samples to generate per scene. Defaults to 1.
            max_interval (int, optional): Max frame interval for sampling. Defaults to 5.
        """
        self.ROOT = ROOT
        self.samples_per_scene = samples_per_scene
        self.is_metric = True  # DTU depth is metric
        self.video = True
        self.max_interval = max_interval
        
        super().__init__(*args, **kwargs)
        self.load_all_scenes()

    def load_all_scenes(self):
        """
        Loads all scene subdirectories from the processed root directory.
        """
        self.scene_list = sorted([
            f for f in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, f))
        ])
        print(f"Found {len(self.scene_list)} processed scenes in the dataset.")

    def __len__(self):
        return len(self.scene_list) * self.samples_per_scene

    def _get_views(self, idx, resolution, rng, num_views):
        """
        Retrieves a set of views for a given index from the processed dataset.

        Args:
            idx (int): The index of the sample to load.
            resolution (tuple): The desired resolution (Note: data is already small, so this may not be needed).
            rng (np.random.Generator): The random number generator.
            num_views (int): The number of views to sample.

        Returns:
            list: A list of dictionaries, where each dictionary represents a view.
        """
        scene_idx = idx // self.samples_per_scene
        scene_id = self.scene_list[scene_idx]

        scene_path = osp.join(self.ROOT, scene_id)
        # Get all processed rgb files and sample a subset
        # pattern = os.path.join(scene_path, '*_0_rgb.png')
        # all_images = sorted([os.path.basename(f) for f in glob.glob(pattern)])
        all_image_ids = sorted([f.removesuffix("_cam.npz") for f in os.listdir(scene_path) if f.endswith("_cam.npz")])
        # all_images = sorted([f for f in os.listdir(scene_path) if f.endswith('_rgb.png')])
        try:
            pos, ordered_video = self.efficient_random_intervals_revised(
                0,
                len(all_image_ids),
                num_views,
                rng,
                min_interval=1,
                max_interval=self.max_interval,
            )
        except ValueError as e:
            print(f"Error in _get_views of {scene_id}, {num_views} of {len(all_image_ids)}: {e}")
            return None

        img_ids = [all_image_ids[i] for i in pos]

        # 1. Initialize a dictionary of lists to collect batch data
        batched_views = defaultdict(list)

        for v, base_name in enumerate(img_ids):
            # --- Changed: Construct paths for processed files ---
            i = rng.integers(7)  # Random lighting condition from 0 to 6
            rgb_path = osp.join(scene_path, f"{base_name}_{i}_rgb.png")
            depth_path = osp.join(scene_path, f'{base_name}_depth.exr')
            cam_path = osp.join(scene_path, f'{base_name}_cam.npz')

            if not all(osp.exists(p) for p in [rgb_path, depth_path, cam_path]):
                print(f"Warning: Missing processed file for {base_name}. Skipping.")
                continue

            # --- Changed: Load processed data ---
            rgb_image = imread_cv2(rgb_path)
            # Use cv2.IMREAD_UNCHANGED for single-channel float .exr files
            depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)/1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            
            camera_data = np.load(cam_path)
            # print(type(camera_data))
            # print(list(camera_data.keys()))
            # sys.exit(0)
            intrinsics = camera_data["intrinsics"]
            camera_pose = camera_data['pose']
            camera_pose[:3, 3] /= 1000

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
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=rgb_path
                )
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            # 2. Append each piece of data to its corresponding list
            batched_views['img'].append(rgb_image)
            batched_views['depthmap'].append(depthmap.astype(np.float32))
            batched_views['camera_pose'].append(camera_pose.astype(np.float32))
            batched_views['camera_intrinsics'].append(intrinsics.astype(np.float32))
            batched_views['dataset'].append('dtu')
            batched_views['label'].append(scene_id + "-" + os.path.basename(rgb_path))
            batched_views['instance'].append(f"{scene_idx}_{v}")
            batched_views['is_metric'].append(self.is_metric)
            batched_views['is_video'].append(ordered_video)
            batched_views['quantile'].append(np.array(1.0, dtype=np.float32))
            batched_views['img_mask'].append(img_mask)
            batched_views['ray_mask'].append(ray_mask)
            batched_views['camera_only'].append(False)
            batched_views['depth_only'].append(False)
            batched_views['single_view'].append(False)
            batched_views['reset'].append(False)
        
        return batched_views
