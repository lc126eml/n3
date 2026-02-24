import os
import PIL
import cv2
import re
import numpy as np
import os.path as osp
from collections import deque, defaultdict

# Assuming the base class and utilities are available in your project structure
from training.datasets.base.base_multiview_dataset import BaseMultiViewDataset, cropping
from dust3r.utils.image import imread_cv2


class ETH3D_Multi(BaseMultiViewDataset):
    """
    A PyTorch dataset for loading data from a PRE-PROCESSED ETH3D dataset.
    This class is designed to read the output of our preprocessing script
    (resized images, .exr depth maps, and .npz camera files) and is
    compatible with the DUST3R model.
    """

    def __init__(self, *args, ROOT, samples_per_scene=1, max_interval=15, **kwargs):
        """
        Initializes the ETH3D dataset for processed data.

        Args:
            ROOT (str): The root directory of the PROCESSED ETH3D dataset.
            samples_per_scene (int, optional): Number of samples to generate per subscene. Defaults to 1.
            max_interval (int, optional): Max frame interval for sampling. Defaults to 10.
        """
        self.ROOT = ROOT
        self.samples_per_scene = samples_per_scene
        self.is_metric = True  # Processed ETH3D depth is metric
        self.video = True
        self.max_interval = max_interval
        
        super().__init__(*args, **kwargs)
        self.load_all_scenes()

    def load_all_scenes(self):
        """
        Loads all scene and subscene directories from the processed root directory.
        The ETH3D dataset has a nested structure: scene -> subscene (camera rig).
        """
        self.subscene_list = []
        # Find top-level scenes (e.g., 'forest', 'playground')
        top_level_scenes = sorted([
            f for f in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, f))
        ])
        
        # For each top-level scene, find the subscenes (camera rigs)
        for scene in top_level_scenes:
            scene_path = osp.join(self.ROOT, scene)
            subscenes = sorted([
                f for f in os.listdir(scene_path) if osp.isdir(osp.join(scene_path, f))
            ])
            for subscene in subscenes:
                self.subscene_list.append((scene, subscene))

        print(f"Found {len(self.subscene_list)} processed subscenes in the dataset.")

    def __len__(self):
        return len(self.subscene_list) * self.samples_per_scene

    def _get_views(self, idx, resolution, rng, num_views):
        """
        Retrieves a set of views for a given index from the processed dataset.

        Args:
            idx (int): The index of the sample to load.
            resolution (tuple): The desired resolution for potential resizing/cropping.
            rng (np.random.Generator): The random number generator.
            num_views (int): The number of views to sample.

        Returns:
            list: A list of dictionaries, where each dictionary represents a view.
        """
        subscene_idx = idx // self.samples_per_scene
        scene_id, subscene_id = self.subscene_list[subscene_idx]
        
        subscene_path = osp.join(self.ROOT, scene_id, subscene_id)

        # Get all processed rgb files and sample a subset
        all_images = sorted([f for f in os.listdir(subscene_path) if f.endswith('_rgb.png')])
        
        # Use a sampling strategy to select view indices
        pos, ordered_video = self.efficient_random_intervals_revised(
            0,
            len(all_images),
            num_views,
            rng,
            min_interval=1,
            max_interval=self.max_interval,
        )

        img_filenames = [all_images[i] for i in pos]

        # 1. Initialize a dictionary of lists to collect batch data
        batched_views = defaultdict(list)

        for v, rgb_filename in enumerate(img_filenames):
            # e.g., '1477833684658155598_rgb.png' -> '1477833684658155598'
            base_name = rgb_filename.replace('_rgb.png', '')

            # Construct paths for processed files within the subscene directory
            rgb_path = osp.join(subscene_path, rgb_filename)
            depth_path = osp.join(subscene_path, f'{base_name}_depth.exr')
            cam_path = osp.join(subscene_path, f'{base_name}_cam.npz')

            if not all(osp.exists(p) for p in [rgb_path, depth_path, cam_path]):
                print(f"Warning: Missing processed file for {base_name} in {subscene_path}. Skipping.")
                continue

            # Load the processed data
            rgb_image = imread_cv2(rgb_path) # Assumes RGB format
            
            # Use cv2.IMREAD_UNCHANGED for single-channel float .exr files
            depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            # Our preprocessed .exr files already store metric depth, so no scaling is needed.
            # Ensure correct dtype and handle potential NaNs.
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            
            camera_data = np.load(cam_path)
            intrinsics = camera_data["intrinsics"]
            camera_pose = camera_data['pose']
            
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
            batched_views['dataset'].append('eth3d')
            batched_views['label'].append(scene_id + "_" + subscene_id + "-" + base_name)
            batched_views['instance'].append(f"{subscene_idx}_{v}")
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
