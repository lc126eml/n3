from collections import defaultdict
import os.path as osp
import os
import sys
import itertools

# sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2
import numpy as np

from training.datasets.base.base_multiview_dataset import BaseMultiViewDataset, cropping
from dust3r.utils.image import imread_cv2

# Depth images represent Euclidean distances in meters from the camera center.
# Euclidean distances (in meters) to the optical center of the camera
class HyperSim_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, samples_per_scene=10, max_interval=16, min_interval=1, **kwargs):
        self.video = True
        self.is_metric = True
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.samples_per_scene = samples_per_scene
        # self.split = split
        super().__init__(*args, **kwargs)

        # self.ROOT = osp.join(ROOT, self.split)
        self.ROOT = ROOT
        self.loaded_data = self._load_data()

    def _load_data(self):
        self.all_scenes = sorted(
            [f for f in os.listdir(self.ROOT) if os.path.isdir(osp.join(self.ROOT, f))]
        )
        subscenes = []
        for scene in self.all_scenes:
            # not empty
            subscenes.extend(
                [
                    osp.join(scene, f)
                    for f in os.listdir(osp.join(self.ROOT, scene))
                    if os.path.isdir(osp.join(self.ROOT, scene, f))
                    and len(os.listdir(osp.join(self.ROOT, scene, f))) > 0
                ]
            )

        offset = 0
        scenes = []
        images = []
        scene_img_list = []
        for scene_idx, scene in enumerate(subscenes):
            scene_dir = osp.join(self.ROOT, scene)
            rgb_paths = sorted([f for f in os.listdir(scene_dir) if f.endswith(".png")])
            assert len(rgb_paths) > 0, f"{scene_dir} is empty."
            num_imgs = len(rgb_paths)
            cut_off = (
                self.num_views*self.min_interval if not self.allow_repeat else max(self.num_views*self.min_interval // 3, 3)
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
        """
        Returns a dictionary where each key corresponds to a data field, and
        the value is a stacked numpy array (for images, poses, etc.) or a
        list (for metadata) of all views.
        """
        scene_id = scene_id // self.samples_per_scene
        all_image_ids = self.scene_img_list[scene_id]

        try:
            pos, ordered_video = self.efficient_random_intervals_revised(
                0,
                len(all_image_ids),
                num_views,
                rng,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
            )
        except ValueError as e:
            print(f"Error in _get_views of {scene_id}, {num_views} of {len(all_image_ids)}: {e}")
            return None

        image_idxs = np.array(all_image_ids)[pos]
        
        # 1. Initialize a dictionary of lists to collect batch data
        batched_views = defaultdict(list)

        for v, view_idx in enumerate(image_idxs):
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            rgb_path = self.images[view_idx]
            depth_path = rgb_path.replace("rgb.png", "depth.npy")
            cam_path = rgb_path.replace("rgb.png", "cam.npz")

            rgb_image = imread_cv2(osp.join(scene_dir, rgb_path), cv2.IMREAD_COLOR)
            depthmap = np.load(osp.join(scene_dir, depth_path)).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
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
            
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            # 2. Append each piece of data to its corresponding list
            batched_views["img"].append(rgb_image)
            batched_views["depthmap"].append(depthmap.astype(np.float32))
            batched_views["camera_pose"].append(camera_pose.astype(np.float32))
            batched_views["camera_intrinsics"].append(intrinsics.astype(np.float32))
            batched_views["dataset"].append("hypersim")
            batched_views["label"].append(self.scenes[scene_id] + "-" + rgb_path)
            batched_views["instance"].append(f"{str(scene_id)}_{str(view_idx)}")
            batched_views["is_metric"].append(self.is_metric)
            batched_views["is_video"].append(ordered_video)
            batched_views["quantile"].append(np.array(1.0, dtype=np.float32))
            batched_views["img_mask"].append(img_mask)
            batched_views["ray_mask"].append(ray_mask)
            batched_views["camera_only"].append(False)
            batched_views["depth_only"].append(False)
            batched_views["single_view"].append(False)
            batched_views["reset"].append(False)

        return batched_views

# if __name__ == "__main__":
#     from dust3r.datasets.utils.transforms import ImgNorm, SeqColorJitter
#     dataset = HyperSim_Multi(allow_repeat=False, split='train', ROOT="/lc/data/3D/processed_hypersim", aug_crop=1, resolution=224, transform=SeqColorJitter, num_views=4, n_corres=0)
#     print(type(dataset), len(dataset))
#     print(dataset.get_image_num())

#     rng = np.random.default_rng()
#     views = dataset._get_views(2, (320, 228), rng, 1)
#     print(len(views), type(views[0]))
#     view = views[0]
#     print(view.keys())
#     img, depth,pose = view['img'], view['depthmap'], view['camera_pose']
#     print(depth.shape, pose.shape)
#     print(type(img))
