import os.path as osp
import os
import numpy as np
import cv2
from collections import defaultdict
from training.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2
# cameras accurate, depthmaps noisy 
class BlendedMVS_Multi(BaseMultiViewDataset):
    """Dataset of outdoor scenes from BlendedMVS dataset"""

    def __init__(
        self,
        *args,
        ROOT,
        split='train',
        samples_per_scene=10,
        max_interval=9,
        image_list_path=None,
        dataset_tag="blendedmvs",
        **kwargs,
    ):
        self.video = False
        self.is_metric = False
        self.max_interval = max_interval
        self.samples_per_scene = samples_per_scene
        self.quantile = 0.90
        self.image_list_path = image_list_path
        self.dataset_tag = dataset_tag
        super().__init__(*args, split=split, **kwargs)

        if isinstance(ROOT, (list, tuple)):
            root_list = list(ROOT)
        else:
            root_list = [ROOT]
        self.roots = root_list
	
        self.loaded_data = self._load_data()

    def _get_cache_files(self):
        if not self.image_list_path:
            return None

        split_tag = self.split if self.split is not None else "all"
        path_str = str(self.image_list_path)
        if path_str.endswith(".txt"):
            cache_dir = osp.dirname(path_str)
            base_name = osp.splitext(osp.basename(path_str))[0]
            prefix = base_name
        else:
            cache_dir = path_str
            prefix = f"{self.dataset_tag}_{split_tag}"

        return {
            "scenes": osp.join(cache_dir, f"{prefix}_scenes.txt"),
            "images": osp.join(cache_dir, f"{prefix}_images_fullpath.txt"),
            "scene_img_list": osp.join(cache_dir, f"{prefix}_scene_img_list.txt"),
        }

    def _read_cache(self):
        cache_files = self._get_cache_files()
        if not cache_files:
            return None
        if not all(osp.isfile(p) for p in cache_files.values()):
            return None

        with open(cache_files["scenes"], "r", encoding="utf-8") as f:
            scenes = [line.rstrip("\n") for line in f]
        with open(cache_files["images"], "r", encoding="utf-8") as f:
            images = [line.strip() for line in f if line.strip()]
        with open(cache_files["scene_img_list"], "r", encoding="utf-8") as f:
            scene_img_list = []
            for line in f:
                line = line.strip()
                if not line:
                    scene_img_list.append([])
                    continue
                scene_img_list.append([int(x) for x in line.split()])

        if len(scenes) != len(scene_img_list):
            return None
        if not scenes or not images:
            return None
        return scenes, images, scene_img_list

    def _write_cache(self, scenes, images, scene_img_list):
        cache_files = self._get_cache_files()
        if not cache_files:
            return
        cache_dir = osp.dirname(cache_files["scenes"])
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        with open(cache_files["scenes"], "w", encoding="utf-8") as f:
            for scene in scenes:
                f.write(str(scene) + "\n")
        with open(cache_files["images"], "w", encoding="utf-8") as f:
            for p in images:
                f.write(p + "\n")
        with open(cache_files["scene_img_list"], "w", encoding="utf-8") as f:
            for img_ids in scene_img_list:
                f.write(" ".join(str(i) for i in img_ids) + "\n")

    def _revalidate_cached_data(self, scenes, images, scene_img_list):
        cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)

        kept_scenes = []
        kept_scene_img_list = []
        for scene, img_ids in zip(scenes, scene_img_list):
            if len(img_ids) < cut_off:
                print(f"Skipping cached {scene}")
                continue
            kept_scenes.append(scene)
            kept_scene_img_list.append(img_ids)

        if not kept_scenes:
            return [], [], []

        old_to_new = {}
        new_images = []
        new_scene_img_list = []
        for img_ids in kept_scene_img_list:
            remapped = []
            for idx in img_ids:
                if idx < 0 or idx >= len(images):
                    return None
                if idx not in old_to_new:
                    old_to_new[idx] = len(new_images)
                    new_images.append(images[idx])
                remapped.append(old_to_new[idx])
            new_scene_img_list.append(remapped)

        return kept_scenes, new_images, new_scene_img_list

    def _load_data(self):
        cached = self._read_cache()
        if cached is not None:
            cached = self._revalidate_cached_data(*cached)
            if cached is None:
                cached = None
            else:
                self.scenes, self.images, self.scene_img_list = cached
                self.all_scenes = sorted({osp.basename(osp.dirname(p)) for p in self.images})
                return

        scene_to_rgbs = defaultdict(list)
        all_scenes = []
        for root in self.roots:
            if not osp.isdir(root):
                continue
            for scene in sorted([f for f in os.listdir(root) if os.path.isdir(osp.join(root, f))]):
                scene_dir = osp.join(root, scene)
                all_scenes.append(scene)
                rgb_paths = sorted(
                    [osp.join(scene_dir, f) for f in os.listdir(scene_dir) if f.endswith("_rgb.png")]
                )
                if not rgb_paths:
                    print(f"Skipping {scene_dir}: No *_rgb.png files found.")
                    continue
                scene_to_rgbs[scene_dir].extend(rgb_paths)
        self.all_scenes = sorted(set(all_scenes))

        scenes = []
        images = []
        scene_img_list = []
        offset = 0
        for scene_dir in sorted(scene_to_rgbs.keys()):
            rgb_paths = sorted(scene_to_rgbs[scene_dir])

            num_imgs = len(rgb_paths)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            if num_imgs < cut_off:
                print(f"Skipping {scene_dir}")
                continue

            img_ids = list(np.arange(num_imgs) + offset)
            scene_label = None
            for root in self.roots:
                try:
                    candidate = osp.relpath(scene_dir, root)
                except ValueError:
                    continue
                if not candidate.startswith(".."):
                    scene_label = candidate
                    break
            scenes.append(scene_label if scene_label is not None else scene_dir)
            scene_img_list.append(img_ids)
            images.extend(rgb_paths)
            offset += num_imgs

        self.scenes = scenes
        self.images = images
        self.scene_img_list = scene_img_list
        self._write_cache(self.scenes, self.images, self.scene_img_list)

    def __len__(self):
        return len(self.scenes) * self.samples_per_scene

    def get_image_num(self):
        return len(self.images)

    def get_stats(self):
        return f"{len(self)} imgs from {len(self.scenes)} scenes"

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
            rgb_path = self.images[view_idx]
            rgb_name = osp.basename(rgb_path)
            depth_path = rgb_path.replace("_rgb.png", "_depth.exr")
            cam_path = rgb_path.replace("_rgb.png", "_cam.npz")

            rgb_image = imread_cv2(rgb_path, cv2.IMREAD_COLOR)
            depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
            cam_file = np.load(cam_path)
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

            # 2. Append each piece of data to its corresponding list
            batched_views['img'].append(rgb_image)
            batched_views['depthmap'].append(depthmap.astype(np.float32))
            batched_views['camera_pose'].append(camera_pose.astype(np.float32))
            batched_views['camera_intrinsics'].append(intrinsics.astype(np.float32))
            batched_views['dataset'].append("blendedmvs")
            batched_views['label'].append(self.scenes[scene_id] + "_" + rgb_name)
            batched_views['instance'].append(f"{str(scene_id)}_{str(view_idx)}")
            batched_views['is_metric'].append(self.is_metric)
            batched_views['is_video'].append(ordered_video)
            batched_views['quantile'].append(np.array(self.quantile, dtype=np.float32))
            batched_views['img_mask'].append(img_mask)
            batched_views['ray_mask'].append(ray_mask)
            batched_views['camera_only'].append(False)
            batched_views['depth_only'].append(False)
            batched_views['single_view'].append(False)
            batched_views['reset'].append(False)
        
        return batched_views
