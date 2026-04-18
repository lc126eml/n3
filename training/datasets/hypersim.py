from collections import defaultdict
import os.path as osp
import os

# sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2
import numpy as np

from training.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2

# Depth images represent Euclidean distances in meters from the camera center.
# Euclidean distances (in meters) to the optical center of the camera
class HyperSim_Multi(BaseMultiViewDataset):
    DEPTH_STATS_HEADER = "# version=1 columns=valid_min_depth\tvalid_mean_depth\tvalid_max_depth\tvalid_count"
    DEPTH_STAT_COLUMNS = {
        "valid_min_depth": 0,
        "valid_mean_depth": 1,
        "valid_max_depth": 2,
        "valid_count": 3,
    }

    def __init__(
        self,
        *args,
        ROOT,
        split='train',
        samples_per_scene=10,
        max_interval=16,
        min_interval=1,
        image_list_path=None,
        depth_filter_spec=None,
        depth_filter_fn=None,
        **kwargs,
    ):
        self.video = True
        self.is_metric = True
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.samples_per_scene = samples_per_scene
        self.image_list_path = image_list_path
        self.depth_filter_spec = depth_filter_spec
        self.depth_filter_fn = depth_filter_fn
        super().__init__(*args, split=split, **kwargs)

        # Keep single-root compatibility while supporting multiple roots.
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
            prefix = f"hypersim_{split_tag}"

        return {
            "scenes": osp.join(cache_dir, f"{prefix}_scenes.txt"),
            "images": osp.join(cache_dir, f"{prefix}_images_fullpath.txt"),
            "scene_img_list": osp.join(cache_dir, f"{prefix}_scene_img_list.txt"),
            "depth_stats": osp.join(cache_dir, f"{prefix}_depth_stats.tsv"),
        }

    def _read_cache(self):
        cache_files = self._get_cache_files()
        if not cache_files:
            return None
        required_files = ("scenes", "images", "scene_img_list")
        if not all(osp.isfile(cache_files[k]) for k in required_files):
            return None

        with open(cache_files["scenes"], "r", encoding="utf-8") as f:
            scenes = [line.rstrip("\n") for line in f]
        with open(cache_files["images"], "r", encoding="utf-8") as f:
            image_fullpaths = [line.strip() for line in f if line.strip()]
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
        if not scenes or not image_fullpaths:
            return None

        depth_stats = None
        depth_stats_path = cache_files["depth_stats"]
        if osp.isfile(depth_stats_path):
            try:
                depth_stats = np.loadtxt(
                    depth_stats_path,
                    delimiter="\t",
                    comments="#",
                    dtype=np.float32,
                )
            except ValueError:
                return None
            depth_stats = np.atleast_2d(depth_stats)
            if depth_stats.shape != (len(image_fullpaths), 4):
                return None

        return scenes, image_fullpaths, scene_img_list, depth_stats

    def _write_cache(self, scenes, images, scene_img_list, depth_stats=None):
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

        if depth_stats is not None:
            np.savetxt(
                cache_files["depth_stats"],
                np.asarray(depth_stats),
                delimiter="\t",
                fmt=["%.8f", "%.8f", "%.8f", "%d"],
                header=self.DEPTH_STATS_HEADER,
                comments="",
            )

    def _depth_filter_enabled(self):
        return self.depth_filter_fn is not None or self.depth_filter_spec is not None

    @staticmethod
    def _depth_stats_for_filter(valid_min_depth, valid_mean_depth, valid_max_depth, valid_count):
        return {
            "valid_min_depth": float(valid_min_depth),
            "valid_mean_depth": float(valid_mean_depth),
            "valid_max_depth": float(valid_max_depth),
            "valid_count": int(valid_count),
        }

    def _normalize_depth_filter_spec(self, spec):
        aliases = {
            "valid_min_depth_ge": "valid_min_depth_ge",
            "min_depth_ge": "valid_min_depth_ge",
            "valid_min_depth_le": "valid_min_depth_le",
            "min_depth_le": "valid_min_depth_le",
            "valid_mean_depth_ge": "valid_mean_depth_ge",
            "mean_depth_ge": "valid_mean_depth_ge",
            "valid_mean_depth_le": "valid_mean_depth_le",
            "mean_depth_le": "valid_mean_depth_le",
            "valid_max_depth_ge": "valid_max_depth_ge",
            "max_depth_ge": "valid_max_depth_ge",
            "valid_max_depth_le": "valid_max_depth_le",
            "max_depth_le": "valid_max_depth_le",
            "valid_count_ge": "valid_count_ge",
            "count_ge": "valid_count_ge",
            "valid_count_le": "valid_count_le",
            "count_le": "valid_count_le",
        }

        normalized_spec = {}
        for key, value in spec.items():
            if key not in aliases:
                raise ValueError(
                    f"Unsupported depth filter key '{key}'. "
                    f"Supported keys: {sorted(aliases.keys())}"
                )
            normalized_spec[aliases[key]] = value

        return normalized_spec

    def _default_depth_keep_mask(self, depth_stats):
        keep_mask = np.ones(len(depth_stats), dtype=bool)
        if not self.depth_filter_spec:
            return keep_mask

        spec = self._normalize_depth_filter_spec(self.depth_filter_spec)
        for field, col_idx in self.DEPTH_STAT_COLUMNS.items():
            ge_key = f"{field}_ge"
            le_key = f"{field}_le"
            if ge_key in spec:
                keep_mask &= depth_stats[:, col_idx] >= float(spec[ge_key])
            if le_key in spec:
                keep_mask &= depth_stats[:, col_idx] <= float(spec[le_key])

        return keep_mask

    def _callable_depth_keep_mask(self, images, depth_stats):
        keep_mask = np.ones(len(depth_stats), dtype=bool)
        for idx, image_path in enumerate(images):
            stats_dict = self._depth_stats_for_filter(*depth_stats[idx])
            keep_mask[idx] = bool(self.depth_filter_fn(image_path, stats_dict, self.depth_filter_spec))
        return keep_mask

    def _apply_depth_filter(self, scenes, images, scene_img_list, depth_stats):
        if not self._depth_filter_enabled():
            return scenes, images, scene_img_list
        if depth_stats is None:
            return None
        if len(depth_stats) != len(images):
            return None

        if self.depth_filter_fn is not None:
            keep_mask = self._callable_depth_keep_mask(images, depth_stats)
        else:
            keep_mask = self._default_depth_keep_mask(depth_stats)

        filtered_scene_img_list = []
        for img_ids in scene_img_list:
            filtered_scene_img_list.append(
                [idx for idx in img_ids if 0 <= idx < len(images) and keep_mask[idx]]
            )

        return scenes, images, filtered_scene_img_list

    @staticmethod
    def _get_depth_path(rgb_path):
        return rgb_path.replace("rgb.png", "depth.npy")

    def compute_depth_stats(self, rgb_path):
        depth_path = self._get_depth_path(rgb_path)
        depthmap = np.load(depth_path).astype(np.float32)
        valid_mask = np.isfinite(depthmap) & (depthmap > 0)
        valid_depths = depthmap[valid_mask]
        if valid_depths.size == 0:
            return (0.0, 0.0, 0.0, 0)
        return (
            float(valid_depths.min()),
            float(valid_depths.mean()),
            float(valid_depths.max()),
            int(valid_depths.size),
        )

    def build_depth_stats(self, images):
        return np.asarray([self.compute_depth_stats(rgb_path) for rgb_path in images], dtype=np.float32)

    def _revalidate_cached_data(self, scenes, images, scene_img_list):
        cut_off = (
            self.num_views * self.min_interval
            if not self.allow_repeat
            else max(self.num_views * self.min_interval // 3, 3)
        )

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
            scenes, images, scene_img_list, depth_stats = cached
            if depth_stats is None and self._depth_filter_enabled():
                depth_stats = self.build_depth_stats(images)
                self._write_cache(scenes, images, scene_img_list, depth_stats)
            cached = self._apply_depth_filter(scenes, images, scene_img_list, depth_stats)
            if cached is not None:
                cached = self._revalidate_cached_data(*cached)
            if cached is not None:
                self.scenes, self.images, self.scene_img_list = cached
                self.all_scenes = sorted({osp.basename(osp.dirname(osp.dirname(p))) for p in self.images})
                return

        rgb_paths = []
        for root in self.roots:
            if not osp.isdir(root):
                continue
            for scene in sorted(f for f in os.listdir(root) if osp.isdir(osp.join(root, f))):
                scene_root = osp.join(root, scene)
                for subscene in sorted(
                    f
                    for f in os.listdir(scene_root)
                    if osp.isdir(osp.join(scene_root, f)) and os.listdir(osp.join(scene_root, f))
                ):
                    subscene_dir = osp.join(scene_root, subscene)
                    rgb_paths.extend(
                        osp.join(subscene_dir, f)
                        for f in sorted(os.listdir(subscene_dir))
                        if f.endswith("rgb.png")
                    )
        if not rgb_paths:
            raise FileNotFoundError(f"No '*rgb.png' files found in roots: {self.roots}")

        scene_to_rgbs = defaultdict(list)
        for rgb_path in rgb_paths:
            scene_to_rgbs[osp.dirname(rgb_path)].append(rgb_path)

        offset = 0
        scenes = []
        images = []
        scene_img_list = []
        all_scene_dirs = sorted(scene_to_rgbs.keys())
        self.all_scenes = sorted({osp.basename(osp.dirname(s)) for s in all_scene_dirs})
        for scene_dir in all_scene_dirs:
            scene_rgbs = sorted(scene_to_rgbs[scene_dir])
            assert len(scene_rgbs) > 0, f"{scene_dir} is empty."
            num_imgs = len(scene_rgbs)
            cut_off = (
                self.num_views*self.min_interval if not self.allow_repeat else max(self.num_views*self.min_interval // 3, 3)
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
            images.extend(scene_rgbs)
            offset += num_imgs

        depth_stats = self.build_depth_stats(images)
        self._write_cache(scenes, images, scene_img_list, depth_stats)

        filtered = self._apply_depth_filter(scenes, images, scene_img_list, depth_stats)
        if filtered is None:
            raise RuntimeError("Depth filtering requested but depth stats were unavailable.")
        filtered = self._revalidate_cached_data(*filtered)
        if filtered is None:
            raise RuntimeError("Failed to prepare HyperSim cache after depth filtering.")

        self.scenes, self.images, self.scene_img_list = filtered
        # self.all_scenes = sorted({osp.basename(osp.dirname(osp.dirname(p))) for p in self.images})

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
            rgb_path = self.images[view_idx]
            rgb_name = osp.basename(rgb_path)
            depth_path = rgb_path.replace("rgb.png", "depth.npy")
            cam_path = rgb_path.replace("rgb.png", "cam.npz")

            rgb_image = imread_cv2(rgb_path, cv2.IMREAD_COLOR)
            depthmap = np.load(depth_path).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            cam_file = np.load(cam_path)
            intrinsics = cam_file["intrinsics"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

            if self.split == "train" and rng.random() < self.random_crop_prob:
                try:
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
                except ValueError as e:
                    # Output the full rgb image path if such error occurred
                    print(f"ValueError: {e}: Image: {rgb_path}")
                    return None  # 
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
            batched_views["label"].append(self.scenes[scene_id] + "-" + rgb_name)
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
