import PIL
import numpy as np
import torch
import random
import itertools
from .easy_dataset import EasyDataset
from training.datasets.utils.transforms import SeqColorJitter
from dust3r.utils.image import ImgNorm, ImgnetNorm
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, align_poses
from training.vggt.utils.geometry import align_poses_batched
from training.datasets.utils.augmentation import get_image_augmentation
import training.datasets.utils.cropping as cropping
from training.datasets.utils.geometry_augment import random_crop
from training.datasets.utils.rotate import rotate_k90_aug
from torchvision import transforms


def get_ray_map(c2w1, c2w2, intrinsics, h, w):
    c2w = np.linalg.inv(c2w1) @ c2w2
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    grid = np.stack([i, j, np.ones_like(i)], axis=-1)
    ro = c2w[:3, 3]
    rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
    rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
    rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.broadcast_to(ro, (h, w, 3))
    ray_map = np.concatenate([ro, rd], axis=-1)
    return ray_map


class BaseMultiViewDataset(EasyDataset):
    """Define all basic options.

    Usage:
        class MyDataset (BaseMultiViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(
        self,
        *,  # only keyword arguments
        num_views=None,
        split=None,
        resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
        transform=None,
        n_corres=0,
        nneg=0,
        seed=None,
        allow_repeat=False,
        seq_aug_crop=False,
        augs=None,
        align_poses=False, 
        **kwargs,
    ):
        assert num_views is not None, "undefined num_views"
        self.num_views = num_views
        self.split = split
        self.random_crop_prob = 0.0
        self._set_resolutions(resolution)

        self.n_corres = n_corres
        self.nneg = nneg
        assert (
            self.n_corres == "all"
            or isinstance(self.n_corres, int)
            or (
                isinstance(self.n_corres, list) and len(self.n_corres) == self.num_views
            )
        ), f"Error, n_corres should either be 'all', a single integer or a list of length {self.num_views}"
        assert (
            self.nneg == 0 or self.n_corres != "all"
        ), "nneg should be 0 if n_corres is all"

        self.transform = transforms.ToTensor()
        self.aug_crop = 0
        self.set_augs(augs, transform)
        self.seed = seed
        self.allow_repeat = allow_repeat
        self.seq_aug_crop = seq_aug_crop
        self.align_poses = align_poses

    def set_augs(self, augs, transform=None):
        """Dynamically set augmentations after instantiation."""
        if augs:
            self.augs = augs
            self.prot=augs.get("prot", 0.5)
            self.pcrop=augs.get("pcrop", 0.95)
            if "aug_crop" in augs:
                self.aug_crop = augs.get("aug_crop", 0)
            random_crop_prob = augs.get("random_crop_prob", augs.get("random_crop", 0.0))
            if isinstance(random_crop_prob, bool):
                random_crop_prob = 1.0 if random_crop_prob else 0.0
            self.random_crop_prob = max(0.0, min(1.0, float(random_crop_prob)))
            self.image_aug = get_image_augmentation(
                color_jitter=augs.get("color_jitter"),
                gray_scale=augs.get("gray_scale", False),
                gau_blur=augs.get("gau_blur", False),
            )
            if transform is not None:
                # This assumes torchvision.transforms.Compose is available
                self.transform = transforms.Compose([self.image_aug, transform])
            else:
                self.transform = transforms.Compose([self.image_aug, transforms.ToTensor()])
        elif transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
            
    def set_resolutions(self, resolutions):
        """Dynamically set resolutions after instantiation."""
        self._set_resolutions(resolutions)

    def __len__(self):
        return len(self.scenes)
    
    @staticmethod
    def efficient_random_intervals_uniform(
        start,
        max_idx,
        num_elements,
        rng,
        interval_range=None,
        min_interval=1,
        max_interval=25,
        video_prob=0.8,
        weights=None,
        seed=42,
    ):
        """
        Generate random intervals with equal probability for all indices between start and max_idx.
        
        Args:
            start: Starting index from 0
            num_elements: Number of elements to select
            interval_range: List of possible interval sizes
            max_idx: Maximum index allowed (exclusive)
            video_prob: Probability of using ordered intervals
            weights: Weights for the interval choices
            seed: Random seed
            
        Returns:
            List of indices between start and max_idx
        """
        # random.seed(seed)
        
        # Calculate available range
        available_range = max_idx - start
        
        # Ensure we can select num_elements points within the range
        if num_elements > available_range:
            raise ValueError(f"Cannot select {num_elements} elements in range of size {available_range}")
        
        if interval_range is None:
            interval_range = list(range(min_interval, max_interval + 1))

        # Generate intervals
        intervals = rng.choice(interval_range, size=num_elements - 1, replace=True, p=weights).tolist()
        # Adjust intervals if they exceed max_idx
        total_interval = sum(intervals)
        while total_interval > available_range - 1:
            # Find the largest interval to decrease
            max_interval_idx = intervals.index(max(intervals))
            intervals[max_interval_idx] -= 1
            total_interval -= 1

        if total_interval < available_range - 1:
            start += rng.integers(0, available_range - total_interval)
        # Generate the final indices
        pos = list(itertools.accumulate([start] + intervals))
        
        return pos, False

    @staticmethod
    def efficient_random_intervals_revised(
        start,
        max_idx,
        num_elements,
        rng,
        interval_range=None,
        min_interval=1,
        max_interval=25,
        weights=None,
    ):
        """
        Generate random intervals efficiently. If the total interval sum exceeds the allowed range,
        it is reduced by subtracting the largest possible chunks from randomly selected intervals.
        
        Args:
            start: Starting index from 0.
            max_idx: Maximum index allowed (exclusive).
            num_elements: Number of elements to select.
            rng: A numpy random number generator.
            interval_range: List of possible interval sizes.
            min_interval: The minimum allowed interval.
            max_interval: The maximum allowed interval.
            weights: Weights for the interval choices.
            
        Returns:
            A list of indices and a boolean flag (always False in this implementation).
        """
        available_range = max_idx - start
        if num_elements * min_interval > available_range:
            raise ValueError(f"Cannot select {num_elements} elements with {min_interval} min interval in a range of size {available_range}")

        if interval_range is None:
            interval_range = list(range(min_interval, max_interval + 1))

        # Generate initial intervals
        intervals = rng.choice(interval_range, size=num_elements - 1, replace=True, p=weights).tolist()
        
        excess = sum(intervals) - (available_range - 1)

        # Reduce intervals by the maximum possible amount from random intervals until excess is gone.
        while excess > 0:
            reducible_indices = [i for i, interval in enumerate(intervals) if interval > min_interval]
            if not reducible_indices:
                break 
            
            idx_to_reduce = rng.choice(reducible_indices)
            
            # Determine the maximum possible reduction for the chosen interval
            max_reduction_for_interval = intervals[idx_to_reduce] - min_interval
            
            # Reduce by the smaller of the excess or the max possible reduction
            reduction_amount = min(excess, max_reduction_for_interval)
            
            intervals[idx_to_reduce] -= reduction_amount
            excess -= reduction_amount

        # Randomly shift the start position within the available slack
        total_interval = sum(intervals)
        slack = (available_range - 1) - total_interval
        if slack > 0:
            start += rng.integers(0, slack + 1)

        # Generate the final indices by accumulating intervals from the start
        pos = list(itertools.accumulate([start] + intervals))
        
        return pos, False

    @staticmethod
    def efficient_random_intervals(
        start,
        num_elements,
        interval_range,
        fixed_interval_prob=0.8,
        weights=None,
        seed=42,
    ):
        if random.random() < fixed_interval_prob:
            intervals = random.choices(interval_range, weights=weights) * (
                num_elements - 1
            )
        else:
            intervals = [
                random.choices(interval_range, weights=weights)[0]
                for _ in range(num_elements - 1)
            ]
        return list(itertools.accumulate([start] + intervals))

    def sample_based_on_timestamps(self, i, timestamps, num_views, interval=1):
        time_diffs = np.abs(timestamps - timestamps[i])
        ids_candidate = np.where(time_diffs < interval)[0]
        ids_candidate = np.sort(ids_candidate)
        if (self.allow_repeat and len(ids_candidate) < num_views // 3) or (
            len(ids_candidate) < num_views
        ):
            return []
        ids_sel_list = []
        ids_candidate_left = ids_candidate.copy()
        while len(ids_candidate_left) >= num_views:
            ids_sel = np.random.choice(ids_candidate_left, num_views, replace=False)
            ids_sel_list.append(sorted(ids_sel))
            ids_candidate_left = np.setdiff1d(ids_candidate_left, ids_sel)

        if len(ids_candidate_left) > 0 and len(ids_candidate) >= num_views:
            ids_sel = np.concatenate(
                [
                    ids_candidate_left,
                    np.random.choice(
                        np.setdiff1d(ids_candidate, ids_candidate_left),
                        num_views - len(ids_candidate_left),
                        replace=False,
                    ),
                ]
            )
            ids_sel_list.append(sorted(ids_sel))

        if self.allow_repeat:
            ids_sel_list.append(
                sorted(np.random.choice(ids_candidate, num_views, replace=True))
            )

        # add sequences with fixed intervals (all possible intervals)
        pos_i = np.where(ids_candidate == i)[0][0]
        curr_interval = 1
        stop = len(ids_candidate) < num_views
        while not stop:
            pos_sel = [pos_i]
            count = 0
            while len(pos_sel) < num_views:
                if count % 2 == 0:
                    curr_pos_i = pos_sel[-1] + curr_interval
                    if curr_pos_i >= len(ids_candidate):
                        stop = True
                        break
                    pos_sel.append(curr_pos_i)
                else:
                    curr_pos_i = pos_sel[0] - curr_interval
                    if curr_pos_i < 0:
                        stop = True
                        break
                    pos_sel.insert(0, curr_pos_i)
                count += 1
            if not stop and len(pos_sel) == num_views:
                ids_sel = sorted([ids_candidate[pos] for pos in pos_sel])
                if ids_sel not in ids_sel_list:
                    ids_sel_list.append(ids_sel)
            curr_interval += 1
        return ids_sel_list

    @staticmethod
    def blockwise_shuffle(x, rng, block_shuffle):
        if block_shuffle is None:
            return rng.permutation(x).tolist()
        else:
            assert block_shuffle > 0
            blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
            shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
            shuffled_list = [item for block in shuffled_blocks for item in block]
            return shuffled_list

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        """
        args:
            num_views: number of views to return
            id_ref: the reference id (first id)
            ids_all: all the ids
            rng: random number generator
            max_interval: maximum interval between two views
        returns:
            pos: list of positions of the views in ids_all, i.e., index for ids_all
            is_video: True if the views are consecutive
        """
        assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
        assert (
            min_interval <= max_interval
        ), f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
        assert id_ref in ids_all
        pos_ref = ids_all.index(id_ref)
        all_possible_pos = np.arange(pos_ref, len(ids_all))

        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
            intervals = [
                rng.choice(range(min_interval, max_interval + 1))
                for _ in range(num_views - 1)
            ]

            # if video or collection
            if rng.random() < video_prob:
                # if fixed interval or random
                if rng.random() < fix_interval_prob:
                    # regular interval
                    fixed_interval = rng.choice(
                        range(
                            1,
                            min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                        )
                    )
                    intervals = [fixed_interval for _ in range(num_views - 1)]
                is_video = True
            else:
                is_video = False

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = (
                pos
                + rng.choice(
                    pos_candidates, num_views - len(pos), replace=False
                ).tolist()
            )

            pos = (
                sorted(pos)
                if is_video
                else self.blockwise_shuffle(pos, rng, block_shuffle)
            )
        else:
            # assert self.allow_repeat
            uniq_num = remaining_sum
            new_pos_ref = rng.choice(np.arange(pos_ref + 1))
            new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
            new_intervals = [
                rng.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)
            ]

            revisit_random = rng.random()
            video_random = rng.random()

            if rng.random() < fix_interval_prob and video_random < video_prob:
                # regular interval
                fixed_interval = rng.choice(range(1, new_max_interval + 1))
                new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:  # revisit, video / collection
                is_video = video_random < video_prob
                pos = (
                    self.blockwise_shuffle(pos, rng, block_shuffle)
                    if not is_video
                    else pos
                )
                num_full_repeat = num_views // uniq_num
                pos = (
                    pos * num_full_repeat
                    + pos[: num_views - len(pos) * num_full_repeat]
                )
            elif revisit_random < 0.9:  # random
                pos = rng.choice(pos, num_views, replace=True)
            else:  # ordered
                pos = sorted(rng.choice(pos, num_views, replace=True))
        assert len(pos) == num_views
        return pos, is_video

    def get_img_and_ray_masks(self, is_metric, v, rng, p=[0.8, 0.15, 0.05]):
        # generate img mask and raymap mask
        if v == 0 or (not is_metric):
            img_mask = True
            raymap_mask = False
        else:
            rand_val = rng.random()
            if rand_val < p[0]:
                img_mask = True
                raymap_mask = False
            elif rand_val < p[0] + p[1]:
                img_mask = False
                raymap_mask = True
            else:
                img_mask = True
                raymap_mask = True
        return img_mask, raymap_mask

    def get_stats(self):
        return f"{len(self)} groups of views"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.num_views=},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, resolution, rng, num_views):
        raise NotImplementedError()

    def __getitem__(self, idx):
        # print("Receiving:" , idx)
        if isinstance(idx, (tuple, list, np.ndarray)):
            # the idx is specifying the aspect-ratio
            idx, ar_idx, nview = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
            nview = self.num_views

        assert nview >= 1 and nview <= self.num_views
        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=seed)

        if self.aug_crop > 1 and self.seq_aug_crop:
            self.delta_target_resolution = self._rng.integers(0, self.aug_crop)

        # over-loaded code
        resolution = self._resolutions[
            ar_idx
        ]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng, nview)
        assert len(views['img']) == nview

        assert "camera_pose" in views
        # if "camera_pose" not in views[0]:
        #     views[0]["camera_pose"] = np.ones((4, 4), dtype=np.float32)
        if self.align_poses and "camera_pose" in views:
            views['camera_pose'] = align_poses_batched(views['camera_pose'])
        # first_view_camera_pose = views[0]["camera_pose"]
        # transform = SeqColorJitter() if self.is_seq_color_jitter else self.transform

        assert (
                "pts3d" not in views
            ), f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap"
        assert "camera_intrinsics" in views
        assert "valid_mask" not in views
        for v in range(nview):
            # The original assertion is now implicitly handled by the loop structure.
            # We are building the "pts3d" list from scratch.
            
            views["idx"].append((idx, ar_idx, v))

            # Get image from the list and process it
            img = views["img"][v]
            width, height = img.size
            views["true_shape"].append(np.int32((height, width)))

            if self.transform is not None:
                # Modify the image in the list in-place
                views["img"][v] = self.transform(img)

            views["sky_mask"].append(views["depthmap"][v] < 0)


            
            assert np.isfinite(
                views["camera_pose"][v]
            ).all(), f"NaN in camera pose for view index {v}"
            # ray_map = get_ray_map(
                    #     first_view_camera_pose,
                    #     view["camera_pose"],
                    #     view["camera_intrinsics"],
                    #     height,
                    #     width,
                    # )
                    # view["ray_map"] = ray_map.astype(np.float32)
            assert np.isfinite(
                views["depthmap"][v]
            ).all(), f"NaN in depthmap for view index {v}"

            # --- Reconstruct a temporary dict for the current view's data ---
            # This is needed to call the function with keyword arguments (**).
            current_view_data = {
                "depthmap": views["depthmap"][v],
                "camera_intrinsics": views["camera_intrinsics"][v],
                "camera_pose": views["camera_pose"][v],
                # Add any other keys the function requires from the views dict
            }
            pts3d, valid_mask, pts3d_cam = depthmap_to_absolute_camera_coordinates(**current_view_data)

            # --- Append the new computed values to their lists ---
            views["pts3d"].append(pts3d)
            views["pts3d_cam"].append(pts3d_cam)
            views["valid_mask"].append(valid_mask & np.isfinite(pts3d).all(axis=-1))

        for key, val_list in views.items():
            for v, val in enumerate(val_list):
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view index {v}"

        # 3. Stack the lists of numpy arrays into a single batched array
        final_batch = {}
        for key, value_list in views.items():
            if isinstance(value_list[0], np.ndarray):
                # Stack along a new batch dimension (axis=0)
                final_batch[key] = np.stack(value_list, axis=0)
            elif isinstance(value_list[0], torch.Tensor):
                final_batch[key] = torch.stack(value_list, dim=0)
            else:
                # For non-array data (like strings or booleans), just keep the list
                final_batch[key] = value_list
        return final_batch

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(
                width, int
            ), f"Bad type for {width=} {type(width)=}, should be int"
            assert isinstance(
                height, int
            ), f"Bad type for {height=} {type(height)=}, should be int"
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(
        self, image, depthmap, intrinsics, resolution, rng=None, info=None, crop=False
    ):
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        if crop:
            W, H = image.size
            cx, cy = intrinsics[:2, 2].round().astype(int)
            min_margin_x = min(cx, W - cx)
            min_margin_y = min(cy, H - cy)
            assert min_margin_x > W / 5, f"Bad principal point in view={info}"
            assert min_margin_y > H / 5, f"Bad principal point in view={info}"
            # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
            l, t = cx - min_margin_x, cy - min_margin_y
            r, b = cx + min_margin_x, cy + min_margin_y
            crop_bbox = (l, t, r, b)
            image, depthmap, intrinsics = cropping.crop_image_depthmap(
                image, depthmap, intrinsics, crop_bbox
            )

        # transpose the resolution if necessary
        W, H = image.size  # new size

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )        

        return image, depthmap, intrinsics2

    def _crop_resize(
        self,
        image,
        depthmap,
        intrinsics,
        resolution,
        rng=None,
        prot=0.5,
        pcrop=0.95,
        pose=None,
    ):
        rng = rng or np.random.default_rng()
        target_resolution = np.array(resolution)

        if rng.random() < prot:
            k = int(rng.integers(1, 4))
            image_np = np.array(image) if isinstance(image, PIL.Image.Image) else image
            image_np, depthmap, intrinsics, pose, _ = rotate_k90_aug(
                image_np, depthmap, intrinsics, pose, k=k
            )
            image = image_np

        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        if rng.random() < pcrop:
            if self.augs is not None:
                aspect_ratio_range = self.augs.get("aspects", [1.0, 1.5])
            else:
                aspect_ratio_range = [1.0, 1.5]
            image, depthmap, intrinsics = random_crop(
                image, depthmap, intrinsics, target_size=resolution, aspect_ratio_range=aspect_ratio_range, rng=rng
            )            
        elif self.aug_crop > 1:
                target_resolution += (
                    rng.integers(0, self.aug_crop)
                    if not self.seq_aug_crop
                    else self.delta_target_resolution
                )
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        target_w, target_h = resolution
        W, H = image.size
        if W < target_w or H < target_h:
            raise ValueError(
                f"target_size {resolution} exceeds image size {(W, H)}"
            )
        if (W, H) != (target_w, target_h):
            left = (W - target_w) // 2
            top = (H - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            image, depthmap, intrinsics = cropping.crop_image_depthmap(
                image, depthmap, intrinsics, (left, top, right, bottom)
            )

        return image, depthmap, intrinsics, pose


def is_good_type(key, v):
    """returns (is_good, err_msg)"""
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8, np.float64):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    print(f'transpose_to_landscape!{view["true_shape"]}')
    height, width = view["true_shape"]

    if width < height:
        # rectify portrait to landscape
        assert view["img"].shape == (3, height, width)
        view["img"] = view["img"].swapaxes(1, 2)

        assert view["valid_mask"].shape == (height, width)
        view["valid_mask"] = view["valid_mask"].swapaxes(0, 1)

        assert view["depthmap"].shape == (height, width)
        view["depthmap"] = view["depthmap"].swapaxes(0, 1)

        assert view["pts3d"].shape == (height, width, 3)
        view["pts3d"] = view["pts3d"].swapaxes(0, 1)

        # transpose x and y pixels
        view["camera_intrinsics"] = view["camera_intrinsics"][[1, 0, 2]]

        assert view["ray_map"].shape == (height, width, 6)
        view["ray_map"] = view["ray_map"].swapaxes(0, 1)

        assert view["sky_mask"].shape == (height, width)
        view["sky_mask"] = view["sky_mask"].swapaxes(0, 1)

        if "corres" in view:
            # transpose correspondences x and y
            view["corres"][0] = view["corres"][0][:, [1, 0]]
            view["corres"][1] = view["corres"][1][:, [1, 0]]
