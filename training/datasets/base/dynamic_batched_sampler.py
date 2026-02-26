import numpy as np
import logging
import torch
import torch.utils
from torch.utils.data import BatchSampler, Sampler
import torch.utils.data
import math
from typing import List, Tuple, Iterator

class DynamicResolutionSampler(BatchSampler):
    """
    A BatchSampler that creates batches with dynamic properties for efficient training.

    This sampler ensures all items within a single batch share the same resolution
    and view size. It dynamically adjusts the number of samples per batch to keep the
    total number of pixels processed roughly constant.

    It is designed to work in both single-GPU and distributed training settings.

    Args:
        dataset: The dataset to sample from.
        resolutions (List[Tuple[int, int]]): A list of possible (height, width) resolutions.
        base_batch_size (int): The target batch size for the first resolution in the list.
        min_view_size (int): The minimum number of views for a sample.
        max_view_size (int): The maximum number of views for a sample.
        accum_steps (int): Gradient accumulation steps used by training. Dynamic
            global batch size is rounded to accumulation-friendly values.
        cost_batch_mode (str): Cost-estimation batch unit, either:
            "micro" (default; estimate using micro-batch size and map back to global),
            "global" (legacy; estimate directly from global batch size).
        patch_size (int): Patch size used for ViT token estimation.
        cost_mode (str): Batch sizing cost model. One of:
            "vit_tokens" (default; linear token count),
            "pixels" (legacy linear H*W),
            "vit_tokens" (linear token count),
            "vit_attention" (quadratic token count),
            "vit_mixed" (linear + normalized quadratic token cost).
        debug_enumerate_batches (bool): If True, print all (view_size, resolution)
            combinations with their computed batch size in __iter__, then terminate.
        drop_last (bool): If True, drop the last incomplete batch.
        world_size (int): The number of processes in the distributed training environment.
        rank (int): The rank of the current process in the distributed training environment.
        seed (int): The random seed for reproducibility.
    """

    def __init__(
        self,
        dataset,
        resolutions: List[Tuple[int, int]],
        base_batch_size: int,
        min_view_size: int,
        max_view_size: int,
        accum_steps: int = 1,
        cost_batch_mode: str = "micro",
        patch_size: int = 16,
        cost_mode: str = "vit_tokens",
        debug_enumerate_batches: bool = True,
        drop_last: bool = True,
        world_size: int = 1,
        seed: int = 777,
    ):
        if not resolutions:
            raise ValueError("The 'resolutions' list cannot be empty.")

        # BatchSampler does not have a 'data_source' argument in its __init__
        # so we don't call super().__init__()
        self.dataset = dataset
        self.resolutions = resolutions
        self.base_batch_size = base_batch_size
        self.min_view_size = min_view_size
        self.max_view_size = max_view_size
        self.accum_steps = int(accum_steps)
        self.cost_batch_mode = str(cost_batch_mode)
        self.patch_size = int(patch_size)
        self.cost_mode = str(cost_mode)
        self.debug_enumerate_batches = bool(debug_enumerate_batches)
        self.drop_last = drop_last
        self.world_size = world_size
        self.seed = seed
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if self.accum_steps <= 0:
            raise ValueError("accum_steps must be > 0")
        valid_batch_modes = {"micro", "global"}
        if self.cost_batch_mode not in valid_batch_modes:
            raise ValueError(
                f"Unsupported cost_batch_mode={self.cost_batch_mode!r}. "
                f"Expected one of {sorted(valid_batch_modes)}"
            )
        valid_cost_modes = {"pixels", "vit_tokens", "vit_attention", "vit_mixed"}
        if self.cost_mode not in valid_cost_modes:
            raise ValueError(
                f"Unsupported cost_mode={self.cost_mode!r}. "
                f"Expected one of {sorted(valid_cost_modes)}"
            )

        self.len_dataset = len(dataset)
        self.epoch = 0

        # Calculate the target batch cost used to determine dynamic batch sizes.
        base_res_h, base_res_w = self.resolutions[0]
        self._base_tokens = self._num_tokens(base_res_h, base_res_w)
        base_batch_for_cost = (
            math.ceil(self.base_batch_size / self.accum_steps)
            if self.cost_batch_mode == "micro"
            else self.base_batch_size
        )
        self.target_batch_cost = (
            base_batch_for_cost
            * self.min_view_size
            * self._resolution_cost(base_res_h, base_res_w)
        )
        self._base_target_batch_cost = float(self.target_batch_cost)
        self._target_batch_cost_discount = 1.0
        # Per-resolution view cap uses a less conservative linear-token proxy.
        # This better matches observed memory envelopes than mixed quadratic cost.
        self._target_cap_cost = (
            base_batch_for_cost
            * self.min_view_size
            * float(self._num_tokens(base_res_h, base_res_w))
        )
        self._base_target_cap_cost = float(self._target_cap_cost)
        self._resolution_view_max = self._build_resolution_view_max_cache()
        self._valid_resolution_indices = [
            res_idx
            for res_idx in range(len(self.resolutions))
            if self._resolution_view_max[res_idx] >= self.min_view_size
        ]
        if not self._valid_resolution_indices:
            raise ValueError(
                "No valid resolutions remain after applying per-resolution max view size constraints."
            )
        
        # Approximate number of samples per rank
        self.num_samples_per_rank = math.ceil(self.len_dataset / self.world_size)
        self.total_size = self.num_samples_per_rank * self.world_size

    def _num_tokens(self, h: int, w: int) -> int:
        return math.ceil(h / self.patch_size) * math.ceil(w / self.patch_size)

    def _resolution_cost(self, h: int, w: int) -> float:
        if self.cost_mode == "pixels":
            return float(h * w)

        n = float(self._num_tokens(h, w))
        if self.cost_mode == "vit_tokens":
            return n
        if self.cost_mode == "vit_attention":
            return n * n
        if self.cost_mode == "vit_mixed":
            # Mixed proxy keeps the scale closer to linear-token cost while
            # penalizing high resolutions where attention activation memory dominates.
            return n + (n * n) / max(1.0, float(self._base_tokens))
        raise RuntimeError(f"Unhandled cost_mode: {self.cost_mode}")

    def _build_resolution_view_max_cache(self) -> List[int]:
        resolution_view_max: List[int] = []
        for res_h, res_w in self.resolutions:
            res_cost = float(self._num_tokens(res_h, res_w))
            # Upper-bound view count so each sampled pair is at least batch_size >= 1
            # under the configured cost proxy.
            max_view_for_resolution = min(
                self.max_view_size,
                max(0, math.floor(self._target_cap_cost / res_cost)),
            )
            if self.debug_enumerate_batches and max_view_for_resolution < self.max_view_size:
                logging.warning(
                    "DynamicResolutionSampler capped max view size for resolution "
                    f"({res_h}, {res_w}) from {self.max_view_size} to {max_view_for_resolution}"
                )
                # max_view_for_resolution = 5
            resolution_view_max.append(max_view_for_resolution)
        return resolution_view_max

    def _adjust_batch_size_for_accum(self, global_batch_size: int) -> int:
        global_batch_size = max(1, int(global_batch_size))
        if self.accum_steps <= 1 or global_batch_size <= self.accum_steps:
            return global_batch_size
        return max(
            self.accum_steps,
            (global_batch_size // self.accum_steps) * self.accum_steps,
        )

    def _cost_batch_size_to_global(self, batch_size_for_cost: int) -> int:
        batch_size_for_cost = max(1, int(batch_size_for_cost))
        if self.cost_batch_mode == "micro":
            return batch_size_for_cost * self.accum_steps
        return batch_size_for_cost

    def set_target_batch_cost_discount(self, discount: float) -> None:
        discount = float(discount)
        if discount <= 0:
            raise ValueError("target batch cost discount must be > 0")
        self._target_batch_cost_discount = discount
        self.target_batch_cost = self._base_target_batch_cost * discount
        self._target_cap_cost = self._base_target_cap_cost * discount
        self._resolution_view_max = self._build_resolution_view_max_cache()
        self._valid_resolution_indices = [
            res_idx
            for res_idx in range(len(self.resolutions))
            if self._resolution_view_max[res_idx] >= self.min_view_size
        ]
        if not self._valid_resolution_indices:
            raise ValueError(
                "No valid resolutions remain after applying per-resolution max view size constraints."
            )

    def __len__(self) -> int:
        """
        Returns a statistically approximate number of batches per epoch for one rank.
        This is used by DataLoader progress bars.
        """
        # 1. Calculate the average resolution cost across sampled resolutions.
        avg_res_cost = (
            sum(
                self._resolution_cost(*self.resolutions[res_idx])
                for res_idx in self._valid_resolution_indices
            )
            / len(self._valid_resolution_indices)
        )

        # 2. Calculate the average view size, accounting for per-resolution caps.
        avg_view_sizes = []
        for res_idx in self._valid_resolution_indices:
            res_max_view = self._resolution_view_max[res_idx]
            avg_view_sizes.append((self.min_view_size + res_max_view) / 2.0)
        avg_view_size = sum(avg_view_sizes) / len(avg_view_sizes)
        
        # 3. Calculate the expected cost per sample on average
        avg_cost_per_sample = avg_view_size * avg_res_cost

        # 4. Calculate the expected dynamic batch size for the entire global batch
        # if avg_cost_per_sample == 0:
        #     return math.ceil(self.num_samples_per_rank / self.base_batch_size)
            
        avg_global_batch_size = self.target_batch_cost / avg_cost_per_sample
        
        # 5. Calculate the expected number of batches for a single rank
        # The number of samples per rank divided by the average batch size per rank
        avg_batch_size_per_rank = avg_global_batch_size / self.world_size
        
        # if avg_batch_size_per_rank == 0:
        #     # Avoid division by zero if average batch size is extremely small
        #     # Fall back to a simple estimate
        #     return self.num_samples_per_rank

        return math.ceil(self.num_samples_per_rank / avg_batch_size_per_rank)
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[Tuple[int, Tuple[int, int], int]]]:
        if self.debug_enumerate_batches:
            rng = np.random.default_rng(seed=self.seed + self.epoch)
            all_sample_indices = np.arange(self.len_dataset)
            rng.shuffle(all_sample_indices)
            cursor = 0
            for res_idx in self._valid_resolution_indices:
                res_h, res_w = self.resolutions[res_idx]
                res_max_view = self._resolution_view_max[res_idx]
                for view_size in range(self.min_view_size, res_max_view + 1):
                    cost_per_sample = view_size * self._resolution_cost(res_h, res_w)
                    dynamic_global_batch_size = (
                        max(1, math.floor(self.target_batch_cost / cost_per_sample))
                        if cost_per_sample > 0
                        else self.base_batch_size
                    )
                    dynamic_global_batch_size = self._cost_batch_size_to_global(
                        dynamic_global_batch_size
                    )
                    dynamic_global_batch_size = self._adjust_batch_size_for_accum(
                        dynamic_global_batch_size
                    )
                    # print(
                    #     "[DynamicResolutionSampler][debug] "
                    #     f"res_idx={res_idx}, resolution=({res_h}, {res_w}), "
                    #     f"view_size={view_size}, batch_size={dynamic_global_batch_size}"
                    # )
                    if self.len_dataset == 0:
                        continue
                    batch_indices = [
                        all_sample_indices[(cursor + i) % self.len_dataset]
                        for i in range(dynamic_global_batch_size)
                    ]
                    cursor = (cursor + dynamic_global_batch_size) % self.len_dataset
                    yield [
                        (int(sample_idx), res_idx, view_size)
                        for sample_idx in batch_indices
                    ]
            return

        rng = np.random.default_rng(seed=self.seed + self.epoch)
        
        # Generate a shuffled list of all sample indices
        all_sample_indices = np.arange(self.len_dataset)
        rng.shuffle(all_sample_indices)
        
        # Pad the list to be evenly divisible by world_size if drop_last is not True
        if not self.drop_last:
            padding_size = self.total_size - len(all_sample_indices)
            if padding_size > 0:
                all_sample_indices = np.pad(all_sample_indices, (0, padding_size), 'wrap')
        
        samples_yielded = 0
        while samples_yielded < len(all_sample_indices):
            # 1. Randomly choose properties for the next global batch
            sampled_idx = int(rng.integers(len(self._valid_resolution_indices)))
            res_idx = self._valid_resolution_indices[sampled_idx]
            resolution = self.resolutions[res_idx]
            res_max_view = self._resolution_view_max[res_idx]
            view_size = int(rng.integers(self.min_view_size, res_max_view + 1))
            res_h, res_w = resolution

            # 2. Calculate dynamic batch size for the chosen properties
            cost_per_sample = view_size * self._resolution_cost(res_h, res_w)
            dynamic_global_batch_size = (
                max(1, math.floor(self.target_batch_cost / cost_per_sample))
                if cost_per_sample > 0
                else self.base_batch_size
            )
            dynamic_global_batch_size = self._cost_batch_size_to_global(
                dynamic_global_batch_size
            )
            dynamic_global_batch_size = self._adjust_batch_size_for_accum(
                dynamic_global_batch_size
            )
            
            # print(resolution, view_size, pixels_per_sample, dynamic_global_batch_size, pixels_per_sample * dynamic_global_batch_size)
            # 3. Check for end of epoch
            remaining_samples = len(all_sample_indices) - samples_yielded
            if remaining_samples < dynamic_global_batch_size:
                if self.drop_last:
                    break
                else:
                    dynamic_global_batch_size = remaining_samples
            
            # 4. Get the indices for the full global batch
            batch_end_idx = samples_yielded + dynamic_global_batch_size
            global_batch_indices = all_sample_indices[samples_yielded:batch_end_idx]

            # 5. Distribute indices among ranks
            indices_for_this_rank = global_batch_indices
            
            # 6. Create the list of tuples for this rank's batch
            batch_for_this_rank = [
                (int(sample_idx), res_idx, view_size)
                for sample_idx in indices_for_this_rank
            ]
            # print(resolution, view_size, dynamic_global_batch_size)
            
            if batch_for_this_rank:
                 yield batch_for_this_rank

            samples_yielded += dynamic_global_batch_size

def round_by(total, multiple, up=False):
    if up:
        total = total + multiple - 1
    return (total // multiple) * multiple
