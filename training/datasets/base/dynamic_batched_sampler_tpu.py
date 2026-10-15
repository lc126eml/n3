import math
from typing import Iterator, List, Tuple

import numpy as np

from training.datasets.base.dynamic_batched_sampler import DynamicResolutionSampler


class DynamicResolutionSamplerTPU(DynamicResolutionSampler):
    """
    TPU/XLA variant of the dynamic batch sampler.

    The default sampler already reasons in terms of global batch size, but it does
    not shard the sampled global batch across workers. TPU workers must see
    disjoint, equal-sized local batches, so this wrapper enforces divisibility by
    world size and then slices the global batch by rank.
    """

    def __init__(
        self,
        *args,
        rank: int = 0,
        world_size: int = 1,
        dynamic_schedule: str = "batch",
        fix_batch_views: bool = True,
        epoch_res_num: int = 1,
        **kwargs,
    ):
        super().__init__(*args, world_size=world_size, **kwargs)
        self.rank = int(rank)
        self.dynamic_schedule = str(dynamic_schedule)
        self.fix_batch_views = bool(fix_batch_views)
        self.epoch_res_num = int(epoch_res_num)
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"rank must be in [0, {self.world_size}), got {self.rank}"
            )
        if self.dynamic_schedule not in {"batch", "epoch"}:
            raise ValueError(
                f"dynamic_schedule must be 'batch' or 'epoch', got {self.dynamic_schedule!r}"
            )

    def _adjust_batch_size_for_xla(self, global_batch_size: int) -> int:
        global_batch_size = max(1, int(global_batch_size))
        if self.world_size <= 1:
            return global_batch_size

        multiple = math.lcm(self.accum_steps, self.world_size)
        if global_batch_size < multiple:
            return multiple
        return max(multiple, (global_batch_size // multiple) * multiple)

    def _sample_global_batch_size(self, res_idx: int, view_size: int) -> int:
        res_h, res_w = self.resolutions[res_idx]
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
        dynamic_global_batch_size = self._adjust_batch_size_for_xla(
            dynamic_global_batch_size
        )
        return dynamic_global_batch_size

    def _sample_epoch_layout(self):
        """Draw the epoch's (resolutions, view_size) layout deterministically.

        Returns (res_indices, view_size, avg_global_batch_size):
        - res_indices: round-robin order of resolutions for this epoch. With
          epoch_res_num > 0 a random subset of that size; with epoch_res_num <= 0
          the full valid resolution set.
        - view_size: a single view size constant for the epoch when
          fix_batch_views is True, else None (sampled per batch).
        - avg_global_batch_size: mean cost-derived global batch size over the
          layout, used only by __len__ for progress reporting.
        """
        rng = np.random.default_rng(seed=self.seed + self.epoch)
        order = rng.permutation(self._valid_resolution_indices).tolist()
        if self.epoch_res_num > 0:
            res_indices = order[: min(self.epoch_res_num, len(order))]
        else:
            res_indices = order
        if not res_indices:
            raise RuntimeError("No valid resolutions available for epoch layout sampling")

        if self.fix_batch_views:
            view_cap = min(self._resolution_view_max[res_idx] for res_idx in res_indices)
            view_size = int(rng.integers(self.min_view_size, view_cap + 1))
        else:
            view_size = None

        batch_sizes = []
        for res_idx in res_indices:
            if view_size is not None:
                vs = view_size
            else:
                vs = max(
                    self.min_view_size,
                    int(round((self.min_view_size + self._resolution_view_max[res_idx]) / 2.0)),
                )
            batch_sizes.append(float(self._sample_global_batch_size(res_idx, vs)))
        avg_global_batch_size = sum(batch_sizes) / len(batch_sizes)
        return res_indices, view_size, avg_global_batch_size

    def __len__(self) -> int:
        if self.dynamic_schedule == "epoch":
            _, _, avg_global_batch_size = self._sample_epoch_layout()
            avg_batch_size_per_rank = avg_global_batch_size / self.world_size
            return math.ceil(self.num_samples_per_rank / avg_batch_size_per_rank)

        avg_res_cost = (
            sum(
                self._resolution_cost(*self.resolutions[res_idx])
                for res_idx in self._valid_resolution_indices
            )
            / len(self._valid_resolution_indices)
        )
        avg_view_sizes = []
        for res_idx in self._valid_resolution_indices:
            res_max_view = self._resolution_view_max[res_idx]
            avg_view_sizes.append((self.min_view_size + res_max_view) / 2.0)
        avg_view_size = sum(avg_view_sizes) / len(avg_view_sizes)
        avg_cost_per_sample = avg_view_size * avg_res_cost
        avg_global_batch_size = self.target_batch_cost / avg_cost_per_sample
        avg_global_batch_size = self._adjust_batch_size_for_xla(avg_global_batch_size)
        avg_batch_size_per_rank = avg_global_batch_size / self.world_size
        return math.ceil(self.num_samples_per_rank / avg_batch_size_per_rank)

    def __iter__(self) -> Iterator[List[Tuple[int, Tuple[int, int], int]]]:
        rng = np.random.default_rng(seed=self.seed + self.epoch)
        all_sample_indices = np.arange(self.len_dataset)
        rng.shuffle(all_sample_indices)

        if not self.drop_last:
            padding_size = self.total_size - len(all_sample_indices)
            if padding_size > 0:
                all_sample_indices = np.pad(all_sample_indices, (0, padding_size), "wrap")

        if self.debug_enumerate_batches:
            cursor = 0
            for res_idx in self._valid_resolution_indices:
                res_max_view = self._resolution_view_max[res_idx]
                for view_size in range(self.min_view_size, res_max_view + 1):
                    dynamic_global_batch_size = self._sample_global_batch_size(
                        res_idx, view_size
                    )
                    if self.len_dataset == 0:
                        continue
                    batch_indices = [
                        all_sample_indices[(cursor + i) % self.len_dataset]
                        for i in range(dynamic_global_batch_size)
                    ]
                    cursor = (cursor + dynamic_global_batch_size) % self.len_dataset
                    indices_for_this_rank = batch_indices[self.rank :: self.world_size]
                    if indices_for_this_rank:
                        yield [
                            (int(sample_idx), res_idx, view_size)
                            for sample_idx in indices_for_this_rank
                        ]
            return

        if self.dynamic_schedule == "epoch":
            epoch_res_indices, epoch_view_size, _ = self._sample_epoch_layout()
            samples_yielded = 0
            batch_count = 0
            while samples_yielded < len(all_sample_indices):
                res_idx = int(epoch_res_indices[batch_count % len(epoch_res_indices)])
                if epoch_view_size is not None:
                    view_size = epoch_view_size
                else:
                    res_max_view = self._resolution_view_max[res_idx]
                    view_size = int(rng.integers(self.min_view_size, res_max_view + 1))
                dynamic_global_batch_size = self._sample_global_batch_size(res_idx, view_size)
                remaining_samples = len(all_sample_indices) - samples_yielded
                if remaining_samples < dynamic_global_batch_size:
                    if self.drop_last or remaining_samples < self.world_size:
                        break
                    dynamic_global_batch_size = (remaining_samples // self.world_size) * self.world_size
                    if dynamic_global_batch_size <= 0:
                        break

                batch_end_idx = samples_yielded + dynamic_global_batch_size
                global_batch_indices = all_sample_indices[samples_yielded:batch_end_idx]
                indices_for_this_rank = global_batch_indices[self.rank :: self.world_size]
                batch_for_this_rank = [
                    (int(sample_idx), res_idx, view_size)
                    for sample_idx in indices_for_this_rank
                ]
                if batch_for_this_rank:
                    yield batch_for_this_rank
                samples_yielded += dynamic_global_batch_size
                batch_count += 1
            return

        samples_yielded = 0
        while samples_yielded < len(all_sample_indices):
            sampled_idx = int(rng.integers(len(self._valid_resolution_indices)))
            res_idx = self._valid_resolution_indices[sampled_idx]
            res_max_view = self._resolution_view_max[res_idx]
            view_size = int(rng.integers(self.min_view_size, res_max_view + 1))

            dynamic_global_batch_size = self._sample_global_batch_size(res_idx, view_size)
            remaining_samples = len(all_sample_indices) - samples_yielded
            if remaining_samples < dynamic_global_batch_size:
                if self.drop_last or remaining_samples < self.world_size:
                    break
                dynamic_global_batch_size = (remaining_samples // self.world_size) * self.world_size
                if dynamic_global_batch_size <= 0:
                    break

            batch_end_idx = samples_yielded + dynamic_global_batch_size
            global_batch_indices = all_sample_indices[samples_yielded:batch_end_idx]
            indices_for_this_rank = global_batch_indices[self.rank :: self.world_size]
            batch_for_this_rank = [
                (int(sample_idx), res_idx, view_size)
                for sample_idx in indices_for_this_rank
            ]
            if batch_for_this_rank:
                yield batch_for_this_rank
            samples_yielded += dynamic_global_batch_size
