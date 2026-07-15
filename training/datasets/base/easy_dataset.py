# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import numpy as np
from training.datasets.base.dynamic_batched_sampler import DynamicResolutionSampler
# from datasets.base.batched_sampler import (
#     BatchedRandomSampler,
#     CustomRandomSampler,
# )
import torch


class EasyDataset:
    """a dataset that you can easily resize and combine.
    Examples:
    ---------
        2 * dataset ==> duplicate each element 2x

        10 @ dataset ==> set the size to 10 (random sampling, duplicates if necessary)

        dataset1 + dataset2 ==> concatenate datasets
    """

    def __add__(self, other):
        return CatDataset([self, other])

    def __rmul__(self, factor):
        return MulDataset(factor, self)

    def __rmatmul__(self, factor):
        return ResizedDataset(factor, self)

    def set_epoch(self, epoch):
        pass  # nothing to do by default

    def set_seed(self, seed):
        self.seed = seed

    # def make_sampler(
    #     self, batch_size, shuffle=True, drop_last=True, world_size=1, rank=0, fixed_length=False, seed=None
    # ):
    #     if not (shuffle):
    #         raise NotImplementedError()  # cannot deal yet
    #     num_of_aspect_ratios = len(self._resolutions)
    #     num_of_views = self.num_views
    #     sampler = CustomRandomSampler(
    #         self,
    #         batch_size,
    #         num_of_aspect_ratios,
    #         4 if not fixed_length else num_of_views,
    #         num_of_views,
    #         world_size,
    #         warmup=1,
    #         drop_last=drop_last,
    #         seed=seed
    #     )
    #     return BatchedRandomSampler(sampler, batch_size, drop_last)
    
    def make_sampler(
        self,
        batch_size,
        shuffle=True,
        drop_last=True,
        world_size=1,
        rank=0,
        fixed_length=False,
        seed=None,
        accum_steps=1,
        debug_enumerate_batches=False,
        resolution_cost_power=1.0,
    ):
        """
        Creates and returns a DynamicResolutionSampler which acts as a batch sampler.
        """
        if not shuffle:
            # The sampler is inherently random, so sequential mode is not supported.
            raise NotImplementedError("DynamicResolutionSampler does not support non-shuffled mode.")
        
        # These properties would be defined on the class that contains this method
        # e.g., self._resolutions = [(224, 224), (256, 256), ...]
        # e.g., self.num_views = 8
        resolutions = self._resolutions 
        num_of_views = self.num_views

        # Determine the range of views to sample from
        min_views = num_of_views if fixed_length else 4 # Example: Use 4 as a minimum if not fixed
        max_views = num_of_views

        # Instantiate the new sampler directly. It will handle batching.
        batch_sampler = DynamicResolutionSampler(
            dataset=self,
            resolutions=resolutions,
            base_batch_size=batch_size,
            min_view_size=min_views,
            max_view_size=max_views,
            accum_steps=accum_steps,
            resolution_cost_power=resolution_cost_power,
            debug_enumerate_batches=debug_enumerate_batches,
            drop_last=drop_last,
            world_size=world_size,
            seed=seed
        )

        return batch_sampler


class MulDataset(EasyDataset):
    """Artifically augmenting the size of a dataset."""

    multiplicator: int

    def __init__(self, multiplicator, dataset):
        assert isinstance(multiplicator, int) and multiplicator > 0
        self.multiplicator = multiplicator
        self.dataset = dataset

    def __len__(self):
        return self.multiplicator * len(self.dataset)

    def __repr__(self):
        return f"{self.multiplicator}*{repr(self.dataset)}"

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, *metadata = idx
            return self.dataset[(idx // self.multiplicator, *metadata)]
        else:
            return self.dataset[idx // self.multiplicator]

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def set_seed(self, seed):
        self.seed = seed
        self.dataset.set_seed(seed)

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    @property
    def num_views(self):
        return self.dataset.num_views


class ResizedDataset(EasyDataset):
    """Artifically changing the size of a dataset."""

    new_size: int

    def __init__(self, new_size, dataset):
        assert isinstance(new_size, int) and new_size > 0
        self.new_size = new_size
        self.dataset = dataset
        # The mapping has a fixed size, so keep it in shared memory. Persistent
        # DataLoader workers then observe in-place epoch updates made by the
        # main process instead of retaining their initial mapping copy.
        self._idxs_mapping = torch.empty(new_size, dtype=torch.int64).share_memory_()
        self._mapping_epoch = torch.full((), -1, dtype=torch.int64).share_memory_()

    def __len__(self):
        return self.new_size

    def __repr__(self):
        size_str = str(self.new_size)
        for i in range((len(size_str) - 1) // 3):
            sep = -4 * i - 3
            size_str = size_str[:sep] + "_" + size_str[sep:]
        return f"{size_str} @ {repr(self.dataset)}"

    def set_epoch(self, epoch):
        # this random shuffle only depends on the epoch
        epoch = int(epoch)
        rng = np.random.default_rng(seed=epoch + 777)

        # shuffle all indices
        perm = rng.permutation(len(self.dataset))

        # rotary extension until target size is met
        shuffled_idxs = np.concatenate(
            [perm] * (1 + (len(self) - 1) // len(self.dataset))
        )
        mapping = torch.as_tensor(
            shuffled_idxs[: self.new_size], dtype=torch.int64
        )
        self._idxs_mapping.copy_(mapping)
        # Publish the epoch only after the complete mapping has been copied.
        self._mapping_epoch.fill_(epoch)
        self.dataset.set_epoch(epoch)

        assert len(self._idxs_mapping) == self.new_size

    def __getitem__(self, idx):
        assert self._mapping_epoch.item() >= 0, (
            "You need to call dataset.set_epoch() to use ResizedDataset.__getitem__()"
        )
        if isinstance(idx, tuple):
            idx, *metadata = idx
            mapped_idx = int(self._idxs_mapping[idx].item())
            return self.dataset[(mapped_idx, *metadata)]
        else:
            return self.dataset[int(self._idxs_mapping[idx].item())]

    def set_seed(self, seed):
        self.seed = seed
        self.dataset.set_seed(seed)

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    @property
    def num_views(self):
        return self.dataset.num_views


class CatDataset(EasyDataset):
    """Concatenation of several datasets"""

    def __init__(self, datasets):
        for dataset in datasets:
            assert isinstance(dataset, EasyDataset)
        self.datasets = datasets
        self._cum_sizes = np.cumsum([len(dataset) for dataset in datasets])

    def __len__(self):
        return self._cum_sizes[-1]

    def __repr__(self):
        # remove uselessly long transform
        return " + ".join(
            repr(dataset).replace(
                ",transform=Compose( ToTensor() Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))",
                "",
            )
            for dataset in self.datasets
        )

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def set_seed(self, seed):
        self.seed = seed
        for dataset in self.datasets:
            dataset.set_seed(seed)

    def __getitem__(self, idx):
        metadata = ()
        if isinstance(idx, tuple):
            idx, *metadata = idx

        if not (0 <= idx < len(self)):
            raise IndexError()

        db_idx = np.searchsorted(self._cum_sizes, idx, "right")
        dataset = self.datasets[db_idx]
        new_idx = idx - (self._cum_sizes[db_idx - 1] if db_idx > 0 else 0)

        if metadata:
            new_idx = (new_idx, *metadata)
        return dataset[new_idx]

    @property
    def _resolutions(self):
        resolutions = self.datasets[0]._resolutions
        for dataset in self.datasets[1:]:
            assert tuple(dataset._resolutions) == tuple(resolutions)
        return resolutions

    @property
    def num_views(self):
        num_views = self.datasets[0].num_views
        for dataset in self.datasets[1:]:
            assert dataset.num_views == num_views
        return num_views
