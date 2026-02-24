import numpy as np
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
        self.drop_last = drop_last
        self.world_size = world_size
        self.seed = seed

        self.len_dataset = len(dataset)
        self.epoch = 0

        # Calculate the target total pixels for a batch, used to determine dynamic batch sizes
        base_res_h, base_res_w = self.resolutions[0]
        self.target_batch_pixels = (
            self.base_batch_size * self.max_view_size * base_res_h * base_res_w
        )
        
        # Approximate number of samples per rank
        self.num_samples_per_rank = math.ceil(self.len_dataset / self.world_size)
        self.total_size = self.num_samples_per_rank * self.world_size

    def __len__(self) -> int:
        """
        Returns a statistically approximate number of batches per epoch for one rank.
        This is used by DataLoader progress bars.
        """
        # 1. Calculate the average number of pixels across all possible resolutions
        avg_res_pixels = sum(h * w for h, w in self.resolutions) / len(self.resolutions)

        # 2. Calculate the average view size
        # This assumes view sizes are chosen uniformly from the range [min, max]
        avg_view_size = (self.min_view_size + self.max_view_size) / 2
        
        # 3. Calculate the expected pixels per sample on average
        avg_pixels_per_sample = avg_view_size * avg_res_pixels

        # 4. Calculate the expected dynamic batch size for the entire global batch
        # if avg_pixels_per_sample == 0:
        #     return math.ceil(self.num_samples_per_rank / self.base_batch_size)
            
        avg_global_batch_size = self.target_batch_pixels / avg_pixels_per_sample
        
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
            res_idx = rng.integers(len(self.resolutions))
            resolution = self.resolutions[res_idx]
            view_size = rng.integers(self.min_view_size, self.max_view_size + 1)
            res_h, res_w = resolution

            # 2. Calculate dynamic batch size for the chosen properties
            pixels_per_sample = view_size * res_h * res_w
            dynamic_global_batch_size = max(1, math.floor(self.target_batch_pixels / pixels_per_sample)) if pixels_per_sample > 0 else self.base_batch_size
            
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
