from training.datasets.utils.transforms import *
from training.datasets.blendedmvs_claude import BlendedMVS_Multi  # noqa
from training.datasets.hypersim import HyperSim_Multi  # noqa
from training.datasets.infinigen import Infinigen_Multi  # noqa
from training.datasets.seven_scene import SevenScenes_Multi  # noqa
from training.datasets.scannetpp2 import ScanNetpp_Multi  # noqa
from training.datasets.vkitti2 import VirtualKITTI2_Multi  # noqa
from training.datasets.dtu import DTU_Multi  # noqa
from training.datasets.eth3d import ETH3D_Multi  # noqa

from training.datasets.base.dynamic_batched_sampler_tpu import DynamicResolutionSamplerTPU


class UnevenDistributedEvalSampler:
    """
    Shard eval datasets without padding or dropping samples.

    Unlike torch DistributedSampler, this intentionally permits different ranks
    to have different numbers of eval batches. TrainerTPU only performs the
    distributed metric reduction after each rank finishes its local eval loop,
    so equal per-rank eval lengths are not required.
    """

    def __init__(self, dataset, num_replicas: int, rank: int) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        del epoch


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    persistent_workers=False,
    prefetch_factor=None,
    fixed_length=False,
    seed=None,
    accum_steps=1,
    debug_enumerate_batches=False,
    dynamic_schedule="batch",
    world_size=1,
    rank=0,
    distributed_sampler_mode="pad",
):
    import torch
    from torch.utils.data.distributed import DistributedSampler

    if isinstance(dataset, str):
        dataset = eval(dataset)

    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None

    loader_kwargs = dict(
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_mem,
        prefetch_factor=prefetch_factor,
    )

    if shuffle and hasattr(dataset, "_resolutions") and hasattr(dataset, "num_views"):
        num_of_views = dataset.num_views
        min_views = num_of_views if fixed_length else 4
        sampler = DynamicResolutionSamplerTPU(
            dataset=dataset,
            resolutions=dataset._resolutions,
            base_batch_size=batch_size,
            min_view_size=min_views,
            max_view_size=num_of_views,
            accum_steps=accum_steps,
            debug_enumerate_batches=debug_enumerate_batches,
            dynamic_schedule=dynamic_schedule,
            drop_last=drop_last,
            world_size=world_size,
            rank=rank,
            seed=seed if seed is not None else 777,
        )
        dataset.seed = seed
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            **loader_kwargs,
        )

    sampler = None
    if world_size > 1:
        if distributed_sampler_mode not in {"pad", "drop_last", "uneven"}:
            raise ValueError(
                "distributed_sampler_mode must be one of {'pad', 'drop_last', 'uneven'}; "
                f"got {distributed_sampler_mode!r}."
            )
        if not shuffle and distributed_sampler_mode == "uneven":
            sampler = UnevenDistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
        else:
            sampler_drop_last = drop_last if shuffle else distributed_sampler_mode == "drop_last"
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=sampler_drop_last,
                seed=seed if seed is not None else 0,
            )
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        **loader_kwargs,
    )
