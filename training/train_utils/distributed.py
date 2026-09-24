import os
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from train_utils.priority_lock import PriorityLock


def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current gpu.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed_rank = int(os.environ.get("RANK", "0"))
    return local_rank, distributed_rank


def _distributed_worker(rank, config, world_size, rendezvous_file):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    cfg = OmegaConf.create(config)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend=cfg.distributed.backend,
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=cfg.distributed.timeout_mins),
    )
    try:
        from trainer import Trainer

        Trainer(cfg).run()
    finally:
        dist.destroy_process_group()


def run_distributed(cfg):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("distributed.enabled requires at least two visible CUDA devices")

    # Resolve the timestamp once so every worker writes to the same run folder.
    if cfg.logging.get("run_folder_name") is not None:
        cfg.logging.run_folder_name = str(cfg.logging.run_folder_name)
    config = OmegaConf.to_container(cfg, resolve=False)
    lock_priority = int(cfg.get("gpu_lock_priority", 10))
    lock = PriorityLock(lock_dir="/tmp/gpu.lock", priority=lock_priority) if lock_priority > 0 else None
    if lock is not None:
        lock.acquire()
    try:
        with tempfile.TemporaryDirectory(prefix="n3r_ddp_") as rendezvous_dir:
            mp.spawn(
                _distributed_worker,
                args=(config, world_size, os.path.join(rendezvous_dir, "rendezvous")),
                nprocs=world_size,
                join=True,
            )
    finally:
        if lock is not None:
            lock.release()
