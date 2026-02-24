import os
import time
import torch

def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current gpu.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed_rank = int(os.environ.get("RANK", "0"))
    return local_rank, distributed_rank
