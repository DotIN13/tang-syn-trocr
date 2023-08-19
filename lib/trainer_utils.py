import torch
from transformers import trainer_utils


def seed_worker(worker_id):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    # Check if torch is running distributed
    is_distributed = torch.distributed.is_available() and \
        torch.distributed.is_initialized()

    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers

    rank = torch.distributed.get_rank() if is_distributed else 0

    if rank == 0:
        worker_id_offset = 0
    else:
        worker_id_offset = num_workers * rank + worker_id

    worker_seed = torch.initial_seed() % 2**32 + worker_id_offset
    print(f"Rank {rank} worker {worker_id}: set seed to {worker_seed}")
    trainer_utils.set_seed(worker_seed)


trainer_utils.seed_worker = seed_worker
