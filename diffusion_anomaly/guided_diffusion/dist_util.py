"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

def setup_dist():
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        master_ip = os.environ['MASTER_IP'] if 'MASTER_IP' in os.environ else os.environ['AZ_BATCHAI_JOB_MASTER_NODE_IP']
        master_port = '6688'

        print('OpenMPI mode.')

        print('world size: ', world_size)
        print('world rank: ', world_rank)
        print('local rank: ', local_rank)
        print('master_ip: ', master_ip)

        master_uri = "tcp://%s:%s" % (master_ip, master_port)
        dist.init_process_group(
            backend="nccl",
            init_method=master_uri,
            rank=world_rank,
            world_size=world_size
        )
        
        th.cuda.set_device(local_rank)
        print('Rank: %d' % dist.get_rank())
    else:
        dist.init_process_group(backend="nccl")
        th.cuda.set_device(dist.get_rank())
        print('Rank: %d' % dist.get_rank())


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    return th.load(path, map_location='cpu')
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()