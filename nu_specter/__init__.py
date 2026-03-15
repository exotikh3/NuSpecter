"""
nu_specter
==========
Zero-copy numpy array sharing between Python processes via shared memory.

Instead of pickling array data across process boundaries, this library stores
arrays in POSIX shared memory and passes lightweight handles. The receiving
process attaches to the same memory region and gets a numpy view with no
deserialization overhead.

Quick start
-----------
    import numpy as np
    import multiprocessing as mp
    from nu_specter import SharedQueue

    def worker(q_in, results):
        while True:
            block = q_in.get()
            if block is None:
                break
            results.put(block.array.mean())
            block.close()   # required: releases mapping and ACKs producer

    if __name__ == "__main__":
        q = SharedQueue()
        results = mp.Queue()
        p = mp.Process(target=worker, args=(q, results))
        p.start()

        for _ in range(10):
            q.put(np.random.rand(1024, 1024).astype(np.float32))

        q.put(None)   # sentinel (plain None is passed through as-is)
        p.join()
        q.close()

For fixed-shape high-throughput pipelines, use BlockPool to eliminate
per-message allocation overhead.
"""
from .handle import ArrayHandle
from .block import SharedBlock
from .pool import BlockPool
from .queue import SharedQueue
from .exceptions import (
    SharedMemoryError,
    BlockClosedError,
    BlockAlreadyUnlinkedError,
    PoolExhaustedError,
    HandleMismatchError,
)

__all__ = [
    "ArrayHandle",
    "SharedBlock",
    "BlockPool",
    "SharedQueue",
    "SharedMemoryError",
    "BlockClosedError",
    "BlockAlreadyUnlinkedError",
    "PoolExhaustedError",
    "HandleMismatchError",
]

__version__ = "0.1.0"
