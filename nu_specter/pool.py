"""
BlockPool: pre-allocates N fixed-size shared memory slots and recycles them.

For high-throughput pipelines where every message has the same shape/dtype,
the pool eliminates per-message allocation overhead. The producer acquires a
slot, fills it directly (zero-copy), sends the handle, and waits for an ACK
before reusing the slot.

    pool = BlockPool(shape=(1024, 1024), dtype=np.float32, capacity=4)

    # Producer:
    with pool.acquire() as slot:
        slot.array[:] = my_data       # write directly into shared memory
        queue.put(slot.handle)
        # context manager calls close(), but NOT unlink()

    # Consumer:
    block = SharedBlock.from_handle(queue.get())
    process(block.array)
    block.close()
    ack_queue.put(block.handle.ref_id)  # return slot to pool

    # Back in producer — call pool.release(ref_id) after receiving ACK.
    pool.release(ack_queue.get())

    # Shutdown:
    pool.shutdown()
"""
from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import numpy as np

from .block import SharedBlock
from .exceptions import PoolExhaustedError


class BlockPool:
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: "np.dtype | str",
        capacity: int = 4,
        name_prefix: Optional[str] = None,
    ) -> None:
        self._shape = tuple(shape)
        self._dtype = np.dtype(dtype)
        self._capacity = capacity

        # Semaphore controls how many slots are available
        self._sem = threading.Semaphore(capacity)

        # All pre-allocated blocks, keyed by ref_id
        self._blocks: Dict[str, SharedBlock] = {}
        # ref_ids of currently free slots
        self._free: list[str] = []
        self._lock = threading.Lock()

        # Pre-allocate all slots
        template = np.zeros(self._shape, dtype=self._dtype)
        for i in range(capacity):
            name = f"{name_prefix or 'shpool'}_{i}" if name_prefix else None
            block = SharedBlock.from_array(template, name=name)
            ref_id = block.handle.ref_id
            self._blocks[ref_id] = block
            self._free.append(ref_id)

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    def acquire(self, timeout: Optional[float] = None) -> SharedBlock:
        """
        Return a free slot as a SharedBlock. Blocks until one is available.

        The slot's array is writable — fill it directly to avoid any copy:
            slot = pool.acquire()
            slot.array[:] = my_data
            queue.put(slot.handle)
            slot.close()          # detach (does NOT unlink or return to pool)

        Call pool.release(ref_id) when the consumer ACKs.
        """
        acquired = self._sem.acquire(timeout=timeout)
        if not acquired:
            raise PoolExhaustedError(
                f"All {self._capacity} pool slots are in use (timeout={timeout}s)."
            )

        with self._lock:
            ref_id = self._free.pop()
            block = self._blocks[ref_id]

        # Re-open the mapping if a previous consumer closed it
        if block._closed:
            block = SharedBlock.from_handle(block.handle)
            block._is_owner = True  # pool allocated these blocks; it owns them
            # Replace in registry so we hold the live reference
            with self._lock:
                self._blocks[block.handle.ref_id] = block

        return block

    def release(self, ref_id: str) -> None:
        """
        Return a slot to the pool. Call this after receiving the consumer's ACK.
        ref_id must be the handle.ref_id of the acquired slot.
        """
        with self._lock:
            if ref_id not in self._blocks:
                return
            self._free.append(ref_id)
        self._sem.release()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Unlink all pre-allocated blocks. Call once from the owner process
        when the pool is no longer needed.
        """
        with self._lock:
            blocks = list(self._blocks.values())
            self._blocks.clear()
            self._free.clear()

        for block in blocks:
            block.unlink()

    def __enter__(self) -> "BlockPool":
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        with self._lock:
            free = len(self._free)
        return (
            f"BlockPool(shape={self._shape}, dtype={self._dtype}, "
            f"capacity={self._capacity}, free={free})"
        )
