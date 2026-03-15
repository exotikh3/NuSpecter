"""
SharedQueue: drop-in multiprocessing.Queue replacement for numpy arrays.

Instead of pickling array data, put() stores the array in shared memory and
sends only an ArrayHandle over the underlying queue. get() reconstructs a
zero-copy numpy view in the receiving process.

Lifecycle is automatic:
  - put() allocates a SharedBlock and registers it.
  - get() wraps the received block so that close() on the consumer side
    posts an ACK to an internal ack_queue.
  - A background thread in the producer process drains ACKs and calls
    unlink() on acknowledged blocks.

Basic usage:
    q = SharedQueue()

    def worker(q):
        block = q.get()
        result = block.array.sum()
        block.close()          # releases mapping + sends ACK
        return result

    p = mp.Process(target=worker, args=(q,))
    p.start()
    q.put(np.random.rand(1000, 1000))
    p.join()
    q.close()
"""
from __future__ import annotations

import multiprocessing
import threading
from typing import Optional

import numpy as np

from .block import SharedBlock
from .handle import ArrayHandle


class SharedQueue:
    def __init__(
        self,
        maxsize: int = 0,
    ) -> None:
        self._data_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=maxsize)
        self._ack_queue: multiprocessing.Queue = multiprocessing.Queue()

        # In-flight blocks: ref_id -> SharedBlock (in the producer process)
        self._in_flight: dict[str, SharedBlock] = {}
        self._in_flight_lock = threading.Lock()

        # Background thread drains ACKs and unlinks acknowledged blocks
        self._ack_thread = threading.Thread(
            target=self._ack_loop, daemon=True, name="SharedQueue-ack"
        )
        self._ack_thread.start()

        self._closed = False

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------

    def put(
        self,
        arr,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Copy arr into shared memory and enqueue only its handle.
        The underlying queue carries a tiny descriptor, not the array data.
        Non-ndarray values (e.g. None sentinel) are passed through as-is.
        """
        if self._closed:
            raise ValueError("SharedQueue is closed.")

        if not isinstance(arr, np.ndarray):
            self._data_queue.put(arr, block=block, timeout=timeout)
            return

        shared_block = SharedBlock.from_array(arr)

        with self._in_flight_lock:
            self._in_flight[shared_block.handle.ref_id] = shared_block

        # Detach producer's mapping — the data lives in shm until ACK arrives
        shared_block.close()

        self._data_queue.put(shared_block.handle, block=block, timeout=timeout)

    def _ack_loop(self) -> None:
        """Background thread: drain ACKs and unlink acknowledged blocks."""
        while True:
            try:
                ref_id = self._ack_queue.get(timeout=0.1)
            except Exception:
                if self._closed:
                    break
                continue

            if ref_id is None:  # sentinel
                break

            with self._in_flight_lock:
                block = self._in_flight.pop(ref_id, None)

            if block is not None:
                block.unlink()

    # ------------------------------------------------------------------
    # Consumer side
    # ------------------------------------------------------------------

    def get(
        self,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> SharedBlock:
        """
        Receive an ArrayHandle and return a SharedBlock with a zero-copy
        numpy view. The caller MUST call block.close() when done, which
        releases the mapping and signals the producer to free shared memory.
        """
        item = self._data_queue.get(block=block, timeout=timeout)

        if not isinstance(item, ArrayHandle):
            return item  # pass-through for sentinels (e.g. None)

        shared_block = SharedBlock.from_handle(item)

        # Wrap close() so it also posts an ACK
        ack_queue = self._ack_queue
        ref_id = item.ref_id
        original_close = shared_block.close

        def _close_with_ack() -> None:
            original_close()
            ack_queue.put(ref_id)

        shared_block.close = _close_with_ack  # type: ignore[method-assign]
        return shared_block

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Stop accepting new puts, drain pending ACKs, and unlink any
        remaining in-flight blocks. Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True

        # Signal background thread to stop after draining
        self._ack_queue.put(None)
        self._ack_thread.join(timeout=5.0)

        # Unlink anything that never got ACK'd (e.g., worker crashed)
        with self._in_flight_lock:
            remaining = list(self._in_flight.values())
            self._in_flight.clear()

        for b in remaining:
            b.unlink()

        self._data_queue.close()
        self._ack_queue.close()

    # ------------------------------------------------------------------
    # Pickle support (consumer side only needs the two mp.Queues)
    # ------------------------------------------------------------------

    def __getstate__(self):
        return {
            "data_queue": self._data_queue,
            "ack_queue": self._ack_queue,
        }

    def __setstate__(self, state):
        self._data_queue = state["data_queue"]
        self._ack_queue = state["ack_queue"]
        self._in_flight = {}
        self._in_flight_lock = threading.Lock()
        self._ack_thread = None  # consumer doesn't run the ACK loop
        self._closed = False

    def join_thread(self) -> None:
        self._data_queue.join_thread()

    def __enter__(self) -> "SharedQueue":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def qsize(self) -> int:
        return self._data_queue.qsize()

    def empty(self) -> bool:
        return self._data_queue.empty()
