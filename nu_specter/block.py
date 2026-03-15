"""
SharedBlock: RAII wrapper around a SharedMemory region and its numpy view.

Producer:
    block = SharedBlock.from_array(arr)
    queue.put(block.handle)       # send only the descriptor
    block.close()                 # detach mapping (data still lives in shm)
    # ... after consumer ACKs:
    block.unlink()                # destroy the OS block

Consumer:
    block = SharedBlock.from_handle(handle)
    process(block.array)          # zero-copy view
    block.close()                 # releases mapping; never calls unlink
"""
from __future__ import annotations

import struct
import sys
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np

from .exceptions import BlockClosedError, BlockAlreadyUnlinkedError
from .handle import ArrayHandle
from ._compat import is_64bit


class SharedBlock:
    def __init__(
        self,
        shm: SharedMemory,
        handle: ArrayHandle,
        *,
        is_owner: bool,
    ) -> None:
        self._shm = shm
        self._handle = handle
        self._is_owner = is_owner
        self._closed = False
        self._unlinked = False
        self._array: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_array(cls, arr: np.ndarray, name: Optional[str] = None) -> "SharedBlock":
        """
        Allocate new shared memory and copy arr into it.

        The caller owns this block and is responsible for calling unlink()
        once all consumers have closed it (or delegate to SharedQueue).
        """
        if arr.size == 0:
            raise ValueError("Cannot create a SharedBlock from a zero-size array.")

        if not is_64bit() and arr.nbytes > 2 * 1024**3:
            raise ValueError("Array exceeds 2 GB limit on 32-bit platforms.")

        # Always store C-contiguous data so receivers can reshape reliably.
        contiguous = np.ascontiguousarray(arr)
        nbytes = contiguous.nbytes

        shm = SharedMemory(name=name, create=True, size=nbytes)
        # Copy array data into shared memory
        dest = np.frombuffer(shm.buf, dtype=contiguous.dtype).reshape(contiguous.shape)
        np.copyto(dest, contiguous)

        handle = ArrayHandle.from_array(contiguous, shm)
        block = cls(shm, handle, is_owner=True)

        from .resource_tracker import ResourceTracker
        ResourceTracker.get().register(block)

        return block

    @classmethod
    def from_handle(cls, handle: ArrayHandle) -> "SharedBlock":
        """
        Attach to an existing shared memory block described by handle.
        Returns a zero-copy view. Does NOT own the block; never calls unlink().
        """
        from .exceptions import HandleMismatchError

        shm = SharedMemory(name=handle.shm_name, create=False)
        if shm.size != handle.nbytes:
            shm.close()
            raise HandleMismatchError(
                f"SharedMemory size {shm.size} != handle.nbytes {handle.nbytes} "
                f"for block {handle.shm_name!r}"
            )
        return cls(shm, handle, is_owner=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def handle(self) -> ArrayHandle:
        return self._handle

    @property
    def array(self) -> np.ndarray:
        """Zero-copy numpy view of the shared memory. Raises BlockClosedError if closed."""
        if self._closed:
            raise BlockClosedError(
                f"Cannot access array of a closed SharedBlock (ref_id={self._handle.ref_id})"
            )
        if self._array is None:
            self._array = self._handle.to_numpy(self._shm)
        return self._array

    def close(self) -> None:
        """
        Release this process's mapping of the shared memory.
        Idempotent. Does NOT destroy the underlying block.
        """
        if self._closed:
            return
        self._closed = True
        self._array = None  # drop the numpy view before closing the buffer
        self._shm.close()

    def unlink(self) -> None:
        """
        Destroy the OS-level shared memory block.
        Must be called exactly once, from the owning process, after all
        consumers have closed their mappings.
        Idempotent after the first successful call.
        """
        if self._unlinked:
            return
        if not self._is_owner:
            raise PermissionError(
                f"Only the owning process may unlink a SharedBlock "
                f"(ref_id={self._handle.ref_id}, owner_pid={self._handle.owner_pid})"
            )
        self._unlinked = True
        if not self._closed:
            self.close()
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass  # already gone — that's fine

        from .resource_tracker import ResourceTracker
        ResourceTracker.get().deregister(self._handle.ref_id)

    def __enter__(self) -> "SharedBlock":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return (
            f"SharedBlock(name={self._handle.shm_name!r}, "
            f"shape={self._handle.shape}, dtype={self._handle.dtype}, "
            f"owner={self._is_owner}, status={status})"
        )
