"""ArrayHandle: the serializable descriptor passed between processes."""
from __future__ import annotations

import dataclasses
import os
import uuid
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class ArrayHandle:
    """
    A small, fully pickle-safe descriptor that represents a numpy array
    stored in shared memory. This is the only object that crosses process
    boundaries — no array data is ever serialized.

    The receiving process calls SharedBlock.from_handle(handle) to get a
    zero-copy numpy view of the data.
    """

    shm_name: str
    shape: Tuple[int, ...]
    dtype: str          # dtype.str preserves byte order, e.g. '<f4'
    strides: Optional[Tuple[int, ...]]  # None means C-contiguous
    nbytes: int         # total bytes of the shared memory block
    owner_pid: int
    ref_id: str         # UUID for resource tracking and ACK correlation

    def to_numpy(self, shm: SharedMemory) -> np.ndarray:
        """
        Construct a zero-copy numpy view onto an already-opened SharedMemory.
        """
        if shm.size != self.nbytes:
            from .exceptions import HandleMismatchError
            raise HandleMismatchError(
                f"SharedMemory size {shm.size} != handle.nbytes {self.nbytes}"
            )
        arr = np.frombuffer(shm.buf, dtype=np.dtype(self.dtype))
        arr = arr.reshape(self.shape)
        if self.strides is not None:
            arr = np.lib.stride_tricks.as_strided(arr, shape=self.shape, strides=self.strides)
        return arr

    @staticmethod
    def from_array(arr: np.ndarray, shm: SharedMemory) -> "ArrayHandle":
        """Create a handle describing arr already copied into shm."""
        return ArrayHandle(
            shm_name=shm.name,
            shape=tuple(arr.shape),
            dtype=arr.dtype.str,
            strides=None,  # we always store C-contiguous in shared memory
            nbytes=shm.size,
            owner_pid=os.getpid(),
            ref_id=str(uuid.uuid4()),
        )
