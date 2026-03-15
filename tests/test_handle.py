"""Tests for ArrayHandle."""
import pickle

import numpy as np
import pytest

from nu_specter import ArrayHandle, SharedBlock
from nu_specter.exceptions import HandleMismatchError


def test_handle_is_picklable():
    arr = np.ones((5, 5), dtype=np.float32)
    block = SharedBlock.from_array(arr)
    handle = block.handle

    pickled = pickle.dumps(handle)
    restored = pickle.loads(pickled)

    assert restored.shm_name == handle.shm_name
    assert restored.shape == handle.shape
    assert restored.dtype == handle.dtype
    assert restored.ref_id == handle.ref_id

    block.unlink()


def test_handle_mismatch_raises():
    """Manually constructing a handle with wrong nbytes should raise on attach."""
    arr = np.ones(10, dtype=np.float32)
    block = SharedBlock.from_array(arr)
    handle = block.handle

    bad_handle = ArrayHandle(
        shm_name=handle.shm_name,
        shape=(999,),
        dtype=handle.dtype,
        strides=None,
        nbytes=handle.nbytes + 1,  # intentionally wrong
        owner_pid=handle.owner_pid,
        ref_id=handle.ref_id,
    )

    with pytest.raises(HandleMismatchError):
        SharedBlock.from_handle(bad_handle)

    block.unlink()


def test_dtype_str_preserves_byte_order():
    """dtype.str (e.g., '<f4') is used, not dtype.name ('float32')."""
    arr = np.ones(5, dtype=np.float32)
    block = SharedBlock.from_array(arr)
    # dtype.str includes endianness marker
    assert block.handle.dtype in ("<f4", ">f4", "=f4", "|f4")
    block.unlink()
