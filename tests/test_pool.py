"""Tests for BlockPool."""
import multiprocessing as mp
import threading
import time

import numpy as np
import pytest

from nu_specter import BlockPool, SharedBlock
from nu_specter.exceptions import PoolExhaustedError


def test_basic_acquire_release():
    with BlockPool(shape=(10,), dtype=np.float32, capacity=2) as pool:
        slot = pool.acquire()
        assert slot.array.shape == (10,)
        ref_id = slot.handle.ref_id
        slot.close()
        pool.release(ref_id)


def test_pool_capacity_blocks():
    pool = BlockPool(shape=(5,), dtype=np.int32, capacity=2)

    s1 = pool.acquire()
    s2 = pool.acquire()

    with pytest.raises(PoolExhaustedError):
        pool.acquire(timeout=0.1)

    s1.close()
    pool.release(s1.handle.ref_id)
    s2.close()
    pool.release(s2.handle.ref_id)

    pool.shutdown()


def test_slot_data_survives_attach():
    pool = BlockPool(shape=(4, 4), dtype=np.float64, capacity=1)
    slot = pool.acquire()

    slot.array[:] = np.arange(16).reshape(4, 4)
    handle = slot.handle
    slot.close()

    # Consumer attaches
    consumer = SharedBlock.from_handle(handle)
    np.testing.assert_array_equal(consumer.array, np.arange(16).reshape(4, 4))
    consumer.close()

    pool.release(handle.ref_id)
    pool.shutdown()


def test_pool_reuse():
    """Verify the same slot can be reused across multiple send/receive cycles."""
    pool = BlockPool(shape=(3,), dtype=np.int32, capacity=1)

    for value in [10, 20, 30]:
        slot = pool.acquire()
        slot.array[:] = value
        handle = slot.handle
        slot.close()

        consumer = SharedBlock.from_handle(handle)
        assert list(consumer.array) == [value, value, value]
        consumer.close()

        pool.release(handle.ref_id)

    pool.shutdown()


def test_context_manager():
    with BlockPool(shape=(8,), dtype=np.uint8, capacity=3) as pool:
        assert repr(pool).startswith("BlockPool(")
