"""Tests for SharedBlock."""
import multiprocessing as mp

import numpy as np
import pytest

from nu_specter import SharedBlock
from nu_specter.exceptions import BlockClosedError, HandleMismatchError


def test_roundtrip_basic():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    block = SharedBlock.from_array(arr)
    np.testing.assert_array_equal(block.array, arr)
    block.unlink()


def test_roundtrip_2d():
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    block = SharedBlock.from_array(arr)
    np.testing.assert_array_equal(block.array, arr)
    block.unlink()


def test_attach_via_handle():
    arr = np.random.rand(100).astype(np.float32)
    producer = SharedBlock.from_array(arr)
    handle = producer.handle
    producer.close()

    consumer = SharedBlock.from_handle(handle)
    np.testing.assert_array_almost_equal(consumer.array, arr)
    consumer.close()

    producer.unlink()


def test_close_raises_on_array_access():
    block = SharedBlock.from_array(np.ones(5))
    block.close()
    with pytest.raises(BlockClosedError):
        _ = block.array


def test_close_is_idempotent():
    block = SharedBlock.from_array(np.ones(5))
    block.close()
    block.close()  # should not raise
    block.unlink()


def test_unlink_is_idempotent():
    block = SharedBlock.from_array(np.ones(5))
    block.unlink()
    block.unlink()  # should not raise


def test_consumer_cannot_unlink():
    arr = np.ones(10)
    producer = SharedBlock.from_array(arr)
    consumer = SharedBlock.from_handle(producer.handle)
    with pytest.raises(PermissionError):
        consumer.unlink()
    consumer.close()
    producer.unlink()


def test_non_contiguous_array_preserved():
    """Non-contiguous input should be stored contiguously but logical values preserved."""
    arr = np.arange(20)[::2]  # stride of 2, shape (10,)
    assert not arr.flags["C_CONTIGUOUS"]
    block = SharedBlock.from_array(arr)
    np.testing.assert_array_equal(block.array, arr)
    block.unlink()


def test_context_manager():
    arr = np.ones((4, 4))
    with SharedBlock.from_array(arr) as block:
        np.testing.assert_array_equal(block.array, arr)
        block.unlink()
    # block.close() was called by __exit__; array access should fail
    with pytest.raises(BlockClosedError):
        _ = block.array


def test_zero_size_raises():
    with pytest.raises(ValueError):
        SharedBlock.from_array(np.array([]))


def _worker_read(handle_bytes, result_queue):
    """Child process worker: attach to shm, compute sum, signal done."""
    import pickle
    handle = pickle.loads(handle_bytes)
    block = SharedBlock.from_handle(handle)
    result_queue.put(block.array.sum())
    block.close()


def test_cross_process_sharing():
    arr = np.ones((100, 100), dtype=np.float32)
    block = SharedBlock.from_array(arr)

    import pickle
    handle_bytes = pickle.dumps(block.handle)

    result_queue = mp.Queue()
    p = mp.Process(target=_worker_read, args=(handle_bytes, result_queue))
    p.start()
    p.join(timeout=10)

    assert p.exitcode == 0
    result = result_queue.get_nowait()
    assert result == pytest.approx(10000.0)

    block.unlink()
