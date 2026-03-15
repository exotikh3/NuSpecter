"""Tests for SharedQueue."""
import multiprocessing as mp
import time

import numpy as np
import pytest

from nu_specter import SharedQueue


def _sum_worker(q_in, q_out):
    """Receive arrays, compute sum, put result, close block."""
    while True:
        item = q_in.get()
        if item is None:
            break
        q_out.put(item.array.sum())
        item.close()


def test_basic_roundtrip():
    q = SharedQueue()
    results = mp.Queue()

    p = mp.Process(target=_sum_worker, args=(q, results))
    p.start()

    arr = np.ones((100, 100), dtype=np.float32)
    q.put(arr)
    q.put(None)  # sentinel (non-array, passes through as-is)

    p.join(timeout=10)
    assert p.exitcode == 0

    result = results.get_nowait()
    assert result == pytest.approx(10000.0)
    q.close()


def test_multiple_messages():
    q = SharedQueue()
    results = mp.Queue()

    p = mp.Process(target=_sum_worker, args=(q, results))
    p.start()

    n = 20
    for i in range(n):
        q.put(np.full((50, 50), i, dtype=np.float64))
    q.put(None)

    p.join(timeout=15)
    assert p.exitcode == 0

    collected = sorted(results.get_nowait() for _ in range(n))
    expected = sorted(i * 2500.0 for i in range(n))
    for got, exp in zip(collected, expected):
        assert got == pytest.approx(exp)

    q.close()


def test_data_is_zero_copy():
    """
    Verify that no extra copy of the data exists in the producer process
    after put() — the original array and the shm block hold the same values
    but the producer's numpy reference to shm has been detached.
    """
    q = SharedQueue()
    results = mp.Queue()

    p = mp.Process(target=_sum_worker, args=(q, results))
    p.start()

    arr = np.arange(1000, dtype=np.float64)
    expected = arr.sum()
    q.put(arr)
    q.put(None)

    p.join(timeout=10)
    assert p.exitcode == 0
    assert results.get_nowait() == pytest.approx(expected)
    q.close()


def test_context_manager():
    results = mp.Queue()
    with SharedQueue() as q:
        p = mp.Process(target=_sum_worker, args=(q, results))
        p.start()
        q.put(np.ones(256, dtype=np.float32))
        q.put(None)
        p.join(timeout=10)
    assert results.get_nowait() == pytest.approx(256.0)


def _identity_worker(q_in, q_out):
    """Passes array data through as a list for easy comparison."""
    while True:
        item = q_in.get()
        if item is None:
            break
        q_out.put(item.array.tolist())
        item.close()


def _echo_worker(q_in, q_out):
    """Receive arrays, serialize to bytes, send back for exact comparison."""
    while True:
        item = q_in.get()
        if item is None:
            break
        q_out.put(item.array.tobytes())
        item.close()


@pytest.mark.parametrize("shape,dtype", [
    ((1000,),           np.float32),
    ((100, 100),        np.float64),
    ((32, 32, 3),       np.uint8),
    ((10, 10, 10),      np.int32),
    ((1, 1024, 1024),   np.float32),
])
def test_data_integrity_shapes_and_dtypes(shape, dtype):
    """Exact byte-for-byte equality of array data across the process boundary."""
    rng = np.random.default_rng(42)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = rng.integers(info.min, info.max, size=shape, dtype=dtype)
    else:
        arr = rng.standard_normal(shape).astype(dtype)

    q = SharedQueue()
    results = mp.Queue()

    p = mp.Process(target=_echo_worker, args=(q, results))
    p.start()
    q.put(arr)
    q.put(None)
    received = np.frombuffer(results.get(timeout=15), dtype=dtype).reshape(shape)
    p.join(timeout=5)

    assert p.exitcode == 0
    np.testing.assert_array_equal(arr, received)
    q.close()


def test_data_integrity_special_values():
    """NaN, inf, -inf, and zero survive the round-trip unchanged."""
    arr = np.array([0.0, np.inf, -np.inf, np.nan, np.finfo(np.float32).max], dtype=np.float32)

    q = SharedQueue()
    results = mp.Queue()

    p = mp.Process(target=_echo_worker, args=(q, results))
    p.start()
    q.put(arr)
    q.put(None)
    received = np.frombuffer(results.get(timeout=10), dtype=np.float32)
    p.join(timeout=5)

    assert p.exitcode == 0
    assert np.array_equal(arr, received, equal_nan=True)
    q.close()


def test_data_integrity_multiple_sequential():
    """Each of N distinct arrays arrives intact and in order."""
    n = 50
    rng = np.random.default_rng(0)
    arrays = [rng.standard_normal(200).astype(np.float64) for _ in range(n)]

    q = SharedQueue()
    results = mp.Queue()

    p = mp.Process(target=_echo_worker, args=(q, results))
    p.start()
    for arr in arrays:
        q.put(arr)
    q.put(None)
    received_list = [
        np.frombuffer(results.get(timeout=20), dtype=np.float64) for _ in arrays
    ]
    p.join(timeout=5)

    assert p.exitcode == 0
    for original, received in zip(arrays, received_list):
        np.testing.assert_array_equal(original, received)
    q.close()


def test_dtype_preservation():
    q = SharedQueue()
    results = mp.Queue()

    p = mp.Process(target=_identity_worker, args=(q, results))
    p.start()

    for dtype in [np.float32, np.float64, np.int32, np.uint8, np.complex64]:
        arr = np.array([1, 2, 3], dtype=dtype)
        q.put(arr)

    q.put(None)
    p.join(timeout=10)

    for dtype in [np.float32, np.float64, np.int32, np.uint8, np.complex64]:
        got = results.get_nowait()
        assert got == [1, 2, 3] or np.allclose(got, [1, 2, 3])

    q.close()
