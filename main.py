"""
Benchmark: shared memory vs pickle-based multiprocessing for numpy arrays.

Measures throughput (arrays/sec) and total time for sending N large arrays
from a producer to a consumer process.
"""
import multiprocessing as mp
import pickle
import time

import numpy as np

from nu_specter import SharedQueue

SHAPE = (300, 300, 3)
DTYPE = np.float32
N = 10000


# ---------------------------------------------------------------------------
# Shared memory approach
# ---------------------------------------------------------------------------

def _shm_worker(q_in, q_out):
    received = 0
    while True:
        item = q_in.get()
        if item is None:
            break
        _ = item.array.sum()  # force a read to simulate real work
        item.close()
        received += 1
    q_out.put(received)


def bench_shared_memory():
    q = SharedQueue(maxsize=100)
    results = mp.Queue()

    p = mp.Process(target=_shm_worker, args=(q, results))
    p.start()

    arr = np.random.rand(*SHAPE).astype(DTYPE)
    t0 = time.perf_counter()
    for _ in range(N):
        q.put(arr)
    q.put(None)
    p.join()
    elapsed = time.perf_counter() - t0

    received = results.get()
    q.close()
    return elapsed, received


# ---------------------------------------------------------------------------
# Pickle-based approach (standard multiprocessing.Queue)
# ---------------------------------------------------------------------------

def _pickle_worker(q_in, q_out):
    received = 0
    while True:
        item = q_in.get()
        if item is None:
            break
        _ = item.sum()
        received += 1
    q_out.put(received)


def bench_pickle():
    q = mp.Queue(maxsize=100)
    results = mp.Queue(maxsize=100)

    p = mp.Process(target=_pickle_worker, args=(q, results))
    p.start()

    arr = np.random.rand(*SHAPE).astype(DTYPE)
    t0 = time.perf_counter()
    for _ in range(N):
        q.put(arr)
    q.put(None)
    p.join()
    elapsed = time.perf_counter() - t0

    received = results.get()
    return elapsed, received


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    array_mb = np.prod(SHAPE) * np.dtype(DTYPE).itemsize / 1024**2
    total_mb = array_mb * N

    print(f"Benchmark: {N} arrays of shape {SHAPE} ({array_mb:.1f} MB each, {total_mb:.0f} MB total)")
    print()

    print("Running shared memory benchmark ...")
    shm_time, shm_n = bench_shared_memory()

    print("Running pickle benchmark ...")
    pkl_time, pkl_n = bench_pickle()

    shm_tput = shm_n / shm_time
    pkl_tput = pkl_n / pkl_time
    shm_bw = total_mb / shm_time
    pkl_bw = total_mb / pkl_time

    print()
    print(f"{'Method':<20} {'Time (s)':>10} {'Arrays/s':>10} {'Bandwidth':>12}")
    print("-" * 56)
    print(f"{'Shared memory':<20} {shm_time:>10.2f} {shm_tput:>10.1f} {shm_bw:>10.1f} MB/s")
    print(f"{'Pickle':<20} {pkl_time:>10.2f} {pkl_tput:>10.1f} {pkl_bw:>10.1f} MB/s")
    print()
    speedup = pkl_time / shm_time
    print(f"Speedup: {speedup:.2f}x faster with shared memory")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
