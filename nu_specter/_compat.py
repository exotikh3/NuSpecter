"""Platform compatibility helpers."""
import contextlib
import multiprocessing
import sys
import warnings


def get_start_method() -> str:
    return multiprocessing.get_start_method()


def is_spawn() -> bool:
    return get_start_method() == "spawn"


def is_64bit() -> bool:
    return sys.maxsize > 2**32


@contextlib.contextmanager
def suppress_resource_tracker_warnings():
    """
    Suppress the spurious 'leaked ... semaphore/shm' warnings that Python's
    internal resource tracker sometimes emits during tests when blocks are
    explicitly unlinked by library code before the tracker notices.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="resource_tracker: There appear to be",
            category=UserWarning,
        )
        yield
