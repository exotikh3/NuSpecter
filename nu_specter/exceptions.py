class SharedMemoryError(Exception):
    """Base exception for this library."""


class BlockClosedError(SharedMemoryError):
    """Raised when accessing a SharedBlock that has already been closed."""


class BlockAlreadyUnlinkedError(SharedMemoryError):
    """Raised when unlink() is called on an already-unlinked block."""


class PoolExhaustedError(SharedMemoryError):
    """Raised when BlockPool.acquire() times out with no free slots."""


class HandleMismatchError(SharedMemoryError):
    """Raised when attaching to shared memory whose size doesn't match the handle."""
