"""
ResourceTracker: process-local registry that ensures every allocated
SharedMemory block is unlinked at process exit, even if user code forgets.
"""
from __future__ import annotations

import atexit
import logging
import threading
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .block import SharedBlock

logger = logging.getLogger(__name__)


class ResourceTracker:
    """
    Singleton per process. Tracks all SharedBlocks allocated in this process
    and unlinks any survivors at exit.

    Usage is automatic: SharedBlock.from_array() calls register(); unlink()
    calls deregister(). Direct use is only needed for diagnostics.
    """

    _instance: "ResourceTracker | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        # ref_id -> (shm_name, block_weakref or None)
        self._registry: Dict[str, str] = {}  # ref_id -> shm_name
        self._lock = threading.Lock()
        atexit.register(self.cleanup)

    @classmethod
    def get(cls) -> "ResourceTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(self, block: "SharedBlock") -> None:
        with self._lock:
            self._registry[block.handle.ref_id] = block.handle.shm_name

    def deregister(self, ref_id: str) -> None:
        with self._lock:
            self._registry.pop(ref_id, None)

    def alive(self) -> list[str]:
        """Return ref_ids of blocks still registered (for leak detection in tests)."""
        with self._lock:
            return list(self._registry.keys())

    def cleanup(self) -> None:
        """Unlink all tracked blocks. Called at atexit."""
        from multiprocessing.shared_memory import SharedMemory

        with self._lock:
            remaining = dict(self._registry)

        for ref_id, shm_name in remaining.items():
            try:
                shm = SharedMemory(name=shm_name, create=False)
                shm.close()
                shm.unlink()
                logger.debug("ResourceTracker cleaned up shm %s (ref_id=%s)", shm_name, ref_id)
            except Exception as exc:
                logger.debug(
                    "ResourceTracker could not unlink shm %s: %s", shm_name, exc
                )

        with self._lock:
            self._registry.clear()
