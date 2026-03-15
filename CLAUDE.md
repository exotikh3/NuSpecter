# CLAUDE.md

## Project overview

`nu-specter` — a Python library for zero-copy numpy array sharing between processes via POSIX shared memory. Instead of pickling arrays, lightweight handles are passed and the receiving process attaches to the same memory region.

**Key public API:** `SharedQueue`, `SharedBlock`, `BlockPool`, `ArrayHandle` (exported from `nu_specter/__init__.py`).

## Environment

- Python 3.12 (`.python-version`)
- Package manager: `uv`
- Dependencies: `numpy` (runtime), `pytest`, `pytest-timeout` (dev)

## Common commands

```bash
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run benchmark
uv run python main.py
```

## Project structure

```
nu_specter/                  # library source
  __init__.py                # public API exports
  block.py                   # SharedBlock — owns a shared memory region + numpy view
  handle.py                  # ArrayHandle — lightweight descriptor passed between processes
  queue.py                   # SharedQueue — mp.Queue wrapper that sends handles
  pool.py                    # BlockPool — pre-allocated block pool for fixed-shape pipelines
  resource_tracker.py        # cleanup tracking for shared memory segments
  exceptions.py              # custom exceptions
  _compat.py                 # Python version compatibility shims
tests/                       # pytest tests mirroring the source modules
main.py                      # benchmark: shared memory vs pickle throughput
```
