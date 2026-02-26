from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None


@contextmanager
def file_lock(lock_path, timeout_sec=600, poll_interval=3.0):
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+" if path.exists() else "a+"
    with path.open(mode, encoding="utf-8") as f:
        start = time.time()
        while True:
            try:
                if os.name == "nt" and msvcrt is not None:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                elif fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, BlockingIOError):
                if timeout_sec is not None and (time.time() - start) >= timeout_sec:
                    raise TimeoutError(f"Timeout waiting for lock: {path}")
                time.sleep(poll_interval)
        try:
            yield
        finally:
            if os.name == "nt" and msvcrt is not None:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            elif fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

