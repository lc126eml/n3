import os
import time

from filelock import FileLock


class PriorityLock:
    _last_cleanup_time = 0.0
    _cleanup_interval = 180.0

    def __init__(
        self,
        lock_dir: str = "/tmp/gpu.lock",
        priority: int = 10,
        poll_interval: float = 5.0,
        cleanup_mode: str = "pid",
    ) -> None:
        self.lock_dir = lock_dir
        self.priority = priority
        self.poll_interval = poll_interval
        self.cleanup_mode = cleanup_mode
        self._token_path = None
        self._file_lock = FileLock(f"{lock_dir}.lock")

    def acquire(self) -> None:
        if os.path.exists(self.lock_dir) and not os.path.isdir(self.lock_dir):
            os.remove(self.lock_dir)
        os.makedirs(self.lock_dir, exist_ok=True)

        token_name = f"{self.priority:02d}_{time.time_ns()}_{os.getpid()}.token"
        self._token_path = os.path.join(self.lock_dir, token_name)
        with open(self._token_path, "x", encoding="utf-8"):
            pass

        while True:
            tokens = sorted(
                name for name in os.listdir(self.lock_dir) if name.endswith(".token")
            )
            self._cleanup_dead_tokens(tokens)
            if tokens and os.path.join(self.lock_dir, tokens[0]) == self._token_path:
                self._file_lock.acquire()
                os.remove(self._token_path)
                self._token_path = None
                return
            time.sleep(self.poll_interval)

    def release(self) -> None:
        self._file_lock.release()
        if self._token_path and os.path.exists(self._token_path):
            os.remove(self._token_path)
        self._token_path = None

    def _cleanup_dead_tokens(self, tokens: list[str]) -> None:
        now = time.time()
        if now - PriorityLock._last_cleanup_time < PriorityLock._cleanup_interval:
            return
        PriorityLock._last_cleanup_time = now
        if self._token_path is None:
            return
        own_name = os.path.basename(self._token_path)
        if own_name not in tokens:
            return
        for name in tokens:
            if name == own_name:
                return
            token_path = os.path.join(self.lock_dir, name)
            pid = self._parse_pid(name)
            if pid is not None and not self._pid_alive(pid):
                os.remove(token_path)
                continue
            return

    @staticmethod
    def _parse_pid(name: str) -> int | None:
        parts = name.split("_")
        if len(parts) < 3:
            return None
        pid_part = parts[2]
        if pid_part.endswith(".token"):
            pid_part = pid_part[:-6]
        return int(pid_part) if pid_part.isdigit() else None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        return os.path.exists(f"/proc/{pid}")

    def __enter__(self) -> "PriorityLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
