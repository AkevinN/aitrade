"""
单实例互斥锁（迭代 10）：防止多实例并发下单（高可用红线）。

基于文件 flock（建议锁）。同一锁文件第二个持有者无法获取锁，从而保证全局唯一下单实例。
主备切换时，新主接管前必须能拿到锁（旧主释放或进程退出后锁自动失效）。
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


class SingleInstanceLock:
    def __init__(self, lock_path: Path | str) -> None:
        self.lock_path = Path(lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd: int | None = None

    def acquire(self) -> bool:
        """尝试获取锁（非阻塞）。成功 True，已被占用 False。"""
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "SingleInstanceLock":
        if not self.acquire():
            raise RuntimeError("已有实例在运行（单实例互斥），拒绝启动以防并发下单")
        return self

    def __exit__(self, *exc) -> None:
        self.release()
