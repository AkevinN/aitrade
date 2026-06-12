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
    """基于文件 flock 的单实例互斥锁，防止多实例并发下单。

    同一锁文件第二个持有者无法获取锁，从而保证全局唯一下单实例。
    进程退出后锁自动失效（flock 建议锁，OS 自动释放）；主备切换时
    新主接管前需确认旧主已退出或释放锁。

    Example:
        >>> lock = SingleInstanceLock("/tmp/scheduler.lock")
        >>> with lock:
        ...     run_scheduler()
    """

    def __init__(self, lock_path: Path | str) -> None:
        """初始化单实例锁。

        Args:
            lock_path: 锁文件路径；父目录不存在时自动创建。
        """
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
        """释放锁，供 stop() 或 context manager __exit__ 调用。重复释放安全（幂等）。"""
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "SingleInstanceLock":
        """Context manager 入口：acquire 失败时抛 RuntimeError，阻止启动。

        Raises:
            RuntimeError: 锁被占用，说明已有实例在运行，拒绝启动以防并发下单。
        """
        if not self.acquire():
            raise RuntimeError("已有实例在运行（单实例互斥），拒绝启动以防并发下单")
        return self

    def __exit__(self, *exc) -> None:
        """Context manager 退出时自动释放锁。"""
        self.release()
