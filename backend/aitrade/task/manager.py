"""
Async task manager — thread-safe singleton for long-running background tasks.

Usage:
    manager = TaskManager()
    task_id = manager.create_task(TaskType.DATA_DOWNLOAD, {"symbols": ["000001.SZSE"]})
    manager.run_async(task_id, download_function, enable_progress=True)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from ..config import MAX_WORKERS
from ..models import TaskModel, TaskStatus, TaskType

# 保留已完成任务的最大数量
_MAX_COMPLETED_TASKS = 200
# 已完成任务的最大保留时间（小时）
_COMPLETED_TASK_TTL_HOURS = 24


class TaskManager:
    """Thread-safe singleton task manager with daemon threads."""

    _instance: Optional["TaskManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "TaskManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: dict[str, TaskModel] = {}
                    cls._instance._task_lock = threading.Lock()
                    cls._instance._executor = ThreadPoolExecutor(
                        max_workers=max(1, MAX_WORKERS),
                        thread_name_prefix="aitrade-task",
                    )
        return cls._instance

    def create_task(
        self,
        task_type: TaskType,
        params: dict[str, Any] | None = None,
        *,
        title: str = "",
        entity_type: str = "",
        entity_name: str = "",
    ) -> str:
        """Create a new task and return its ID."""
        task_id = uuid.uuid4().hex[:8]
        now = datetime.now()
        task = TaskModel(
            task_id=task_id,
            type=task_type,
            title=title,
            entity_type=entity_type,
            entity_name=entity_name,
            status=TaskStatus.PENDING,
            progress=0.0,
            message=title or "任务已创建",
            created_at=now,
            updated_at=now,
        )
        with self._task_lock:
            self._tasks[task_id] = task
            self._cleanup_old_tasks()
        return task_id

    def update_task(self, task_id: str, **kwargs: Any) -> bool:
        """Update task fields. Returns True on success."""
        with self._task_lock:
            if task_id not in self._tasks:
                return False
            task = self._tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = datetime.now()
            return True

    def get_task(self, task_id: str) -> Optional[TaskModel]:
        """Get task by ID."""
        with self._task_lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[TaskModel]:
        """Get all tasks."""
        with self._task_lock:
            return list(self._tasks.values())

    def run_async(
        self,
        task_id: str,
        func: Callable,
        *args: Any,
        enable_progress: bool = False,
        on_progress: Optional[Callable[[float, str], None]] = None,
        **kwargs: Any,
    ) -> None:
        """Run a function in a daemon thread, automatically updating task status.

        Args:
            task_id: Task ID to track this execution.
            func: The function to run. If enable_progress is True or on_progress
                  is provided, the function will receive an ``on_progress`` keyword
                  argument of type ``Callable[[float, str], None]``.
            enable_progress: When True, auto-create a progress callback that
                             updates the task's progress/message fields.
            on_progress: Explicit progress callback (overrides enable_progress).
        """

        def wrapper() -> None:
            try:
                self.update_task(task_id, status=TaskStatus.RUNNING, message="任务执行中")

                # Determine effective callback
                effective_callback = on_progress
                if effective_callback is None and enable_progress:
                    def effective_callback(progress: float, message: str = "") -> None:
                        self.update_task(task_id, progress=progress, message=message)

                if effective_callback:
                    result = func(*args, on_progress=effective_callback, **kwargs)
                else:
                    result = func(*args, **kwargs)

                self.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    progress=100.0,
                    message="任务完成",
                    result=result,
                )
            except Exception as e:
                self.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=str(e),
                )

        self._executor.submit(wrapper)

    def _cleanup_old_tasks(self) -> None:
        """Remove expired completed/failed tasks. Must be called with _task_lock held."""
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED}
        cutoff = datetime.now() - timedelta(hours=_COMPLETED_TASK_TTL_HOURS)

        # Remove tasks older than TTL
        expired_ids = [
            tid for tid, t in self._tasks.items()
            if t.status in terminal and t.updated_at < cutoff
        ]
        for tid in expired_ids:
            del self._tasks[tid]

        # If still over limit, remove oldest completed tasks
        completed = sorted(
            [(tid, t) for tid, t in self._tasks.items() if t.status in terminal],
            key=lambda x: x[1].updated_at,
        )
        while len(completed) > _MAX_COMPLETED_TASKS:
            tid, _ = completed.pop(0)
            if tid in self._tasks:
                del self._tasks[tid]


# Global singleton
task_manager = TaskManager()
