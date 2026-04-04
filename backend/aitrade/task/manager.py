"""
Async task manager — thread-safe singleton for long-running background tasks.

Usage:
    manager = TaskManager()
    task_id = manager.create_task(TaskType.DATA_DOWNLOAD, {"symbols": ["000001.SZSE"]})
    manager.run_async(task_id, download_function)
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

from ..models import TaskModel, TaskStatus, TaskType


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
        return cls._instance

    def create_task(self, task_type: TaskType, params: dict[str, Any] | None = None) -> str:
        """Create a new task and return its ID."""
        task_id = uuid.uuid4().hex[:8]
        now = datetime.now()
        task = TaskModel(
            task_id=task_id,
            type=task_type,
            status=TaskStatus.PENDING,
            progress=0.0,
            message="任务已创建",
            created_at=now,
            updated_at=now,
        )
        with self._task_lock:
            self._tasks[task_id] = task
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
        on_progress: Optional[Callable[[float, str], None]] = None,
        **kwargs: Any,
    ) -> None:
        """Run a function in a daemon thread, automatically updating task status."""

        def wrapper() -> None:
            try:
                self.update_task(task_id, status=TaskStatus.RUNNING, message="任务执行中")

                if on_progress:
                    def progress_callback(progress: float, message: str = "") -> None:
                        self.update_task(task_id, progress=progress, message=message)

                    result = func(*args, on_progress=progress_callback, **kwargs)
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

        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()


# Global singleton
task_manager = TaskManager()
