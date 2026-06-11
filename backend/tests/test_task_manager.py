from __future__ import annotations

import time

from aitrade.config import MAX_WORKERS
from aitrade.models import TaskStatus, TaskType
from aitrade.task import task_manager


def test_task_manager_uses_bounded_executor() -> None:
    assert task_manager._executor._max_workers == max(1, MAX_WORKERS)


def test_task_manager_completes_async_tasks() -> None:
    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD, {"vt_symbols": ["000001.SZSE"]})

    def _work(on_progress=None):
        if on_progress:
            on_progress(50, "halfway")
        return {"ok": True}

    task_manager.run_async(task_id, _work, enable_progress=True)

    deadline = time.time() + 3
    task = task_manager.get_task(task_id)
    while task is not None and task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED} and time.time() < deadline:
        time.sleep(0.05)
        task = task_manager.get_task(task_id)

    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    assert task.result == {"ok": True}
