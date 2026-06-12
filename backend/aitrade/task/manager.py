"""
异步任务管理器 — 线程安全单例，管理耗时后台任务的生命周期。

Usage:
    manager = TaskManager()
    task_id = manager.create_task(TaskType.DATA_DOWNLOAD, {"symbols": ["000001.SZSE"]})
    manager.run_async(task_id, download_function, enable_progress=True)
"""

from __future__ import annotations

import copy
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from ..config import MAX_WORKERS, TASK_HISTORY_PATH
from ..models import TaskModel, TaskStatus, TaskType

logger = logging.getLogger(__name__)

# 疑似凭证键名关键字（不区分大小写）
_SENSITIVE_KEYS = {"token", "secret", "webhook", "password"}


def _sanitize_value(v: Any) -> Any:
    """递归脱敏任意值（dict / list / tuple / 标量），不修改原对象。

    Args:
        v: 待脱敏的任意值。

    Returns:
        脱敏后的副本；dict/list/tuple 深递归，标量原样返回。
    """
    if isinstance(v, dict):
        return _sanitize_params(v)
    if isinstance(v, list):
        return [_sanitize_value(item) for item in v]
    if isinstance(v, tuple):
        return tuple(_sanitize_value(item) for item in v)
    return v


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """递归扫描 dict，键名小写含凭证关键字的值替换为 '***'（R1.5）。

    凭证关键字集合由模块级 ``_SENSITIVE_KEYS`` 定义（token / secret / webhook / password）。
    list/tuple 元素中的嵌套 dict 也会被递归脱敏。

    Args:
        params: 原始任务参数 dict（深拷贝后传入，本函数不修改外部状态）。

    Returns:
        脱敏后的新 dict，凭证键对应的值替换为 ``"***"``。
    """
    result: dict[str, Any] = {}
    for k, v in params.items():
        if any(kw in k.lower() for kw in _SENSITIVE_KEYS):
            result[k] = "***"
        else:
            result[k] = _sanitize_value(v)
    return result

# 保留已完成任务的最大数量
_MAX_COMPLETED_TASKS = 200
# 已完成任务的最大保留时间（小时）
_COMPLETED_TASK_TTL_HOURS = 24


class TaskManager:
    """线程安全单例任务管理器，使用守护线程执行耗时后台任务。

    通过 double-checked locking 实现进程内单例；``create_task`` 创建任务记录，
    ``run_async`` 投递到 ThreadPoolExecutor 并自动更新 PENDING → RUNNING → COMPLETED/FAILED
    状态流转；终态时 best-effort 归档到 TaskHistoryStore（R2.1/R2.4）。
    """

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
                    # 延迟导入避免循环
                    from .history import TaskHistoryStore
                    cls._instance._history_store = TaskHistoryStore(TASK_HISTORY_PATH)
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
        """创建新任务并返回其 ID。

        参数深拷贝后经 ``_sanitize_params`` 脱敏存储（R1.4/R1.5），
        调用方后续修改 params 不影响任务记录。自动触发 ``_cleanup_old_tasks`` 防内存泄漏。

        Args:
            task_type:   任务类型枚举（``TaskType.DATA_DOWNLOAD`` 等）。
            params:      任务参数 dict，凭证键会被替换为 ``"***"``；传 None 则存空 dict。
            title:       任务标题（展示用，写入 message 初始值）。
            entity_type: 关联实体类型（如 "live_decision"），用于过滤与展示。
            entity_name: 关联实体名称（如方案名），用于展示。

        Returns:
            8 位十六进制任务 ID（UUID4 前缀），全局唯一。
        """
        task_id = uuid.uuid4().hex[:8]
        now = datetime.now()
        # 深拷贝后脱敏存储（R1.4/R1.5）：调用方后续修改不影响记录；疑似凭证键值替换 "***"
        stored_params: dict[str, Any] = {}
        if params:
            stored_params = _sanitize_params(copy.deepcopy(params))
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
            params=stored_params,
        )
        with self._task_lock:
            self._tasks[task_id] = task
            self._cleanup_old_tasks()
        return task_id

    def update_task(self, task_id: str, **kwargs: Any) -> bool:
        """更新任务字段（线程安全）。

        Args:
            task_id: 目标任务 ID。
            **kwargs: 需要更新的字段及新值（须为 TaskModel 上的合法属性）。

        Returns:
            True 表示更新成功；False 表示任务不存在。
        """
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
        """按 ID 获取任务快照（线程安全）。

        Args:
            task_id: 目标任务 ID。

        Returns:
            TaskModel 实例；不存在返回 None。
        """
        with self._task_lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[TaskModel]:
        """返回当前内存中所有任务的列表（线程安全快照）。

        Returns:
            TaskModel 列表（顺序为 dict 插入顺序，即创建时刻升序）。
        """
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
        """在守护线程中异步执行函数，并自动更新任务状态。

        状态流转：PENDING → RUNNING（启动时）→ COMPLETED/FAILED（终态）；
        终态时 best-effort 调用 ``_archive`` 归档到 TaskHistoryStore。

        Args:
            task_id:         要追踪本次执行的任务 ID（须已由 ``create_task`` 创建）。
            func:            待执行函数。若 enable_progress=True 或传入 on_progress，
                             函数将收到 ``on_progress: Callable[[float, str], None]``
                             关键字参数（由框架注入）。
            *args:           额外位置参数，透传给 func。
            enable_progress: True 时自动创建进度回调，调用后更新任务 progress/message；
                             与显式 on_progress 二选一（on_progress 优先）。
            on_progress:     显式进度回调 ``(progress: float, message: str) -> None``；
                             优先于 enable_progress。
            **kwargs:        额外关键字参数，透传给 func。
        """

        def wrapper() -> None:
            started = datetime.now()
            try:
                self.update_task(
                    task_id,
                    status=TaskStatus.RUNNING,
                    message="任务执行中",
                    started_at=started,  # R1.3：记录开始时刻
                )

                # Determine effective callback
                effective_callback = on_progress
                if effective_callback is None and enable_progress:
                    def effective_callback(progress: float, message: str = "") -> None:
                        self.update_task(task_id, progress=progress, message=message)

                if effective_callback:
                    result = func(*args, on_progress=effective_callback, **kwargs)
                else:
                    result = func(*args, **kwargs)

                finished = datetime.now()
                duration = int((finished - started).total_seconds() * 1000)
                self.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    progress=100.0,
                    message="任务完成",
                    result=result,
                    finished_at=finished,   # R1.3：记录终态时刻
                    duration_ms=duration,   # R1.3：计算耗时
                )
                # R2.1/R2.4：终态钩子 best-effort 归档
                self._archive(task_id)
            except Exception as e:
                finished = datetime.now()
                duration = int((finished - started).total_seconds() * 1000)
                tb = traceback.format_exc()[:8000]  # R1.2：截断至 8000 字符
                logger.exception("任务 %s 执行失败", task_id)  # R1.2：logger.exception 输出完整堆栈
                self.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=str(e),
                    finished_at=finished,       # R1.3：终态时刻
                    duration_ms=duration,       # R1.3：耗时
                    error_traceback=tb,         # R1.2：存储截断堆栈
                )
                # R2.1/R2.4：终态钩子 best-effort 归档
                self._archive(task_id)

        self._executor.submit(wrapper)

    def _archive(self, task_id: str) -> None:
        """终态钩子：best-effort 归档到 TaskHistoryStore（R2.1/R2.4）。

        失败时记 WARNING 日志，不影响任务本身的状态流转。
        """
        task = self.get_task(task_id)
        if task is None:
            return
        try:
            self._history_store.archive(task)
        except Exception as exc:
            logger.warning("任务归档失败 %s: %s", task_id, exc)

    def _cleanup_old_tasks(self) -> None:
        """清理过期或超量的终态任务（须在持有 _task_lock 时调用）。

        两轮清理：①超过 TTL（默认 24 h）的终态任务；②超过上限（200 条）的
        最旧终态任务。仅在 create_task 路径调用，保证内存不会无限增长。
        """
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
