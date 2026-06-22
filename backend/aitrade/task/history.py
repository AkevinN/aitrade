"""
任务历史持久化（task-scheduler-observability 新增粘合 2 后半）。

设计：
- TaskHistoryStore 包一个 JsonlDayStore，按日 JSONL 文件归档任务终态记录；
- archive(task) best-effort：失败时向调用方返回 False（manager 层记 WARNING）；
- query() 只读，不修改任何文件；
- dedup_key = f"task:{task_id}"，防终态重复写（update_task 可能多次置同一终态）。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..live.jsonl_store import JsonlDayStore
from ..models.alpha import TaskModel


class TaskHistoryStore:
    """任务终态归档存储（按日 JSONL，只读查询，dedup 防重复归档）。

    注：使用本地时间（datetime.now()）确定"当日"，以对应 date.today() 的语义，
    避免 UTC 与本地日期在 UTC±8 区间跨日时不一致。
    """

    def __init__(self, base_path: Path | str) -> None:
        """
        Args:
            base_path: 日志根目录（如 AITRADE_HOME / "task_history"），不存在时自动创建。
        """
        # 使用本地时间决定当日文件名，与 date.today() 保持一致
        self._store = JsonlDayStore(Path(base_path), now_fn=lambda: datetime.now())

    # ------------------------------------------------------------------
    # 写（best-effort）
    # ------------------------------------------------------------------

    def archive(self, task: TaskModel) -> bool:
        """归档任务终态快照到当日 JSONL 文件。

        Args:
            task: 待归档的 TaskModel（通常为终态 completed/failed）。

        Returns:
            True  = 成功写入；
            False = dedup 命中（已存在）或 IO 异常（best-effort）。

        异常不会向调用方传播（JsonlDayStore 内部已 best-effort 处理 OSError）。
        """
        payload = task.model_dump(mode="json")
        dedup_key = f"task:{task.task_id}"
        return self._store.append(payload, dedup_key=dedup_key)

    # ------------------------------------------------------------------
    # 读（只读）
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        status: str | None = None,
        task_type: str | None = None,
        start: date | None = None,
        end: date | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """按条件查询历史任务，倒序返回（最新在前）。

        Args:
            status:    按状态过滤（如 "completed" / "failed"）。
            task_type: 按任务类型过滤（如 "data_download"）。
            start:     查询起始日期（含），默认今日。
            end:       查询截止日期（含），默认今日。
            limit:     最多返回条数（None = 不限）。

        Returns:
            符合条件的任务记录列表（dict），最新在前。
        """
        today = date.today()  # 本地日期，与 now_fn=datetime.now() 一致
        q_start = start or today
        q_end = end or today

        def predicate(r: dict[str, Any]) -> bool:
            """按 status / task_type 过滤单条记录，命中保留则返回 True。"""
            if status is not None and r.get("status") != status:
                return False
            if task_type is not None and r.get("type") != task_type:
                return False
            return True

        results = self._store.read_range(
            q_start,
            q_end,
            predicate=predicate,
            limit=None,       # 先取全量再倒序截断
            reverse=True,     # 按日期倒序
        )

        # 同日内记录按写入顺序（自然顺序）——read_range 当日内已是文件顺序；
        # 跨日 reverse=True 已将更新的日期排在前面。
        # 对同日内再做一次倒序，使最新写入的排在前。
        # 实现方式：read_range reverse=True 已经按日期降序遍历每天文件（最新日先读），
        # 但同日内文件顺序为追加顺序（旧→新）；此处将整个列表再反转一次
        # 使最新记录排最前。
        results = list(reversed(results))

        if limit is not None:
            results = results[:limit]

        # 剔除内部字段（`_` 前缀键 和 `ts` 键），保持与内存任务响应形态对称
        def _strip_internal(r: dict[str, Any]) -> dict[str, Any]:
            """剔除记录里的内部字段（`_` 前缀键与 `ts` 键），返回对外可见的副本。"""
            return {k: v for k, v in r.items() if not k.startswith("_") and k != "ts"}

        return [_strip_internal(r) for r in results]

    def delete_by_task_id(self, task_id: str, *, days_back: int = 365) -> bool:
        """从最近 ``days_back`` 天的归档中删除指定 task_id 的记录（"运行历史"管理删除用）。

        逐日扫描窗口内的日期文件，重写去掉该 task_id 的记录（文件不存在的日期为快速空操作）。
        归档去重键为 ``task:{task_id}``，删除时一并从底座内存去重集合移除，使该 task_id
        之后可再次归档。

        Args:
            task_id: 待删除的任务 ID。
            days_back: 回看窗口天数（含今天），默认 365。

        Returns:
            是否删掉了至少一条记录。
        """
        from datetime import timedelta

        today = date.today()
        removed = 0
        for i in range(max(1, days_back)):
            day = today - timedelta(days=i)
            removed += self._store.delete_where(day, lambda r: r.get("task_id") == task_id)
        return removed > 0
