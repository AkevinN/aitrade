"""
调度运行日志（task-scheduler-observability 新增粘合 3）。

职责：包装 JsonlDayStore，暴露 record_skip / record_trigger / record_error 三个
语义方法，供 PlanScheduler 在各判定点调用（best-effort：写入失败仅 WARNING，绝不
向调用方传播）。

事件 JSON 形态（design.md 数据模型节）：
  skip    → {"ts":…, "event":"skip",    "plan_id":…, "reason":…, "detail":…}
  trigger → {"ts":…, "event":"trigger", "plan_id":…, "slot":…,   "detail":…}
  error   → {"ts":…, "event":"error",   "plan_id":…, "error":…}

去重：
  skip 事件：dedup_key = f"skip:{plan_id}:{reason}"，同日同因只记一次（R3.2）。
  trigger / error 事件：不去重（同日可多次触发 / 多次报错）。
"""

from __future__ import annotations

import logging

from .jsonl_store import JsonlDayStore

logger = logging.getLogger(__name__)

_ERROR_MAX_LEN = 500  # R3.4 截断长度


class SchedulerRunLog:
    """调度运行日志：包装 JsonlDayStore，提供语义化事件记录方法。

    所有方法 best-effort：内部异常以 WARNING 记录，绝不向调用方传播（R3.5）。
    """

    def __init__(self, store: JsonlDayStore) -> None:
        """注入底层 JSONL 存储，构成语义事件记录层。

        Args:
            store: 已初始化的 JsonlDayStore，所有 record_* / query 调用都委托给它；
                本类不持有其它状态，仅在其之上封装事件形态与去重键。
        """
        self._store = store

    # ------------------------------------------------------------------
    # 写入方法
    # ------------------------------------------------------------------

    def record_skip(self, plan_id: str, reason: str, detail: str = "") -> None:
        """记录一次跳过事件。同日同 (plan_id, reason) 只记一次（R3.2）。

        Args:
            plan_id: 计划 ID。
            reason:  跳过原因（Skip_Reason 枚举值字符串）。
            detail:  可选说明（不含凭证）。
        """
        try:
            dedup_key = f"skip:{plan_id}:{reason}"
            event: dict = {"event": "skip", "plan_id": plan_id, "reason": reason}
            if detail:
                event["detail"] = detail
            else:
                event["detail"] = ""
            self._store.append(event, dedup_key=dedup_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SchedulerRunLog] record_skip 失败（best-effort）: %s", exc)

    def record_trigger(self, plan_id: str, slot: str, detail: str = "") -> None:
        """记录一次触发事件。不去重（同日可多 slot 触发，R3.3）。

        Args:
            plan_id: 计划 ID。
            slot:    触发时点（"HH:MM"）。
            detail:  可选说明。
        """
        try:
            event: dict = {"event": "trigger", "plan_id": plan_id, "slot": slot, "detail": detail}
            self._store.append(event)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SchedulerRunLog] record_trigger 失败（best-effort）: %s", exc)

    def record_error(self, plan_id: str, error: str) -> None:
        """记录一次触发异常事件，error 截断至 500 字符（R3.4）。

        Args:
            plan_id: 计划 ID。
            error:   异常摘要（截断后存储）。
        """
        try:
            truncated = error[:_ERROR_MAX_LEN]
            event: dict = {"event": "error", "plan_id": plan_id, "error": truncated}
            self._store.append(event)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SchedulerRunLog] record_error 失败（best-effort）: %s", exc)

    # ------------------------------------------------------------------
    # 查询方法（只读，供任务 6 API 使用）
    # ------------------------------------------------------------------

    def query(
        self,
        plan_id: str | None = None,
        day=None,
        limit: int | None = None,
    ) -> list[dict]:
        """只读倒序查询调度运行日志（R3 / R6.1）。

        Args:
            plan_id: 可选过滤计划 ID；None = 不过滤。
            day:     查询日期（date 对象）；None = 当日。
            limit:   最多返回条数；None = 不限。

        Returns:
            按时间戳倒序的事件列表（最新在前）。
        """
        from datetime import datetime as _dt

        if day is None:
            # 本地时间（与 JsonlDayStore 生产单例 now_fn=datetime.now 口径一致）
            day = _dt.now().date()

        records = self._store.read_day(day)

        if plan_id is not None:
            records = [r for r in records if r.get("plan_id") == plan_id]

        # 倒序（最新在前）
        records = list(reversed(records))

        if limit is not None:
            records = records[:limit]

        return records
