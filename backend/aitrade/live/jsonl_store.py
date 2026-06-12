"""
按日分文件的 append-only JSONL 存储（task-scheduler-observability 新增粘合 1）。

设计：
- 每天一个文件 {base}/{YYYY-MM-DD}.jsonl，追加写入；
- dedup_key 去重：进程内 set + 启动时回放当日文件重建——**跨进程去重不保证**（已知缺口，
  记录性数据可重不可丢，容忍极端情况下的重复）；
- 写入 best-effort：IO 异常 logger.warning + 返回 False，绝不向调用方抛出；
- 查询只读：read_day / read_range 不修改任何文件内容；坏行 json 解析失败跳过 + warning。
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonlDayStore:
    """按日分文件的 append-only JSONL 存储。写入 best-effort；查询只读。

    跨进程去重不保证（设计已知缺口）：当日去重依赖进程内集合 + 启动时回放，
    多进程并发写极端情况可能重复记录。记录性数据可重不可丢。
    """

    def __init__(
        self,
        base_path: Path | str,
        *,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        """
        Args:
            base_path: 日志根目录，不存在时自动创建。
            now_fn:    可注入的"当前时间"函数（用于测试），默认 datetime.now(timezone.utc)。
        """
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)
        self._now_fn: Callable[[], datetime] = now_fn or (
            lambda: datetime.now(timezone.utc)
        )
        # 按日维护的去重 set；key = date, value = set[str]
        self._dedup: dict[date, set[str]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _day_path(self, day: date) -> Path:
        return self._base / f"{day.isoformat()}.jsonl"

    def _ensure_dedup_loaded(self, day: date) -> None:
        """首次访问某天时，从文件回放重建去重集合（持锁调用）。"""
        if day in self._dedup:
            return
        seen: set[str] = set()
        path = self._day_path(day)
        if path.exists():
            try:
                with path.open(encoding="utf-8") as f:
                    for raw in f:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        dk = obj.get("_dedup")
                        if dk is not None:
                            seen.add(str(dk))
            except OSError as exc:
                logger.warning("JsonlDayStore: 回放 %s 失败，去重集合为空: %s", path, exc)
        self._dedup[day] = seen

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------

    def append(self, event: dict[str, Any], *, dedup_key: str | None = None) -> bool:
        """追加一条事件（自动补 ts）。

        Args:
            event:     要写入的字典；若不含 ts 字段则自动补 ISO 时间戳。
            dedup_key: 非 None 时：当日已存在同键则跳过，返回 False；
                       新键写入后加入当日去重集合，_dedup 字段随记录持久化。

        Returns:
            True  = 成功写入；
            False = 跳过（去重命中）或 IO 异常（best-effort）。

        跨进程去重不保证——记录性数据可重不可丢。
        """
        now = self._now_fn()
        day = now.date()

        with self._lock:
            self._ensure_dedup_loaded(day)

            if dedup_key is not None:
                if dedup_key in self._dedup[day]:
                    return False

            payload: dict[str, Any] = {}
            if "ts" not in event:
                payload["ts"] = now.isoformat(timespec="seconds")
            payload.update(event)
            if dedup_key is not None:
                payload["_dedup"] = dedup_key

            path = self._day_path(day)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning("JsonlDayStore: 写入 %s 失败（best-effort）: %s", path, exc)
                return False

            if dedup_key is not None:
                self._dedup[day].add(dedup_key)

        return True

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------

    def read_day(self, day: date) -> list[dict[str, Any]]:
        """读取指定日期的全部记录（只读，坏行跳过+warning）。"""
        path = self._day_path(day)
        if not path.exists():
            return []
        results: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as exc:
            logger.warning("JsonlDayStore: 读取 %s 失败: %s", path, exc)
            return []
        for lineno, raw in enumerate(lines, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                # 剥离内部字段 _dedup，仅对外返回时投影（M2 修复）；
                # _ensure_dedup_loaded 直接读原始文件行，不经此路径，去重重建不受影响。
                obj.pop("_dedup", None)
                results.append(obj)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "JsonlDayStore: %s 第 %d 行 JSON 解析失败，跳过: %s",
                    path, lineno, exc,
                )
        return results

    def read_range(
        self,
        start: date,
        end: date,
        *,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        limit: int | None = None,
        reverse: bool = False,
    ) -> list[dict[str, Any]]:
        """读取 [start, end] 日期范围内的记录（含两端，只读）。

        Args:
            start:     开始日期（含）。
            end:       结束日期（含）。
            predicate: 可选过滤函数，返回 True 的记录才保留。
            limit:     最多返回条数（None = 不限）。
            reverse:   True = 结果按日期倒序（同日内顺序不变）；
                       False（默认）= 按日期正序。

        Returns:
            符合条件的记录列表。
        """
        if start > end:
            return []

        from datetime import timedelta

        days: list[date] = []
        cur = start
        while cur <= end:
            days.append(cur)
            cur += timedelta(days=1)

        if reverse:
            days = list(reversed(days))

        results: list[dict[str, Any]] = []
        for day in days:
            for record in self.read_day(day):
                if predicate is not None and not predicate(record):
                    continue
                results.append(record)
                if limit is not None and len(results) >= limit:
                    return results
        return results
