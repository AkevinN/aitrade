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
        """初始化存储，建好根目录并准备按日去重集合。

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
        """拼出某天对应的 JSONL 文件路径（{base}/{YYYY-MM-DD}.jsonl）。

        Args:
            day: 目标日期。

        Returns:
            该日文件的 Path（不保证文件已存在）。
        """
        return self._base / f"{day.isoformat()}.jsonl"

    def _ensure_dedup_loaded(self, day: date) -> None:
        """首次访问某天时，从文件回放重建去重集合（持锁调用）。

        懒加载：若 day 已在内存 _dedup 中则直接返回；否则逐行读当日文件，
        把每条记录的 _dedup 字段收集成 set 存入内存。文件不存在或读 IO 异常时，
        以空集合兜底（warning 已记录），保证后续 append 仍能写入。

        Args:
            day: 要重建去重集合的目标日期，对应 {base}/{day}.jsonl 文件。
        """
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
        """读取指定日期的全部记录（只读，坏行跳过+warning）。

        逐行解析当日 JSONL 文件；JSON 解析失败的坏行跳过并记 warning，
        不中断整体读取。返回前会剥离内部去重字段 _dedup，仅对外投影业务字段。

        Args:
            day: 目标日期，对应 {base}/{day}.jsonl 文件（含其全部记录）。

        Returns:
            该日全部有效记录组成的列表，按文件中出现顺序排列，每条已去掉 _dedup 字段；
            文件不存在、读 IO 异常或全为坏行时返回空 list。
        """
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

    # ------------------------------------------------------------------
    # 删（重写当日文件去掉命中记录）
    # ------------------------------------------------------------------

    def delete_where(self, day: date, predicate: Callable[[dict[str, Any]], bool]) -> int:
        """删除当日文件中满足 ``predicate`` 的记录，重写文件，返回删除条数。

        用于"运行历史"删除一条记录等管理操作。``predicate`` 接收**已剥离 ``_dedup``**
        的业务字典（与 :meth:`read_day` 投影一致）。重写时保留未命中记录的原始对象（含
        ``_dedup`` 字段），并从内存去重集合移除被删记录的 ``_dedup`` 键（使该键之后可重新
        写入）。坏行无法解析时原样保留（不丢用户数据）。最佳努力：文件不存在或 IO 异常时
        记 warning 并返回 0。

        Args:
            day: 目标日期（对应一个 JSONL 文件）。
            predicate: 命中判定；对投影后的业务字典返回 True 表示删除该条。

        Returns:
            实际删除的记录条数（0 表示无命中或文件不存在）。
        """
        with self._lock:
            path = self._day_path(day)
            if not path.exists():
                return 0
            try:
                raw_lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                logger.warning("JsonlDayStore: 读取 %s 失败（删除跳过）: %s", path, exc)
                return 0

            kept: list[str] = []
            removed = 0
            removed_keys: list[str] = []
            for raw in raw_lines:
                s = raw.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    kept.append(s)  # 坏行原样保留，不误删
                    continue
                projected = {k: v for k, v in obj.items() if k != "_dedup"}
                if predicate(projected):
                    removed += 1
                    dk = obj.get("_dedup")
                    if dk is not None:
                        removed_keys.append(dk)
                else:
                    kept.append(json.dumps(obj, ensure_ascii=False))

            if removed == 0:
                return 0

            try:
                with path.open("w", encoding="utf-8") as f:
                    for line in kept:
                        f.write(line + "\n")
            except OSError as exc:
                logger.warning("JsonlDayStore: 重写 %s 失败（删除未生效）: %s", path, exc)
                return 0

            if day in self._dedup:
                for k in removed_keys:
                    self._dedup[day].discard(k)
            return removed
