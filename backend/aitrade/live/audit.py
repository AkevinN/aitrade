"""
审计日志（迭代 9）：append-only JSONL，记录 信号→决策→下单→成交 全链路，可追溯。

每条审计带时间戳、事件类型、signal_id、版本（模型/Scheme），便于复盘与合规。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class AuditLog:
    """Append-only JSONL 审计日志，记录信号→决策→下单→成交全链路事件。

    每条审计记录带时间戳、事件类型、signal_id 与版本标签，便于复盘与合规查阅。
    文件路径不存在时自动创建父目录（无需预建）。

    Example:
        >>> log = AuditLog("/tmp/audit.jsonl")
        >>> log.record("order", {"vt_symbol": "000001.SZSE"}, signal_id="2026-06-08:scheme")
    """

    def __init__(self, path: Path | str) -> None:
        """初始化审计日志。

        Args:
            path: JSONL 文件路径，父目录不存在时自动创建。
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        signal_id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> dict[str, Any]:
        """写入一条审计记录并返回该记录 dict。

        追加写入（append-only），线程不安全——高并发场景需调用方加锁。

        Args:
            event_type: 事件类型，如 "order" / "order_rejected" / "signal" / "decision"。
            payload:    事件详情 dict，随 event_type 而定，无固定 schema。
            signal_id:  幂等键，与 Decision/SignalService 的 signal_id 对齐，可为 None。
            version:    模型/scheme 版本标签，可为 None。

        Returns:
            写入的完整记录 dict，含 ts / event_type / signal_id / version / payload。
        """
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event_type": event_type,
            "signal_id": signal_id,
            "version": version,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def read_all(self) -> list[dict[str, Any]]:
        """读取全部审计记录（按写入顺序）。

        文件不存在时返回空列表，空行自动忽略。

        Returns:
            记录 dict 列表，每条对应一条 JSONL 行。
        """
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def for_signal(self, signal_id: str) -> list[dict[str, Any]]:
        """返回指定 signal_id 的全部审计记录。

        Args:
            signal_id: 幂等键，如 "2026-06-08:eod_buy_v1:model@v3"。

        Returns:
            匹配该 signal_id 的记录列表；无匹配时返回空列表。
        """
        return [e for e in self.read_all() if e.get("signal_id") == signal_id]
