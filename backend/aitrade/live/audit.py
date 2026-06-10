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
    def __init__(self, path: Path | str) -> None:
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
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def for_signal(self, signal_id: str) -> list[dict[str, Any]]:
        return [e for e in self.read_all() if e.get("signal_id") == signal_id]
