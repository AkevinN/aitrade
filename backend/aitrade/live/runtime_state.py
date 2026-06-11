"""
运行时状态持久化（迭代 10）：进程重启可恢复，避免重复触发/重复下单。

典型用途：持久化调度器的 last_triggered_date，重启后据此判断当日是否已触发。
权威持仓/订单仍以网关查询为准；本存储只放调度/运行控制类轻状态。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RuntimeStateStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def save(self, state: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self.load().get(key, default)

    def set(self, key: str, value: Any) -> None:
        state = self.load()
        state[key] = value
        self.save(state)
