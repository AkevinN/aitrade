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
    """进程重启可恢复的运行时轻状态持久化（单 JSON 文件，键值存储）。

    用于调度器 Last_Triggered_Map 等调度/运行控制类轻状态。
    权威持仓/订单仍以网关查询为准；本存储只放不敏感的调度辅助状态。

    文件损坏（JSON 非法）时 `load` 返回空 dict（宽容降级，避免进程崩溃）。

    Example:
        >>> store = RuntimeStateStore("/tmp/state.json")
        >>> store.set("plan_last_triggered", {"plan1": "2026-06-08"})
        >>> store.get("plan_last_triggered")
        {'plan1': '2026-06-08'}
    """

    def __init__(self, path: Path | str) -> None:
        """初始化 RuntimeStateStore。

        Args:
            path: JSON 文件路径；父目录不存在时自动创建。
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        """读取全部状态（每次读取均重新从磁盘加载）。

        Returns:
            状态 dict；文件不存在或 JSON 损坏时返回空 dict。
        """
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def save(self, state: dict[str, Any]) -> None:
        """将完整状态写入 JSON 文件（覆盖）。

        Args:
            state: 要持久化的完整状态 dict。
        """
        self.path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        """读取单个键值，不存在时返回 default。

        Args:
            key:     状态键名。
            default: 键不存在时的默认值，默认 None。

        Returns:
            对应值或 default。
        """
        return self.load().get(key, default)

    def set(self, key: str, value: Any) -> None:
        """原子式更新单个键并持久化（读-改-写，不加锁，单进程场景安全）。

        Args:
            key:   状态键名。
            value: 新值（任意 JSON 可序列化类型）。
        """
        state = self.load()
        state[key] = value
        self.save(state)
