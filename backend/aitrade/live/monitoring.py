"""
监控（迭代 7）：运行时状态汇总 + 心跳健康检查。

- MonitorHub：汇总最新持仓/账户/成交/日志，供看板（WS 推送）读取。
- HeartbeatMonitor：各服务定期心跳，超时判定不健康并可触发告警。
真实 WS 推送只需把 MonitorHub.snapshot() 周期性广播到已定义主题即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class MonitorHub:
    """运行时状态汇总（最新快照）。"""
    positions: dict[str, float] = field(default_factory=dict)   # vt_symbol -> 股数
    cash: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)

    def update_position(self, vt_symbol: str, volume: float) -> None:
        if volume == 0:
            self.positions.pop(vt_symbol, None)
        else:
            self.positions[vt_symbol] = volume

    def update_account(self, cash: float, realized_pnl: float = 0.0, unrealized_pnl: float = 0.0) -> None:
        self.cash = cash
        self.realized_pnl = realized_pnl
        self.unrealized_pnl = unrealized_pnl

    def add_trade(self, trade: dict[str, Any]) -> None:
        self.trades.append(trade)

    def log(self, msg: str) -> None:
        self.logs.append(msg)

    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    def snapshot(self) -> dict[str, Any]:
        return {
            "positions": dict(self.positions),
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl(),
            "trade_count": len(self.trades),
        }


class HeartbeatMonitor:
    """服务心跳健康检查。超时未心跳的服务视为不健康。"""

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._last: dict[str, datetime] = {}

    def beat(self, service: str, now: Optional[datetime] = None) -> None:
        self._last[service] = now or datetime.now()

    def stale_services(self, now: Optional[datetime] = None) -> list[str]:
        now = now or datetime.now()
        return [
            svc for svc, ts in self._last.items()
            if (now - ts).total_seconds() > self.timeout_seconds
        ]

    def is_healthy(self, now: Optional[datetime] = None) -> bool:
        return len(self.stale_services(now)) == 0
