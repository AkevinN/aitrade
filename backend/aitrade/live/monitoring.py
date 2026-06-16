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
    """运行时状态汇总，持有实盘最新快照供看板（WS 推送）读取。

    收集运行过程中的持仓、账户资金、盈亏、成交与日志，对外通过 snapshot()
    导出一份扁平 dict。各字段由 update_* / add_trade / log 等方法增量更新，
    本身不做持久化，进程重启即清零。

    Attributes:
        positions: 当前持仓，vt_symbol -> 股数；零持仓的标的会被移除而非保留 0。
        cash: 当前可用现金（元）。
        realized_pnl: 已实现盈亏（元）。
        unrealized_pnl: 未实现（浮动）盈亏（元）。
        trades: 成交记录列表，每条为调用方约定格式的 dict。
        logs: 运行日志文本列表，按追加顺序保存，供看板展示。
    """
    positions: dict[str, float] = field(default_factory=dict)   # vt_symbol -> 股数
    cash: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)

    def update_position(self, vt_symbol: str, volume: float) -> None:
        """更新指定标的的持仓数量。

        Args:
            vt_symbol: 合约代码，如 "000001.SZSE"。
            volume:    新持仓股数；0 时从 positions 中移除该标的（避免零持仓残留）。
        """
        if volume == 0:
            self.positions.pop(vt_symbol, None)
        else:
            self.positions[vt_symbol] = volume

    def update_account(self, cash: float, realized_pnl: float = 0.0, unrealized_pnl: float = 0.0) -> None:
        """更新账户资金与盈亏快照。

        Args:
            cash:           当前可用现金（元）。
            realized_pnl:   已实现盈亏（元），默认 0。
            unrealized_pnl: 未实现盈亏（元），默认 0。
        """
        self.cash = cash
        self.realized_pnl = realized_pnl
        self.unrealized_pnl = unrealized_pnl

    def add_trade(self, trade: dict[str, Any]) -> None:
        """追加一条成交记录到 trades 列表。

        Args:
            trade: 成交记录 dict，格式由调用方约定（至少含 vt_symbol/direction/price/volume）。
        """
        self.trades.append(trade)

    def log(self, msg: str) -> None:
        """追加一条运行日志到 logs 列表（供看板展示）。

        Args:
            msg: 日志文本。
        """
        self.logs.append(msg)

    def total_pnl(self) -> float:
        """返回总盈亏（已实现 + 未实现）。"""
        return self.realized_pnl + self.unrealized_pnl

    def snapshot(self) -> dict[str, Any]:
        """返回当前状态快照（供 WS 推送到看板）。

        Returns:
            含 positions / cash / realized_pnl / unrealized_pnl / total_pnl / trade_count 的 dict。
        """
        return {
            "positions": dict(self.positions),
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl(),
            "trade_count": len(self.trades),
        }


class HeartbeatMonitor:
    """服务心跳健康检查。超时未心跳的服务视为不健康。

    各服务定期调用 beat()；is_healthy() 判断是否有服务超时。

    Example:
        >>> hb = HeartbeatMonitor(timeout_seconds=60)
        >>> hb.beat("signal_service")
        >>> hb.is_healthy()  # True
    """

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        """初始化心跳监控器。

        Args:
            timeout_seconds: 判定服务失活的超时阈值（秒），默认 60 秒。
        """
        self.timeout_seconds = timeout_seconds
        self._last: dict[str, datetime] = {}

    def beat(self, service: str, now: Optional[datetime] = None) -> None:
        """记录指定服务的最近一次心跳时刻。

        Args:
            service: 服务名称标识，如 "signal_service" / "orchestrator"。
            now:     心跳时刻参照；None 时使用 datetime.now()。
        """
        self._last[service] = now or datetime.now()

    def stale_services(self, now: Optional[datetime] = None) -> list[str]:
        """返回当前超时（失活）的服务名列表。

        Args:
            now: 判定基准时刻；None 时使用 datetime.now()。

        Returns:
            超过 timeout_seconds 未心跳的服务名列表；全部健康时返回空列表。
        """
        now = now or datetime.now()
        return [
            svc for svc, ts in self._last.items()
            if (now - ts).total_seconds() > self.timeout_seconds
        ]

    def is_healthy(self, now: Optional[datetime] = None) -> bool:
        """判断是否所有已注册服务均在健康窗口内。

        Args:
            now: 判定基准时刻；None 时使用 datetime.now()。

        Returns:
            True 表示无超时服务（全部健康），False 表示存在至少一个失活服务。
        """
        return len(self.stale_services(now)) == 0
