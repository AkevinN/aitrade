"""
每日对账（迭代 7）：理论 vs 实盘，差异超阈值告警并阻断下一轮自动下单。

- reconcile_positions：逐标的对比理论/实盘持仓，返回差异明细。
- reconcile_value：对比理论/实盘 PnL 或市值。
- ReconciliationResult.should_block：差异超阈值时为 True，调用方据此阻断自动交易。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PositionDiff:
    vt_symbol: str
    theoretical: float
    actual: float

    @property
    def diff(self) -> float:
        return self.actual - self.theoretical


@dataclass
class ReconciliationResult:
    position_diffs: list[PositionDiff] = field(default_factory=list)
    value_diff: float = 0.0
    alerts: list[str] = field(default_factory=list)
    should_block: bool = False

    @property
    def ok(self) -> bool:
        return not self.alerts


def reconcile_positions(
    theoretical: dict[str, float],
    actual: dict[str, float],
    qty_tolerance: float = 0.0,
) -> list[PositionDiff]:
    """逐标的对比持仓，返回超容忍度的差异明细。"""
    symbols = set(theoretical) | set(actual)
    diffs: list[PositionDiff] = []
    for sym in sorted(symbols):
        t = float(theoretical.get(sym, 0.0))
        a = float(actual.get(sym, 0.0))
        if abs(a - t) > qty_tolerance:
            diffs.append(PositionDiff(sym, t, a))
    return diffs


def reconcile(
    theoretical_positions: dict[str, float],
    actual_positions: dict[str, float],
    theoretical_value: float = 0.0,
    actual_value: float = 0.0,
    qty_tolerance: float = 0.0,
    value_tolerance: float = 0.0,
) -> ReconciliationResult:
    """综合对账：持仓 + 市值/盈亏。任一超阈值 → 告警并建议阻断自动下单。"""
    result = ReconciliationResult()
    result.position_diffs = reconcile_positions(
        theoretical_positions, actual_positions, qty_tolerance
    )
    for d in result.position_diffs:
        result.alerts.append(
            f"持仓不一致：{d.vt_symbol} 理论={d.theoretical} 实盘={d.actual} 差={d.diff}"
        )

    result.value_diff = actual_value - theoretical_value
    if abs(result.value_diff) > value_tolerance:
        result.alerts.append(
            f"市值/盈亏不一致：理论={theoretical_value} 实盘={actual_value} 差={result.value_diff}"
        )

    result.should_block = len(result.alerts) > 0
    return result
