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
    """单标的持仓对账差异记录。

    Attributes:
        vt_symbol:   合约代码。
        theoretical: 系统理论持仓（策略/风控账本计算值）。
        actual:      实盘实际持仓（网关 query_positions 返回值）。
    """

    vt_symbol: str
    theoretical: float
    actual: float

    @property
    def diff(self) -> float:
        """实盘 - 理论 = 持仓偏差；正值表示实盘多于理论，负值表示少于理论。"""
        return self.actual - self.theoretical


@dataclass
class ReconciliationResult:
    """综合对账结果。

    Attributes:
        position_diffs: 超容忍度的逐标的持仓差异列表。
        value_diff:     市值/盈亏差异（实盘 - 理论，元）。
        alerts:         人工可读的告警文本列表；空列表表示无差异。
        should_block:   是否建议阻断下一轮自动下单（任一告警 → True）。
    """

    position_diffs: list[PositionDiff] = field(default_factory=list)
    value_diff: float = 0.0
    alerts: list[str] = field(default_factory=list)
    should_block: bool = False

    @property
    def ok(self) -> bool:
        """无任何告警时为 True（对账通过）。"""
        return not self.alerts


def reconcile_positions(
    theoretical: dict[str, float],
    actual: dict[str, float],
    qty_tolerance: float = 0.0,
) -> list[PositionDiff]:
    """逐标的对比持仓，返回超容忍度的差异明细。

    合并 theoretical 与 actual 的全部标的后逐一比较；仅返回 |actual - theoretical| > tolerance 的项。

    Args:
        theoretical:   系统理论持仓 {vt_symbol: 股数}。
        actual:        实盘实际持仓 {vt_symbol: 股数}。
        qty_tolerance: 允许的持仓偏差股数（绝对值），默认 0 即严格相等。

    Returns:
        超容忍度的 PositionDiff 列表，按 vt_symbol 升序排列；无差异时返回空列表。
    """
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
    """综合对账：持仓 + 市值/盈亏。任一超阈值 → 告警并建议阻断自动下单。

    持仓对账委托 reconcile_positions；市值对账单独比较 |actual_value - theoretical_value|。
    任何告警均使 should_block=True，调用方据此阻断 RiskGuardedGateway（block_for_reconcile）。

    Args:
        theoretical_positions: 系统理论持仓 {vt_symbol: 股数}。
        actual_positions:      实盘实际持仓 {vt_symbol: 股数}。
        theoretical_value:     理论总市值/盈亏（元）。
        actual_value:          实盘总市值/盈亏（元）。
        qty_tolerance:         持仓容忍股数偏差（默认 0）。
        value_tolerance:       市值容忍偏差（元，默认 0）。

    Returns:
        ReconciliationResult；无差异时 ok=True, should_block=False。
    """
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
