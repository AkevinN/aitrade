"""引擎 Fill_Policy 撮合钩子测试（做T成交保真度旋钮）。

红线：默认值（fill_penetration=0, fill_ratio=1.0 / fill_policy=None）下撮合行为与改造前
逐字节一致（Property 2）。穿越 ε 要求价格穿过委托价才成交；fill_ratio<1 部分成交。

Feature: half-position-t0-backtest
"""

from __future__ import annotations

from datetime import datetime, timedelta

from aitrade.backtest.engine import BacktestingEngine
from aitrade.backtest.strategy import BaseStrategy
from aitrade.backtest.types import BarData, FillPolicy


class _MemoryLoader:
    """最小 BarDataLoader：内存行情 + 空合约配置。"""

    def __init__(self, bars: list[BarData]) -> None:
        self._bars = bars

    def load_bar_data(self, vt_symbol, interval, start, end) -> list[BarData]:
        return list(self._bars)

    def load_contract_settings(self) -> dict:
        return {}


class _BuyProbeStrategy(BaseStrategy):
    """探针策略：仅在首根 bar 挂一张买入限价单，便于直接观察撮合。"""

    order_price: float = 100.0
    order_volume: float = 1000.0

    def on_init(self) -> None:
        self._placed = False

    def on_bars(self, bars: dict[str, BarData]) -> None:
        if not self._placed:
            vt = self.vt_symbols[0]
            self.buy(vt, self.order_price, self.order_volume)
            self._placed = True

    def on_trade(self, trade) -> None:
        pass


def _bars(lows: list[float], highs: list[float], symbol="AAA", exchange="SSE") -> list[BarData]:
    base = datetime(2024, 1, 1)
    out: list[BarData] = []
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        out.append(BarData(symbol=symbol, exchange=exchange, datetime=base + timedelta(days=i),
                           interval="d", open_price=hi, high_price=hi, low_price=lo,
                           close_price=(hi + lo) / 2, volume=10000))
    return out


def _run(bars: list[BarData], fill_policy: FillPolicy | None,
         order_price=100.0, order_volume=1000.0):
    vt = "AAA.SSE"
    engine = BacktestingEngine(data_loader=_MemoryLoader(bars))
    engine.set_parameters([vt], "d", bars[0].datetime, bars[-1].datetime, capital=10_000_000)
    engine.sizes[vt] = 1
    engine.priceticks[vt] = 0.01
    engine.long_rates[vt] = 0.0
    engine.short_rates[vt] = 0.0
    engine.stamp_duties[vt] = 0.0
    engine.slippages[vt] = 0.0
    engine.limit_ratios[vt] = None        # 关闭涨跌停限制，隔离撮合行为
    engine.fill_policy = fill_policy
    engine.add_strategy(_BuyProbeStrategy, {"order_price": order_price, "order_volume": order_volume}, None)
    engine.load_data()
    engine.run_backtesting()
    return engine.get_all_trades()


def test_default_fill_policy_is_byte_compatible() -> None:
    """Property 2：fill_policy=None 与 FillPolicy(0, 1.0) 成交序列逐字节一致。"""
    # 首根挂单，第二根 low=99.5 穿过买价 100 → 成交
    bars = _bars(lows=[100.0, 99.5, 99.0], highs=[100.0, 101.0, 101.0])
    none_trades = _run(bars, fill_policy=None)
    default_trades = _run(bars, fill_policy=FillPolicy())
    assert len(none_trades) == len(default_trades) == 1
    a, b = none_trades[0], default_trades[0]
    assert (a.price, a.volume, a.direction) == (b.price, b.volume, b.direction)
    assert a.price == 100.0 and a.volume == 1000.0


def test_penetration_requires_cross_not_touch() -> None:
    """穿越 ε：仅触价（low==委托价）不成交；穿过 ε 才成交。"""
    # 第二根 low==100.0 恰好触价但不穿过
    bars = _bars(lows=[100.0, 100.0, 100.0], highs=[100.0, 101.0, 101.0])
    touch = _run(bars, fill_policy=FillPolicy(fill_penetration=0.0))
    assert len(touch) == 1, "ε=0 触价即成交"
    pen = _run(bars, fill_policy=FillPolicy(fill_penetration=0.01))
    assert len(pen) == 0, "ε=0.01 仅触价不穿过 → 不成交"
    # 第二根 low=99.98 穿过 100-0.01=99.99 → 成交
    bars2 = _bars(lows=[100.0, 99.98, 99.0], highs=[100.0, 101.0, 101.0])
    pen2 = _run(bars2, fill_policy=FillPolicy(fill_penetration=0.01))
    assert len(pen2) == 1, "穿过 ε → 成交"


def test_fill_ratio_partial_fills_across_bars() -> None:
    """fill_ratio=0.5：每根触价 bar 仅成交原始量的一半，跨 bar 成交完。"""
    bars = _bars(lows=[100.0, 99.5, 99.5, 99.5], highs=[100.0, 101.0, 101.0, 101.0])
    trades = _run(bars, fill_policy=FillPolicy(fill_ratio=0.5), order_volume=1000.0)
    assert len(trades) == 2, "1000 股按每根 500 分两根成交"
    assert all(t.volume == 500.0 for t in trades)
    assert sum(t.volume for t in trades) == 1000.0
