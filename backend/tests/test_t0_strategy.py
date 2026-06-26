"""HalfPositionT0Strategy 集成测试：在引擎@1m 上跑受控的日内路径，验证做T流程。

Feature: half-position-t0-backtest
"""

from __future__ import annotations

from datetime import datetime, time

from aitrade.backtest.engine import BacktestingEngine
from aitrade.backtest.types import BarData, Direction
from aitrade.backtest.t0.strategy import HalfPositionT0Strategy
from aitrade.backtest.t0.tick_policy import FixedTick


class _MemLoader:
    def __init__(self, bars): self._bars = bars
    def load_bar_data(self, *a): return list(self._bars)
    def load_contract_settings(self): return {}


def _day(d: int, intraday_high: float, intraday_low: float, close: float, open_px=10.0):
    """造一个交易日的 3 根 1m bar：09:30 开盘、10:30 日内高低、15:00 收盘。"""
    base = datetime(2024, 1, d)
    mk = lambda hh, mm, o, h, l, c: BarData(
        symbol="AAA", exchange="SSE", datetime=base.replace(hour=hh, minute=mm),
        interval="1m", open_price=o, high_price=h, low_price=l, close_price=c, volume=10000)
    return [
        mk(9, 30, open_px, open_px, open_px, open_px),               # 开盘：挂单
        mk(10, 30, open_px, intraday_high, intraday_low, open_px),   # 日内：撮合
        mk(15, 0, close, close, close, close),                       # 收盘：回半仓
    ]


def _run(days_bars):
    bars = [b for day in days_bars for b in day]
    vt = "AAA.SSE"
    eng = BacktestingEngine(data_loader=_MemLoader(bars))
    eng.set_parameters([vt], "1m", bars[0].datetime, bars[-1].datetime, capital=1_000_000)
    eng.sizes[vt] = 1
    eng.priceticks[vt] = 0.01
    eng.long_rates[vt] = 0.0
    eng.short_rates[vt] = 0.0
    eng.stamp_duties[vt] = 0.0
    eng.slippages[vt] = 0.0
    eng.limit_ratios[vt] = None
    eng.add_strategy(HalfPositionT0Strategy, {
        "vt_symbol": vt,
        "tick_policy": FixedTick(sell_tick=0.02, buy_tick=0.02),
        "swing_frac": 1.0, "base_weight": 0.5,
        "close_time": time(14, 57),
    }, None)
    eng.load_data()
    eng.run_backtesting()
    return eng


def test_day1_builds_half_position() -> None:
    """第一天建半仓：约 50% 资产买成股票。"""
    eng = _run([_day(1, 10.0, 10.0, 10.0)])
    pos = eng.strategy.get_pos("AAA.SSE")
    assert 49000 <= pos <= 51000, pos          # ~50,000 股 = 半仓


def test_only_sell_day_sells_high_and_rebuys_at_close() -> None:
    """只触卖：日内涨破卖价→卖出，收盘买回，日终回到约半仓。"""
    eng = _run([
        _day(1, 10.0, 10.0, 10.0),                       # 建半仓
        _day(2, intraday_high=10.05, intraday_low=9.99, close=10.01),  # 只触卖(10.02)
    ])
    trades = eng.get_all_trades()
    sells = [t for t in trades if t.direction == Direction.SHORT]
    assert any(abs(t.price - 10.02) < 1e-6 for t in sells), "应有卖在 10.02 的成交"
    # 日终回半仓
    pos = eng.strategy.get_pos("AAA.SSE")
    assert 49000 <= pos <= 51000, pos
