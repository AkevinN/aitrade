"""策略接入 SignalProvider 测试：HalfPositionT0Strategy 按 signal_names 填 ctx.signals。

Feature: conditional-tick-policy, Requirement 4.2/4.4/5.1 · Property 6（point-in-time）。
用「记录型」TickPolicy 捕获策略每日构造的 TickContext.signals，验证：
- 注入 DictSignalProvider 时，命名信号按 (标的, 当日, 信号名) 被读入 ctx.signals；
- 无 provider 时退化为空 signals（规则只能据今开/昨收/历史判断）。
"""

from __future__ import annotations

from datetime import date, datetime, time

from aitrade.backtest.engine import BacktestingEngine
from aitrade.backtest.types import BarData
from aitrade.backtest.t0.strategy import HalfPositionT0Strategy
from aitrade.backtest.t0.signals import DictSignalProvider


class _RecordingTick:
    """记录每次 ticks_for 收到的 ctx.signals 快照，返回固定档位（不影响撮合）。"""

    signal_names = ("mom",)

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def ticks_for(self, ctx) -> tuple[float, float]:
        self.seen.append(dict(ctx.signals))
        return (0.02, 0.02)


class _MemLoader:
    def __init__(self, bars): self._bars = bars
    def load_bar_data(self, *a): return list(self._bars)
    def load_contract_settings(self): return {}


def _day(d: int, open_px: float = 10.0):
    """造一个交易日的 3 根 1m bar（平开平收，仅用于触发挂单/收盘逻辑）。"""
    base = datetime(2024, 1, d)
    mk = lambda hh, mm: BarData(
        symbol="AAA", exchange="SSE", datetime=base.replace(hour=hh, minute=mm),
        interval="1m", open_price=open_px, high_price=open_px,
        low_price=open_px, close_price=open_px, volume=10000)
    return [mk(9, 30), mk(10, 30), mk(15, 0)]


def _run(policy, signal_provider):
    """跑两天，返回引擎；signal_provider 为 None 时不注入该 setting。"""
    days = [_day(1), _day(2)]
    bars = [b for day in days for b in day]
    vt = "AAA.SSE"
    eng = BacktestingEngine(data_loader=_MemLoader(bars))
    eng.set_parameters([vt], "1m", bars[0].datetime, bars[-1].datetime, capital=1_000_000)
    eng.sizes[vt] = 1
    eng.priceticks[vt] = 0.01
    eng.long_rates[vt] = eng.short_rates[vt] = eng.stamp_duties[vt] = eng.slippages[vt] = 0.0
    eng.limit_ratios[vt] = None
    setting = {"vt_symbol": vt, "tick_policy": policy, "swing_frac": 1.0,
               "base_weight": 0.5, "close_time": time(14, 57)}
    if signal_provider is not None:
        setting["signal_provider"] = signal_provider
    eng.add_strategy(HalfPositionT0Strategy, setting, None)
    eng.load_data()
    eng.run_backtesting()
    return eng


def test_strategy_fills_ctx_signals_from_provider() -> None:
    """注入 DictSignalProvider：每日 ctx.signals 取到 (标的, 当日, 'mom') 的注入值。"""
    pol = _RecordingTick()
    sp = DictSignalProvider({
        ("AAA.SSE", date(2024, 1, 1), "mom"): 0.7,
        ("AAA.SSE", date(2024, 1, 2), "mom"): -0.3,
    })
    _run(pol, sp)
    assert pol.seen[0] == {"mom": 0.7}      # 第 1 天
    assert pol.seen[1] == {"mom": -0.3}     # 第 2 天，按当日键取值（point-in-time）


def test_strategy_signals_empty_without_provider() -> None:
    """无 signal_provider：退化为空 signals，规则无法据信号判断。"""
    pol = _RecordingTick()
    _run(pol, None)
    assert pol.seen and all(s == {} for s in pol.seen)


def test_strategy_signal_none_when_not_injected() -> None:
    """provider 存在但该日信号缺失：ctx.signals 该名取到 None（规则据此安全跳过）。"""
    pol = _RecordingTick()
    sp = DictSignalProvider({("AAA.SSE", date(2024, 1, 1), "mom"): 0.7})  # 仅第 1 天有
    _run(pol, sp)
    assert pol.seen[0] == {"mom": 0.7}
    assert pol.seen[1] == {"mom": None}     # 第 2 天缺失 → None
