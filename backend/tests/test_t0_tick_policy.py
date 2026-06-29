"""TickPolicy 族测试：固定 / 波动缩放 / 动量倾斜，及无前视的 DailyHistory 视图。

Feature: half-position-t0-backtest
"""

from __future__ import annotations

from datetime import date

from aitrade.backtest.t0.tick_policy import (
    DailyBar, DailyHistory, FixedTick, VolScaledTick, TrendTiltTick, TickContext,
)


def _hist(ranges: list[float], closes: list[float] | None = None) -> DailyHistory:
    h = DailyHistory()
    for i, r in enumerate(ranges):
        c = closes[i] if closes else 10.0
        h.append(DailyBar(d=date(2024, 1, 1 + i), open=c, high=c + r / 2, low=c - r / 2, close=c))
    return h


def _ctx(hist: DailyHistory) -> TickContext:
    """构造测试用上下文（open/prev_close 对档位无关的策略而言任意）。"""
    return TickContext(date(2024, 3, 1), 10.0, 10.0, hist, {})


def test_fixed_tick_returns_constants_asymmetric() -> None:
    s, b = FixedTick(sell_tick=0.02, buy_tick=0.03).ticks_for(_ctx(DailyHistory()))
    assert (s, b) == (0.02, 0.03)


def test_vol_scaled_tick_scales_with_range() -> None:
    hist = _hist([0.10] * 20)                       # 近 20 日均振幅 0.10
    s, b = VolScaledTick(k=0.4, n=20, pricetick=0.01).ticks_for(_ctx(hist))
    assert s == b == 0.04                           # round_to(0.4*0.10, 0.01)


def test_vol_scaled_tick_fallback_when_insufficient_history() -> None:
    hist = _hist([0.10] * 5)                         # < n=20
    s, b = VolScaledTick(k=0.4, n=20, pricetick=0.01, fallback=0.02).ticks_for(_ctx(hist))
    assert s == b == 0.02


def test_trend_tilt_buys_closer_in_uptrend() -> None:
    hist = _hist([0.06] * 10, closes=[10 + i * 0.2 for i in range(10)])   # 上行
    s, b = TrendTiltTick(base=0.02, tilt=0.01, n=5).ticks_for(_ctx(hist))
    assert b < s                                    # 顺势：上涨→买近(小buy_tick)卖远(大sell_tick)


def test_daily_history_mean_range_no_lookahead() -> None:
    """mean_range 仅聚合已 append 的历史，绝不含未来日。"""
    hist = _hist([0.10, 0.20, 0.30])
    assert abs(hist.mean_range(3) - 0.20) < 1e-9
    assert abs(hist.mean_range(2) - 0.25) < 1e-9    # 最近 2 日 (0.20,0.30)
