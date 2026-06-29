"""自定义算法作为一等公民的条件挂单 + 无前视边界（Property 5）。

Feature: conditional-tick-policy, Requirement 4.1/7.2 · Property 5。

「触发条件」不限于高/低/平开等现有因子——任何只读 ``TickContext``（today 开盘 + 昨收 +
截至昨收的 ``hist`` + point-in-time ``signals``）的可调用对象都能直接作为 ``Rule.condition``。
本测试验证：
1. 端到端：一个自定义「均值回归超卖」算法能命中并选出对应非对称档位；
2. 结构无前视：``ctx`` 不暴露当日 high/low/close，且 ``ctx.hist`` 不含当日及未来 bar——
   自定义算法在结构上无法越界读到尚未发生的数据。
"""

from __future__ import annotations

from datetime import date, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.backtest.t0.tick_policy import (
    ConditionalTickPolicy, DailyBar, DailyHistory, Rule, TickContext,
)


def _hist_from_closes(closes: list[float], start: date) -> DailyHistory:
    """用一串收盘价造历史（每日开=收、振幅固定 0.2），日期自 start 递增。"""
    h = DailyHistory()
    for i, c in enumerate(closes):
        d = start + timedelta(days=i)
        h.append(DailyBar(d=d, open=c, high=c + 0.1, low=c - 0.1, close=c))
    return h


def test_custom_mean_reversion_algo_hits_end_to_end() -> None:
    """自定义超卖算法：今开较 5 日均收盘跌超阈值 → 判为超卖 → 买近卖远。"""

    def oversold(ctx: TickContext) -> bool:
        """today 开盘相对近 5 日均收盘的偏离 < −2% 视为超卖（只读 ctx）。"""
        bars = ctx.hist.bars[-5:]
        if len(bars) < 5:
            return False
        ma5 = sum(b.close for b in bars) / len(bars)
        return ma5 > 0 and (ctx.open / ma5 - 1.0) < -0.02

    rule = Rule("超卖买近卖远", oversold, lambda c: (0.08, 0.01))
    pol = ConditionalTickPolicy(rules=[rule], default=(0.03, 0.03))

    hist = _hist_from_closes([10.0] * 5, date(2024, 1, 1))  # 近 5 日均收盘 10.0
    # 今开 9.7（较 10.0 跌 3%）→ 超卖命中
    hit = TickContext(date(2024, 1, 8), 9.7, 10.0, hist, {})
    assert pol.ticks_for(hit) == (0.08, 0.01)
    # 今开 9.95（仅跌 0.5%）→ 不命中，回 default
    miss = TickContext(date(2024, 1, 8), 9.95, 10.0, hist, {})
    assert pol.ticks_for(miss) == (0.03, 0.03)


def test_cheating_algo_cannot_read_today_outcome() -> None:
    """企图读当日 high/low/close 的「作弊」算法只能拿到兜底 None：结构上无法越界。"""

    def cheat(ctx: TickContext) -> bool:
        # ctx 没有这些属性，getattr 兜底 None → 永远命中不了「今日大涨」这种前视条件
        today_close = getattr(ctx, "close", None)
        today_high = getattr(ctx, "high", None)
        return today_close is not None and today_high is not None

    pol = ConditionalTickPolicy(
        rules=[Rule("前视作弊", cheat, lambda c: (0.10, 0.01))],
        default=(0.02, 0.02),
    )
    ctx = TickContext(date(2024, 1, 8), 10.0, 10.0, _hist_from_closes([10.0] * 5, date(2024, 1, 1)), {})
    assert pol.ticks_for(ctx) == (0.02, 0.02)   # 作弊条件无法命中


@settings(max_examples=100)
@given(
    closes=st.lists(st.floats(min_value=5, max_value=15), min_size=1, max_size=40),
    open_px=st.floats(min_value=5, max_value=15),
)
def test_context_carries_no_same_day_or_future_bar(closes, open_px) -> None:
    """# Feature: conditional-tick-policy, Property 5: ctx 不含当日及未来 bar。

    策略在「今开」时刻构造的 ctx：hist 里每根 bar 的日期都严格早于 ctx.day，
    且 ctx 不暴露当日 high/low/close。任何自定义算法只能读到这份「截至昨收」的快照，
    结构上无法读取当日盘中或未来数据（无前视）。
    """
    start = date(2024, 1, 1)
    hist = _hist_from_closes(closes, start)
    # today 取历史最后一日的次日（模拟策略在新交易日开盘构造 ctx）
    today = start + timedelta(days=len(closes))
    ctx = TickContext(today, open_px, hist.bars[-1].close, hist, {})

    assert all(b.d < ctx.day for b in ctx.hist.bars)          # 无当日/未来 bar
    for attr in ("high", "low", "close"):                     # 不暴露当日盘中结果
        assert not hasattr(ctx, attr)
