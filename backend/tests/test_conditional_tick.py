"""条件挂单档位控制：TickContext / ConditionalTickPolicy / Rule / gap_rules。

Feature: conditional-tick-policy
"""

from __future__ import annotations

from datetime import date

from aitrade.backtest.t0.tick_policy import (
    TickContext, DailyHistory, DailyBar, Rule, ConditionalTickPolicy, gap_rules,
)


def _ctx(open_=10.1, prev_close=10.0, hist=None, signals=None) -> TickContext:
    return TickContext(day=date(2024, 2, 1), open=open_, prev_close=prev_close,
                       hist=hist or DailyHistory(), signals=signals or {})


def test_tick_context_gap() -> None:
    # Feature: conditional-tick-policy, Property 1: gap 只由今开/昨收算
    assert abs(_ctx(10.1, 10.0).gap - 0.01) < 1e-9
    assert _ctx(10.0, 10.0).gap == 0.0
    assert _ctx(9.9, 10.0).gap < 0
    assert _ctx(10.0, 0.0).gap == 0.0          # 昨收为0时安全退化


def test_conditional_first_match_then_default() -> None:
    # Feature: conditional-tick-policy, Property 2: 首个命中→default 回退
    rules = [
        Rule("a", lambda c: c.gap > 0.005, lambda c: (0.07, 0.01)),
        Rule("b", lambda c: c.gap < -0.005, lambda c: (0.09, 0.01)),
    ]
    p = ConditionalTickPolicy(rules=rules, default=(0.03, 0.02))
    assert p.ticks_for(_ctx(10.1, 10.0)) == (0.07, 0.01)
    assert p.ticks_for(_ctx(9.9, 10.0)) == (0.09, 0.01)
    assert p.ticks_for(_ctx(10.0, 10.0)) == (0.03, 0.02)


def test_gap_rules_high_low_flat() -> None:
    p = ConditionalTickPolicy(rules=gap_rules(thresh=0.003, up=(0.07, 0.01), down=(0.09, 0.01)),
                              default=(0.03, 0.02))
    assert p.ticks_for(_ctx(10.05, 10.0)) == (0.07, 0.01)   # +0.5% 高开
    assert p.ticks_for(_ctx(9.95, 10.0)) == (0.09, 0.01)    # −0.5% 低开
    assert p.ticks_for(_ctx(10.0, 10.0)) == (0.03, 0.02)    # 平开→default


def test_ticks_rounded_and_floored() -> None:
    # Feature: conditional-tick-policy, Property 4: 档位对齐且≥最小价位
    p = ConditionalTickPolicy(rules=[Rule("x", lambda c: True, lambda c: (0.0234, 0.0))],
                              default=(0.02, 0.02), pricetick=0.01)
    s, b = p.ticks_for(_ctx())
    assert s == 0.02       # round_to(0.0234, 0.01)
    assert b == 0.01       # 0.0 夹到最小价位


def test_custom_algo_rule_reads_hist() -> None:
    # Feature: conditional-tick-policy, Property 5: 自定义算法只读 ctx.hist
    h = DailyHistory()
    for i in range(5):
        h.append(DailyBar(date(2024, 1, 1 + i), 10.0, 10.2, 9.8, 10.0))   # 振幅0.4
    rule = Rule("超卖算法", lambda c: (c.hist.mean_range(3) or 0) > 0.3, lambda c: (0.10, 0.01))
    p = ConditionalTickPolicy(rules=[rule], default=(0.02, 0.02))
    assert p.ticks_for(_ctx(hist=h)) == (0.10, 0.01)


def test_tick_context_exposes_no_future() -> None:
    # Feature: conditional-tick-policy, Property 5: ctx 不暴露当日 high/low/close
    c = _ctx()
    for attr in ("high", "low", "close"):
        assert not hasattr(c, attr)
