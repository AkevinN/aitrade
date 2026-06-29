"""声明式档位策略编译工厂测试：compile_tick_policy + 单条件规则语义 + 安全红线。

Feature: t0-conditional-tick-frontend
- Property 2：编译往返语义保持（首个命中/default/op 边界/档位夹取）。
- Property 3：只认白名单枚举、绝不执行任意代码（源码无 eval/exec/compile/__import__）。
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.backtest.engine import round_to
from aitrade.backtest.t0.policy_spec import (
    ConditionalCfg, FixedCfg, RuleCfg, TrendTiltCfg, VolScaledCfg,
    compile_tick_policy,
)
from aitrade.backtest.t0.tick_policy import (
    ConditionalTickPolicy, DailyBar, DailyHistory, FixedTick, TickContext,
    TrendTiltTick, VolScaledTick,
)

_OPS = {"gt": lambda x, t: x > t, "ge": lambda x, t: x >= t,
        "lt": lambda x, t: x < t, "le": lambda x, t: x <= t}


def _hist(n: int = 20) -> DailyHistory:
    h = DailyHistory()
    for i in range(n):
        c = 10.0 + 0.1 * (i % 3)
        h.append(DailyBar(date(2024, 1, 1) + (date(2024, 1, 1 + i) - date(2024, 1, 1)),
                          open=c, high=c + 0.1, low=c - 0.1, close=c))
    return h


def _ctx(gap_open=10.05, prev_close=10.0, hist=None, signals=None) -> TickContext:
    return TickContext(date(2024, 2, 1), gap_open, prev_close, hist or _hist(), signals or {})


def _round_pair(s, b, pt=0.01):
    return (max(pt, round_to(s, pt)), max(pt, round_to(b, pt)))


# ---- 基础策略透传 ----

def test_compile_fixed_passthrough() -> None:
    label, pol, names = compile_tick_policy(FixedCfg(label="固定2分", sell_tick=0.02, buy_tick=0.03))
    assert label == "固定2分" and names == ()
    assert isinstance(pol, FixedTick)
    assert pol.ticks_for(_ctx()) == (0.02, 0.03)


def test_compile_vol_scaled_passthrough() -> None:
    _, pol, _ = compile_tick_policy(VolScaledCfg(label="波动", k=0.4, n=20, fallback=0.02))
    assert isinstance(pol, VolScaledTick) and pol.k == 0.4 and pol.n == 20


def test_compile_trend_tilt_passthrough() -> None:
    _, pol, _ = compile_tick_policy(TrendTiltCfg(label="趋势", base=0.02, tilt=0.01, n=5))
    assert isinstance(pol, TrendTiltTick) and pol.base == 0.02 and pol.tilt == 0.01 and pol.n == 5


# ---- 条件策略：首个命中 / default / 非对称 ----

def test_compile_conditional_gap_first_match_then_default() -> None:
    cfg = ConditionalCfg(
        label="高低开", default_sell_tick=0.03, default_buy_tick=0.03,
        rules=[RuleCfg(name="高开", lhs="gap", op="gt", threshold=0.003, sell_tick=0.07, buy_tick=0.01),
               RuleCfg(name="低开", lhs="gap", op="lt", threshold=-0.003, sell_tick=0.09, buy_tick=0.01)])
    _, pol, names = compile_tick_policy(cfg)
    assert isinstance(pol, ConditionalTickPolicy) and names == ()
    assert pol.ticks_for(_ctx(10.10, 10.0)) == (0.07, 0.01)   # 高开
    assert pol.ticks_for(_ctx(9.90, 10.0)) == (0.09, 0.01)    # 低开
    assert pol.ticks_for(_ctx(10.00, 10.0)) == (0.03, 0.03)   # 平开→default


def test_signal_lhs_collects_names_and_reads_ctx_signals() -> None:
    cfg = ConditionalCfg(
        label="信号", default_sell_tick=0.03, default_buy_tick=0.03,
        rules=[RuleCfg(name="强", lhs="signal", op="gt", threshold=0.6,
                       signal_name="mdl_prob", sell_tick=0.08, buy_tick=0.01)])
    _, pol, names = compile_tick_policy(cfg)
    assert names == ("mdl_prob",) and pol.signal_names == ("mdl_prob",)
    assert pol.ticks_for(_ctx(signals={"mdl_prob": 0.7})) == (0.08, 0.01)   # 命中
    assert pol.ticks_for(_ctx(signals={"mdl_prob": 0.5})) == (0.03, 0.03)   # 不命中
    assert pol.ticks_for(_ctx(signals={"mdl_prob": None})) == (0.03, 0.03)  # 缺值安全跳过


def test_unknown_kind_raises() -> None:
    from types import SimpleNamespace
    with pytest.raises(ValueError):
        compile_tick_policy(SimpleNamespace(kind="evil", label="x"))


def test_source_has_no_dynamic_exec() -> None:
    """安全红线：编译工厂的**可执行代码**绝不引用 eval/exec/compile/__import__/globals/getattr。

    用 AST 扫描标识符（忽略 docstring/注释/字符串字面量——文档里写到这些词不算违规）。
    """
    src = Path("aitrade/backtest/t0/policy_spec.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"eval", "exec", "compile", "__import__", "globals", "getattr"}
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
    leaked = used & banned
    assert not leaked, f"工厂可执行代码不应引用动态执行原语：{leaked}"


@settings(max_examples=100, deadline=None)
@given(
    lhs=st.sampled_from(["gap", "mean_range", "momentum", "signal"]),
    op=st.sampled_from(["gt", "ge", "lt", "le"]),
    threshold=st.floats(min_value=-1, max_value=1, allow_nan=False),
    window=st.integers(min_value=1, max_value=20),
    sig_val=st.one_of(st.none(), st.floats(min_value=-2, max_value=2, allow_nan=False)),
    gap_open=st.floats(min_value=9.0, max_value=11.0),
    sell=st.floats(min_value=0.01, max_value=0.2),
    buy=st.floats(min_value=0.01, max_value=0.2),
)
def test_single_condition_roundtrip_matches_reference(lhs, op, threshold, window, sig_val, gap_open, sell, buy) -> None:
    """# Feature: t0-conditional-tick-frontend, Property 2: 编译往返语义保持。

    编译后的 ConditionalTickPolicy 对随机 ctx 的输出，须等于按声明语义直算的参考实现
    （单条件命中→该档，否则→default；档位夹到 ≥pricetick）。
    """
    hist = _hist()
    signals = {"s": sig_val}
    ctx = TickContext(date(2024, 2, 1), gap_open, 10.0, hist, signals)
    cfg = ConditionalCfg(
        label="t", default_sell_tick=0.03, default_buy_tick=0.04,
        rules=[RuleCfg(lhs=lhs, op=op, threshold=threshold, window=window,
                       signal_name="s", sell_tick=sell, buy_tick=buy)])
    _, pol, _ = compile_tick_policy(cfg)

    # 参考左值
    if lhs == "gap":
        x = ctx.gap
    elif lhs == "mean_range":
        x = hist.mean_range(window)
    elif lhs == "momentum":
        x = hist.momentum(window)
    else:
        x = sig_val
    hit = x is not None and _OPS[op](x, threshold)
    expected = _round_pair(sell, buy) if hit else _round_pair(0.03, 0.04)
    assert pol.ticks_for(ctx) == expected
