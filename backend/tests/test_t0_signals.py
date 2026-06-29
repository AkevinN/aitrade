"""SignalProvider 测试：point-in-time 信号读取（DictSignalProvider 桩 + AlphaFactor 桩）。

Feature: conditional-tick-policy, Requirement 4.2/4.3/4.5 · Property 6（信号 point-in-time、无前视）。
"""

from __future__ import annotations

from datetime import date, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.backtest.t0.signals import (
    AlphaFactorSignalProvider,
    DictSignalProvider,
    SignalProvider,
)


def test_dict_provider_returns_injected_value() -> None:
    """注入的 (标的,日,信号名) 命中时原样返回注入值。"""
    sp = DictSignalProvider({("000415.SZSE", date(2025, 1, 2), "mom"): 0.8})
    assert sp.value("000415.SZSE", date(2025, 1, 2), "mom") == 0.8


def test_dict_provider_missing_returns_none() -> None:
    """信号不可得（无该键）返回 None，供规则条件安全跳过。"""
    sp = DictSignalProvider({("000415.SZSE", date(2025, 1, 2), "mom"): 0.8})
    assert sp.value("000415.SZSE", date(2025, 1, 3), "mom") is None  # 别的日
    assert sp.value("600000.SSE", date(2025, 1, 2), "mom") is None   # 别的标的
    assert sp.value("000415.SZSE", date(2025, 1, 2), "vol") is None  # 别的信号名


def test_dict_provider_satisfies_protocol() -> None:
    """DictSignalProvider 满足 SignalProvider 协议（鸭子类型可替换）。"""
    assert isinstance(DictSignalProvider({}), SignalProvider)


def test_alpha_factor_provider_is_stub_returns_none() -> None:
    """AlphaFactorSignalProvider v1 桩：任何查询恒返回 None（未接线、不引入前视）。"""
    sp = AlphaFactorSignalProvider()
    assert sp.value("000415.SZSE", date(2025, 1, 2), "alpha012") is None
    assert isinstance(sp, SignalProvider)


@settings(max_examples=100)
@given(
    entries=st.lists(
        st.tuples(
            st.sampled_from(["A.SZSE", "B.SSE"]),
            st.integers(min_value=0, max_value=400),     # 距基准日的偏移天数
            st.sampled_from(["mom", "vol"]),
            st.floats(min_value=-5, max_value=5, allow_nan=False),
        ),
        max_size=30,
    ),
    probe_offset=st.integers(min_value=0, max_value=400),
)
def test_dict_provider_is_pure_lookup_no_cross_key_leak(entries, probe_offset) -> None:
    """# Feature: conditional-tick-policy, Property 6: 信号 point-in-time。

    DictSignalProvider.value 是表的**纯函数**：对任一查询，结果恒等于该键的注入值（或 None），
    绝不跨键插值、绝不回看其他日期。这保证 value(day) 只可能取到注入端为该 day 准备的（滞后）值，
    实现层不会把未来键的值“漏”到当前查询里。
    """
    base = date(2025, 1, 1)
    table: dict[tuple[str, date, str], float] = {}
    for sym, off, name, v in entries:
        table[(sym, base + timedelta(days=off), name)] = v
    sp = DictSignalProvider(table)

    probe_day = base + timedelta(days=probe_offset)
    for sym in ("A.SZSE", "B.SSE"):
        for name in ("mom", "vol"):
            key = (sym, probe_day, name)
            expected = table.get(key)  # 注入了就返回注入值，否则 None
            assert sp.value(sym, probe_day, name) == expected
