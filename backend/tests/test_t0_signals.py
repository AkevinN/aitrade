"""SignalProvider 测试：point-in-time 信号读取（DictSignalProvider 桩 + AlphaFactor 桩）。

Feature: conditional-tick-policy, Requirement 4.2/4.3/4.5 · Property 6（信号 point-in-time、无前视）。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.backtest.t0.signals import (
    AlphaFactorSignalProvider,
    DictSignalProvider,
    LabSignalProvider,
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


# ---- LabSignalProvider：持久化模型信号的 point-in-time 取值（Property 4） ----


def _signal_frame(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    """造一张信号帧：rows = [(vt_symbol, 'YYYY-MM-DD', value), ...]，列为 datetime/vt_symbol/signal。"""
    return pl.DataFrame({
        "datetime": [datetime.fromisoformat(d) for _, d, _ in rows],
        "vt_symbol": [s for s, _, _ in rows],
        "signal": [v for _, _, v in rows],
    })


def test_lab_provider_returns_latest_strictly_before_day() -> None:
    """value(symbol, day, name) 只取该标的 date < day 的最近一行（滞后、无前视）。"""
    frame = _signal_frame([
        ("000415.SZSE", "2025-01-02", 0.1),
        ("000415.SZSE", "2025-01-03", 0.2),
        ("000415.SZSE", "2025-01-06", 0.3),
    ])
    sp = LabSignalProvider.from_frames({"mom": frame})
    assert sp.value("000415.SZSE", date(2025, 1, 2), "mom") is None   # 无更早行
    assert sp.value("000415.SZSE", date(2025, 1, 3), "mom") == 0.1    # < 1/3 最近=1/2
    assert sp.value("000415.SZSE", date(2025, 1, 6), "mom") == 0.2    # < 1/6 最近=1/3
    assert sp.value("000415.SZSE", date(2025, 1, 7), "mom") == 0.3    # < 1/7 最近=1/6


def test_lab_provider_missing_symbol_name_returns_none() -> None:
    """缺标的/缺信号名/空帧 → None。"""
    sp = LabSignalProvider.from_frames({"mom": _signal_frame([("A.SZSE", "2025-01-02", 0.5)])})
    assert sp.value("B.SSE", date(2025, 1, 5), "mom") is None    # 别的标的
    assert sp.value("A.SZSE", date(2025, 1, 5), "vol") is None   # 别的信号名
    assert LabSignalProvider.from_frames({"empty": _signal_frame([])}).value("A.SZSE", date(2025, 1, 5), "empty") is None


def test_lab_provider_isolates_symbols() -> None:
    """不同标的的信号互不串台。"""
    frame = _signal_frame([("A.SZSE", "2025-01-02", 1.0), ("B.SSE", "2025-01-02", 2.0)])
    sp = LabSignalProvider.from_frames({"s": frame})
    assert sp.value("A.SZSE", date(2025, 1, 3), "s") == 1.0
    assert sp.value("B.SSE", date(2025, 1, 3), "s") == 2.0


def test_lab_provider_intraday_takes_latest_of_prior_date() -> None:
    """同日多行（盘中）时，取 date<day 中最新日期的最后一行。"""
    frame = _signal_frame([
        ("A.SZSE", "2025-01-02", 0.1),
        ("A.SZSE", "2025-01-02", 0.15),   # 同日更晚一行
        ("A.SZSE", "2025-01-03", 0.2),
    ])
    sp = LabSignalProvider.from_frames({"s": frame})
    assert sp.value("A.SZSE", date(2025, 1, 3), "s") == 0.15   # < 1/3 最新日 1/2 的最后一行


def test_lab_provider_satisfies_protocol() -> None:
    assert isinstance(LabSignalProvider.from_frames({}), SignalProvider)


@settings(max_examples=100)
@given(
    entries=st.lists(
        st.tuples(st.sampled_from(["A.SZSE", "B.SSE"]),
                  st.integers(min_value=0, max_value=60),
                  st.floats(min_value=-3, max_value=3, allow_nan=False)),
        max_size=30),
    probe_off=st.integers(min_value=0, max_value=60),
)
def test_lab_provider_matches_reference(entries, probe_off) -> None:
    """# Feature: t0-conditional-tick-frontend, Property 4: 信号 point-in-time。

    value(symbol, day, name) 须等于「该标的所有 date<day 行中、日期最大者的值」（同日取最后一行）。
    """
    base = date(2025, 1, 1)
    rows = [(sym, (base + timedelta(days=off)).isoformat(), v) for sym, off, v in entries]
    sp = LabSignalProvider.from_frames({"s": _signal_frame(rows)}) if rows else LabSignalProvider.from_frames({"s": _signal_frame([])})
    probe = base + timedelta(days=probe_off)
    for sym in ("A.SZSE", "B.SSE"):
        # 参考：该标的所有 date<probe 行，按 (date) 取最大日期、同日取输入顺序最后一行
        cand = [(off, v) for s, off, v in entries if s == sym and (base + timedelta(days=off)) < probe]
        expected = None
        if cand:
            max_off = max(o for o, _ in cand)
            expected = [v for o, v in cand if o == max_off][-1]
        assert sp.value(sym, probe, "s") == expected
