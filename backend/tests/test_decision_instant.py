"""
决策时刻统一（Decision Instant Unification）核心原语属性测试（Wave 1，纯新增）。

覆盖 DecisionInstant / session_close / select_decision_bar / make_signal_id 的
正确性属性，外部无 I/O，确定性。属性见 .kiro/specs/decision-instant-unification/design.md。
"""

from __future__ import annotations

from datetime import date, datetime, time

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.live.decision_instant import (
    INTRADAY_BAR_FREQS,
    SESSION_CLOSE,
    SESSIONS,
    SUPPORTED_BAR_FREQS,
    DecisionInstant,
    bar_close_grid,
    bar_freq_of_interval,
    decision_bar_datetime,
    interval_of_bar_freq,
    make_signal_id,
    select_decision_bar,
    session_close,
)

_dates = st.dates(min_value=date(2024, 1, 1), max_value=date(2027, 12, 31))
_datetimes = st.datetimes(min_value=datetime(2024, 1, 1), max_value=datetime(2027, 12, 31))
_idents = st.text(alphabet="abcdefghijkmnopqrstuvwxyz_0123456789", min_size=1, max_size=10)


def _daily_frame(days: list[date]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [datetime.combine(d, time(0, 0)) for d in days],
            "vt_symbol": ["000001.SZSE"] * len(days),
            "signal": [0.5] * len(days),
        }
    )


# ---------------------------------------------------------------------------
# session_close
# ---------------------------------------------------------------------------
def test_session_close_1d_is_market_close() -> None:
    assert session_close(date(2026, 6, 8), "1d") == datetime(2026, 6, 8, SESSION_CLOSE.hour, 0)


def test_session_close_all_freqs_is_market_close() -> None:
    """任意受支持频率下，当日最后一根 bar 的收盘时刻都是 15:00。"""
    for freq in SUPPORTED_BAR_FREQS:
        assert session_close(date(2026, 6, 8), freq) == datetime(2026, 6, 8, 15, 0)


def test_session_close_unsupported_freq_rejected() -> None:
    with pytest.raises(ValueError):
        session_close(date(2026, 6, 8), "2h")


# ---------------------------------------------------------------------------
# Property P1: bar 收盘时刻网格正确性
# ---------------------------------------------------------------------------
# Feature: intraday-monitoring-decision, Property 1: bar_close_grid 网格正确
# 输出严格升序、全部在 (09:30,11:30] ∪ (13:00,15:00] 内、同一时段内相邻差 = freq、
# 必含 11:30 与 15:00；"1d" = (15:00,)；不受支持频率抛 ValueError。
# Validates: Requirements 1.3
def test_property_p1_bar_close_grid() -> None:
    assert bar_close_grid("1d") == (time(15, 0),)
    for freq in INTRADAY_BAR_FREQS:
        grid = bar_close_grid(freq)
        minutes = int(freq[:-1])
        assert list(grid) == sorted(set(grid))  # 严格升序无重复
        assert time(11, 30) in grid and time(15, 0) in grid
        for t in grid:
            assert any(o < t <= c for o, c in SESSIONS)
        # 同一时段内相邻差 = freq（受支持频率均整除 120 分钟时段）
        for a, b in zip(grid, grid[1:]):
            am, bm = a.hour * 60 + a.minute, b.hour * 60 + b.minute
            same_session = any(o < a <= c and o < b <= c for o, c in SESSIONS)
            if same_session:
                assert bm - am == minutes
    with pytest.raises(ValueError):
        bar_close_grid("2h")


def test_p1_grid_examples() -> None:
    assert bar_close_grid("60m") == (time(10, 30), time(11, 30), time(14, 0), time(15, 0))
    assert bar_close_grid("30m")[:2] == (time(10, 0), time(10, 30))
    assert len(bar_close_grid("5m")) == 48
    assert len(bar_close_grid("1m")) == 240


# ---------------------------------------------------------------------------
# Property P8: interval ↔ bar_freq 双向映射
# ---------------------------------------------------------------------------
# Feature: intraday-monitoring-decision, Property 8: 映射双射
# interval_of_bar_freq(bar_freq_of_interval(x)) == x 对所有受支持 interval 成立；
# 不受支持的值抛 ValueError。
# Validates: Requirements 1.4
def test_property_p8_interval_mapping_bijection() -> None:
    for interval in ("d", *INTRADAY_BAR_FREQS):
        assert interval_of_bar_freq(bar_freq_of_interval(interval)) == interval
    assert bar_freq_of_interval("d") == "1d"
    assert interval_of_bar_freq("1d") == "d"
    for bad in ("w", "2h", "", "1D"):
        with pytest.raises(ValueError):
            bar_freq_of_interval(bad)
        with pytest.raises(ValueError):
            interval_of_bar_freq(bad)


# ---------------------------------------------------------------------------
# Property P2: 日内无前视（end-labeled close_time <= as_of）
# ---------------------------------------------------------------------------
# Feature: intraday-monitoring-decision, Property 2: 日内无前视
# 任意分钟 frame 与 as_of，select_decision_bar 选中 bar 的 datetime <= as_of
# 且为满足条件的最后一根；as_of 之前无已收盘 bar → None。
# Validates: Requirements 1.1, 6.4
@settings(max_examples=100)
@given(
    bar_dts=st.lists(
        st.datetimes(min_value=datetime(2026, 6, 1), max_value=datetime(2026, 6, 30)).map(
            lambda dt: dt.replace(second=0, microsecond=0)
        ),
        min_size=1,
        max_size=20,
        unique=True,
    ),
    as_of=st.datetimes(min_value=datetime(2026, 5, 25), max_value=datetime(2026, 7, 5)),
)
def test_property_p2_intraday_no_lookahead(bar_dts, as_of):
    frame = pl.DataFrame(
        {
            "datetime": sorted(bar_dts),
            "vt_symbol": ["000001.SZSE"] * len(bar_dts),
            "signal": [0.5] * len(bar_dts),
        }
    )
    sel = select_decision_bar(frame, DecisionInstant(as_of, "5m"))
    closed = [dt for dt in bar_dts if dt <= as_of]
    if not closed:
        assert sel is None
    else:
        picked = decision_bar_datetime(sel)
        assert picked == max(closed)
        assert picked <= as_of


# ---------------------------------------------------------------------------
# Property DI-2: As_Of 无前视（结构性）
# ---------------------------------------------------------------------------
# Feature: decision-instant-unification, Property DI-2: As_Of 无前视
# 对任意行情帧与 as_of，select_decision_bar 返回的 bar 的 close_time <= as_of；
# 且为满足该条件的最后一根（最大日期），无任何 close_time > as_of 的 bar 被选中。
# Validates: Requirements 2.1, 2.2
@settings(max_examples=100)
@given(
    days=st.lists(_dates, min_size=1, max_size=12, unique=True),
    as_of=_datetimes,
)
def test_property_di2_as_of_no_lookahead(days, as_of):
    frame = _daily_frame(days)
    sel = select_decision_bar(frame, DecisionInstant(as_of, "1d"))
    closed = [d for d in days if datetime.combine(d, SESSION_CLOSE) <= as_of]
    if not closed:
        assert sel is None
    else:
        picked = decision_bar_datetime(sel).date()
        assert picked == max(closed)
        assert datetime.combine(picked, SESSION_CLOSE) <= as_of


# ---------------------------------------------------------------------------
# Property DI-3: 收盘前自然回退（取代 prev_close）
# ---------------------------------------------------------------------------
# Feature: decision-instant-unification, Property DI-3: 收盘前自然回退
# 对任意两连续交易日 [d0, d1]，as_of >= d1 收盘 → 决策 bar = d1；
# as_of < d1 收盘（但 >= d0 收盘）→ 决策 bar = d0（自然回退到上一已收盘 bar）。
# Validates: Requirements 2.3
@settings(max_examples=100)
@given(d0=_dates, gap=st.integers(min_value=1, max_value=5), minute=st.integers(min_value=0, max_value=59))
def test_property_di3_pre_close_fallback(d0, gap, minute):
    from datetime import timedelta

    d1 = d0 + timedelta(days=gap)
    frame = _daily_frame([d0, d1])

    after = DecisionInstant(datetime.combine(d1, time(15, 5)), "1d")
    assert decision_bar_datetime(select_decision_bar(frame, after)).date() == d1

    before = DecisionInstant(datetime.combine(d1, time(10, minute)), "1d")
    assert decision_bar_datetime(select_decision_bar(frame, before)).date() == d0


# ---------------------------------------------------------------------------
# Property DI-4: signal_id 由 Decision_Bar 决定且日频渲染为日期串
# ---------------------------------------------------------------------------
# Feature: decision-instant-unification, Property DI-4: signal_id 由 Decision_Bar 决定
# 1d 渲染为 date:scheme[@ver]（与历史「按天」signal_id 逐位一致，旧决策文件不孤立）；
# 同一 Decision_Bar 必得同一 signal_id。
# Validates: Requirements 3.1, 3.2
@settings(max_examples=100)
@given(d=_dates, scheme=_idents, ver=st.one_of(st.just(""), _idents))
def test_property_di4_signal_id_1d_legacy_format(d, scheme, ver):
    dt = datetime.combine(d, time(0, 0))
    new_id = make_signal_id(dt, "1d", scheme, ver)
    tag = f"@{ver}" if ver else ""
    assert new_id == f"{d.isoformat()}:{scheme}{tag}"  # 历史「按天」格式逐位一致


def test_di4_signal_id_intraday_format() -> None:
    dt = datetime(2026, 6, 8, 10, 30)
    assert make_signal_id(dt, "30m", "eod_buy_v1", "v3") == "2026-06-08T10:30:eod_buy_v1@v3"
    assert make_signal_id(dt, "30m", "eod_buy_v1") == "2026-06-08T10:30:eod_buy_v1"


# Feature: decision-instant-unification, Property DI-4: 不同 Decision_Bar 必不同 signal_id
# Validates: Requirements 3.3
@settings(max_examples=100)
@given(
    a=_datetimes,
    b=_datetimes,
    scheme=_idents,
    ver=st.one_of(st.just(""), _idents),
)
def test_property_di4_distinct_bars_distinct_ids(a, b, scheme, ver):
    # 截到分钟比较（intraday signal_id 精度为分钟）。
    am = a.replace(second=0, microsecond=0)
    bm = b.replace(second=0, microsecond=0)
    ida = make_signal_id(am, "30m", scheme, ver)
    idb = make_signal_id(bm, "30m", scheme, ver)
    assert (ida == idb) == (am == bm)
