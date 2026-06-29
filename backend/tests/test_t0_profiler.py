"""T0Profiler 测试：在已知"偏离-回归"结构的合成日线上验证画像与建议档位。

Feature: half-position-t0-backtest, Requirement 6 / Property 6（标定窗无前视、严格早于评估窗）

构造一类"买腿有 edge、卖腿≈0"的合成标的：
- 每日开盘 10.00；
- 下探至 9.94（低于开盘 6 分），故买腿在 x≤6 分均可成交；
- 上冲仅到 10.02（高于开盘 2 分），故卖腿仅 x≤2 分成交、再大即"成交率→0"；
- 收盘 10.03（回归至开盘上方），故买腿条件回归边际收益为正、卖腿为负/空。
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.backtest.t0.profiler import (
    BandEdgeRow,
    GapSegmentProfile,
    T0Profile,
    T0Profiler,
    ReversionCalibratedTick,
    profile_by_gap,
)
from aitrade.backtest.t0.tick_policy import DailyBar, DailyHistory, TickContext


def _reversion_daily(n: int = 40) -> pl.DataFrame:
    """构造 n 个交易日、买腿有回归 edge / 卖腿≈0 的合成日线。

    Args:
        n: 交易日数。

    Returns:
        含列 d/open/high/low/close 的 polars DataFrame，时间升序。
    """
    rows = []
    for i in range(n):
        rows.append(
            {
                "d": date(2020, 1, 1) + timedelta(days=i),
                "open": 10.00,
                "high": 10.02,   # 上冲仅 2 分
                "low": 9.94,     # 下探 6 分
                "close": 10.03,  # 收盘回到开盘上方
            }
        )
    return pl.DataFrame(rows)


def test_profile_buy_edge_positive_at_expected_tick() -> None:
    """买腿在下探可成交档位上回归边际收益为正，建议买档落在该结构内。"""
    daily = _reversion_daily()
    prof = T0Profiler().profile("TEST.SZSE", daily, x_grid_fen=range(1, 16))

    # 建议买档应在 1..6 分这个"下探可成交"区间内，且对应净 edge > 0。
    assert 1 <= round(prof.suggested_buy_tick * 100) <= 6
    buy_row = next(r for r in prof.rows if r.x_fen == round(prof.suggested_buy_tick * 100))
    assert buy_row.buy_edge_fen > 0

    # 卖腿在大档位（>2 分）应几乎无成交、edge 不为正；建议卖档不应跑到大档。
    big_sell = next(r for r in prof.rows if r.x_fen == 8)
    assert big_sell.sell_fill < 0.01
    assert prof.suggested_buy_tick != prof.suggested_sell_tick  # 天然非对称


def test_fill_rate_monotonic_non_increasing_in_x() -> None:
    """成交率随偏离档位 x 单调非增（档位越大越难触价）。"""
    daily = _reversion_daily()
    prof = T0Profiler().profile("TEST.SZSE", daily, x_grid_fen=range(1, 16))
    sell_fills = [r.sell_fill for r in prof.rows]
    buy_fills = [r.buy_fill for r in prof.rows]
    assert all(a >= b - 1e-9 for a, b in zip(sell_fills, sell_fills[1:]))
    assert all(a >= b - 1e-9 for a, b in zip(buy_fills, buy_fills[1:]))


def test_buy_edge_known_value() -> None:
    """买腿条件回归边际收益对账：x=5 分时 close-(open-x)=0.08 元，扣成本后*100。"""
    daily = _reversion_daily()
    cr, sd = 0.0003, 0.0005
    prof = T0Profiler().profile(
        "TEST.SZSE", daily, x_grid_fen=range(1, 16), commission_rate=cr, stamp_duty=sd
    )
    row = next(r for r in prof.rows if r.x_fen == 5)
    cost = (cr * 2 + sd) * 10.03
    expected = ((10.03 - (10.00 - 0.05)) - cost) * 100
    assert math.isclose(row.buy_edge_fen, expected, rel_tol=1e-6)
    assert math.isclose(row.buy_fill, 1.0, rel_tol=1e-9)  # 每天都下探到 9.94 ≤ 9.95


def test_suggested_ticks_rescales_proportionally() -> None:
    """suggested_ticks(scale) 相对标定窗均振幅按比例缩放；scale=None 原样返回。"""
    prof = T0Profile(
        symbol="X.SZSE",
        window=(date(2020, 1, 1), date(2020, 3, 1)),
        rows=[],
        suggested_sell_tick=0.02,
        suggested_buy_tick=0.04,
        note="",
        calib_mean_range=0.10,
    )
    assert prof.suggested_ticks(None) == (0.02, 0.04)
    # 近期振幅是标定窗均振幅的 2 倍 → 档位翻倍。
    s, b = prof.suggested_ticks(0.20)
    assert math.isclose(s, 0.04, rel_tol=1e-9)
    assert math.isclose(b, 0.08, rel_tol=1e-9)


def test_reversion_calibrated_tick_no_lookahead_and_deterministic() -> None:
    """ReversionCalibratedTick 只读 hist，不读 day；同一 hist 多次调用确定一致。"""
    prof = T0Profile(
        symbol="X.SZSE",
        window=(date(2020, 1, 1), date(2020, 3, 1)),
        rows=[],
        suggested_sell_tick=0.02,
        suggested_buy_tick=0.04,
        note="",
        calib_mean_range=0.10,
    )
    pol = ReversionCalibratedTick(profile=prof)

    hist = DailyHistory()
    for i in range(20):
        hist.append(DailyBar(d=date(2020, 6, 1 + i), open=10.0, high=10.10, low=9.90, close=10.0))
    # 近 20 日均振幅 0.20，是标定窗均振幅 0.10 的 2 倍 → 翻倍。
    s1, b1 = pol.ticks_for(TickContext(date(2020, 7, 1), 10.0, 10.0, hist, {}))
    s2, b2 = pol.ticks_for(TickContext(date(2099, 12, 31), 10.0, 10.0, hist, {}))  # day 不同，结果应相同
    assert (s1, b1) == (s2, b2)
    assert math.isclose(s1, 0.04, rel_tol=1e-9)
    assert math.isclose(b1, 0.08, rel_tol=1e-9)

    # 历史不足时 mean_range 返回 None → 原样返回建议档位。
    short_hist = DailyHistory()
    short_hist.append(DailyBar(d=date(2020, 6, 1), open=10.0, high=10.1, low=9.9, close=10.0))
    assert pol.ticks_for(TickContext(date(2020, 7, 1), 10.0, 10.0, short_hist, {})) == (0.02, 0.04)


def test_to_dict_round_trips_key_fields() -> None:
    """to_dict() 序列化关键字段，rows 逐档位展开。"""
    prof = T0Profile(
        symbol="X.SZSE",
        window=(date(2020, 1, 1), date(2020, 3, 1)),
        rows=[
            BandEdgeRow(
                x_fen=1,
                sell_fill=0.8,
                sell_edge_fen=0.06,
                buy_fill=0.9,
                buy_edge_fen=0.46,
                day_pnl_fen=0.47,
            )
        ],
        suggested_sell_tick=0.02,
        suggested_buy_tick=0.04,
        note="理想撮合前提",
        calib_mean_range=0.10,
    )
    d = prof.to_dict()
    assert d["symbol"] == "X.SZSE"
    assert d["window"] == ["2020-01-01", "2020-03-01"]
    assert d["suggested_sell_tick"] == 0.02
    assert d["suggested_buy_tick"] == 0.04
    assert d["rows"][0]["x_fen"] == 1
    assert d["rows"][0]["buy_edge_fen"] == 0.46


def test_best_tick_weights_by_fill_count() -> None:
    """建议档位应最大化"成交率×每笔均益"(日均贡献)，而非每笔均益本身。

    2分: 成交率0.75 × 均益0.40 = 贡献0.30；5分: 成交率0.20 × 均益1.00 = 贡献0.20。
    旧逻辑(只看均益)会错选5分；新逻辑(频率加权)应选2分。两档成交率均>0.1(不被尾部门槛剔除)。

    Feature: half-position-t0-backtest
    """
    rows = [
        BandEdgeRow(x_fen=2, sell_fill=0.0, sell_edge_fen=0.0, buy_fill=0.75, buy_edge_fen=0.40, day_pnl_fen=0.0),
        BandEdgeRow(x_fen=5, sell_fill=0.0, sell_edge_fen=0.0, buy_fill=0.20, buy_edge_fen=1.00, day_pnl_fen=0.0),
    ]
    assert T0Profiler._best_tick(rows, "buy") == 0.02


# ---- profile_by_gap：按高/低/平开分场景画像（Property 1/2/3） ----


def _daily_with_gaps(gaps: list[float], base: float = 10.0) -> pl.DataFrame:
    """造日线：gaps[0] 为占位（首日无昨收），其余 i≥1 实现 open=prev_close*(1+gaps[i])。

    每日 close=open+0.03（上方回归），high/low 各外扩 0.05，保证 OHLC 合法。
    """
    rows = []
    prev_close = None
    for i, g in enumerate(gaps):
        o = base if prev_close is None else round(prev_close * (1 + g), 4)
        c = round(o + 0.03, 4)
        rows.append({"d": date(2024, 1, 1) + timedelta(days=i),
                     "open": o, "high": max(o, c) + 0.05, "low": min(o, c) - 0.05, "close": c})
        prev_close = c
    return pl.DataFrame(rows)


def test_profile_by_gap_partitions_high_low_flat() -> None:
    """三场景互斥完备：各 n_days 正确、和=总日数−1（首日剔除）。"""
    daily = _daily_with_gaps([0.0, 0.01, -0.01, 0.0, 0.005, -0.005, 0.001])  # 高2/低2/平2
    segs = profile_by_gap("X.SZSE", daily, thresh=0.003)
    by = {s.regime: s for s in segs}
    assert [s.regime for s in segs] == ["high", "low", "flat"]  # 固定顺序
    assert by["high"].n_days == 2 and by["low"].n_days == 2 and by["flat"].n_days == 2
    assert sum(s.n_days for s in segs) == daily.height - 1      # 首日剔除、并集完备
    assert all(isinstance(s, GapSegmentProfile) for s in segs)


def test_profile_by_gap_drops_first_day_no_lookahead() -> None:
    """首日无昨收 → 不计入任何场景；单行日线时三场景全空。"""
    one = _daily_with_gaps([0.0])
    assert sum(s.n_days for s in profile_by_gap("X.SZSE", one)) == 0


def test_profile_by_gap_segment_equals_filtered_whole_profile() -> None:
    """# Feature: t0-strategy-calibration, Property 3: 分场景=先过滤再调既有 profile。"""
    daily = _daily_with_gaps([0.0, 0.01, 0.012, -0.01, 0.0, 0.008, -0.006])
    segs = {s.regime: s for s in profile_by_gap("X.SZSE", daily, thresh=0.003)}
    # 手动按高开过滤后调既有全窗 profile，应与分场景的高开逐字段一致
    d2 = (daily.sort("d")
          .with_columns((pl.col("open") / pl.col("close").shift(1) - 1.0).alias("g"))
          .drop_nulls("g"))
    high_sub = d2.filter(pl.col("g") > 0.003).drop("g")
    ref = T0Profiler().profile("X.SZSE", high_sub)
    assert segs["high"].profile.to_dict()["rows"] == ref.to_dict()["rows"]
    assert segs["high"].profile.suggested_sell_tick == ref.suggested_sell_tick
    assert segs["high"].profile.suggested_buy_tick == ref.suggested_buy_tick


@settings(max_examples=100, deadline=None)
@given(gaps=st.lists(st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
                     min_size=1, max_size=30),
       thresh=st.floats(min_value=0.001, max_value=0.01))
def test_profile_by_gap_partition_property(gaps, thresh) -> None:
    """# Feature: t0-strategy-calibration, Property 1/2: 三场景互斥完备、首日恒剔除。"""
    daily = _daily_with_gaps([0.0] + gaps)              # 占位首日 + 随机后续
    segs = profile_by_gap("X.SZSE", daily, thresh=thresh)
    assert sum(s.n_days for s in segs) == daily.height - 1   # 并集=非首日全集（互斥由条件天然保证）
