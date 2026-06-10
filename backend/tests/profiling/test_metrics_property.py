"""
标的画像（Symbol Profiling）时间裁剪与比例指标的属性测试（Hypothesis）。

本文件覆盖 design.md「Correctness Properties」中的 Property 1–4，针对
`aitrade.profiling.loader` 的时间裁剪纯函数与 `aitrade.profiling.metrics` 的
比例 / 对齐覆盖率指标。每个属性用单个属性测试实现，`@settings(max_examples=100)`，
并在测试函数上方以 `# Feature: symbol-profiling, Property {n}: ...` 注释标注。

生成器要点：
- 随机时间序列：用「基准时间 + 分钟偏移」构造 datetime，偏移可正可负，
  天然覆盖跨 as_of 边界、空 frame、单行等情形，并使观测集合间产生真实重叠。
- 随机零成交 / 成交量分布：成交量取值含 None、0 与正浮点，覆盖零成交占比的各种比例。
- 随机观测 frame 集合：观测标的数量与各自时间戳子集均随机，覆盖对齐覆盖率的交集行为。

所有指标均为只读纯函数，构造 polars.DataFrame 时显式指定 schema，
保证空 frame 也带正确的 Datetime 列类型。
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.profiling.loader import clip_to_as_of, effective_right_bound
from aitrade.profiling.metrics import (
    alignment_coverage,
    amplitude_quantiles,
    atr_ratio,
    gap_ratio,
    realized_volatility,
    zero_volume_ratio,
)

# 构造时间序列的基准时刻；所有 datetime 由「基准 + 分钟偏移」生成
_BASE = datetime(2024, 1, 1, 9, 30)

# polars frame 的显式 schema：保证空 frame 的 datetime 列仍为 Datetime 类型，
# 使下游 filter / 比较不会因 Null 类型列而行为异常
_DT_SCHEMA = {"datetime": pl.Datetime("us")}
_DT_VOL_SCHEMA = {"datetime": pl.Datetime("us"), "volume": pl.Float64}
_OHLC_SCHEMA = {
    "datetime": pl.Datetime("us"),
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
}


def _dt(offset_minutes: int) -> datetime:
    """由分钟偏移构造 datetime（偏移可正可负）。"""
    return _BASE + timedelta(minutes=offset_minutes)


def _frame_from_offsets(offsets: list[int]) -> pl.DataFrame:
    """由分钟偏移列表构造仅含 datetime 列的 frame（空列表得到带正确类型的空 frame）。"""
    return pl.DataFrame({"datetime": [_dt(m) for m in offsets]}, schema=_DT_SCHEMA)


def _frame_with_volume(rows: list[tuple[int, float | None]]) -> pl.DataFrame:
    """由 (分钟偏移, 成交量) 列表构造含 datetime 与 volume 列的 frame。"""
    dts = [_dt(m) for m, _ in rows]
    vols = [v for _, v in rows]
    return pl.DataFrame({"datetime": dts, "volume": vols}, schema=_DT_VOL_SCHEMA)


def _ohlc_frame(rows: list[tuple[int, float, float]]) -> pl.DataFrame:
    """由 (分钟偏移, close, spread) 构造高低收 frame。"""
    dts = [_dt(m) for m, _, _ in rows]
    closes = [close for _, close, _ in rows]
    highs = [close + spread for _, close, spread in rows]
    lows = [max(0.01, close - spread) for _, close, spread in rows]
    return pl.DataFrame(
        {"datetime": dts, "high": highs, "low": lows, "close": closes},
        schema=_OHLC_SCHEMA,
    )


# 偏移与 as_of 的取值范围：覆盖正负偏移，使 as_of 可能早于 / 晚于 / 居中于序列
_offset = st.integers(min_value=-120, max_value=120)
_offsets = st.lists(_offset, min_size=0, max_size=30)

# 成交量：含 None（缺失）、0（零成交）与正浮点（正常成交），覆盖零成交占比各档
_volume = st.one_of(
    st.none(),
    st.just(0.0),
    st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False),
)
_vol_rows = st.lists(st.tuples(_offset, _volume), min_size=0, max_size=30)
_ohlc_rows = st.lists(
    st.tuples(
        _offset,
        st.floats(min_value=0.1, max_value=1_000.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    ),
    min_size=0,
    max_size=50,
)

# 周期与时段画像：覆盖分钟级与日线，session_profile 取若干合法标识
_interval = st.sampled_from(["1m", "5m", "30m", "60m", "d"])
_session = st.sampled_from(["cn_equity", "cn_future", ""])


# Feature: symbol-profiling, Property 1: 时间裁剪绝不泄露未来数据 —
# clip_to_as_of(df, as_of) 的每行 datetime <= as_of，且恰等于原 frame 中所有
# datetime <= as_of 的行（不增不减、时间升序）。
# Validates: Requirements 2.2, 2.3
@settings(max_examples=100)
@given(offsets=_offsets, as_of_offset=_offset)
def test_property1_clip_never_leaks_future(offsets: list[int], as_of_offset: int) -> None:
    as_of = _dt(as_of_offset)
    df = _frame_from_offsets(offsets)

    result = clip_to_as_of(df, as_of)
    result_dts = result["datetime"].to_list()

    # 1) 结果每一行都满足 datetime <= as_of（绝不泄露未来数据）
    assert all(dt <= as_of for dt in result_dts)

    # 2) 结果恰好等于原 frame 中所有 datetime <= as_of 的行，且按时间升序
    expected = sorted(dt for dt in (_dt(m) for m in offsets) if dt <= as_of)
    assert result_dts == expected


# Feature: symbol-profiling, Property 2: 有效右边界不超过 as_of 且等于裁剪后最大时间 —
# 裁剪后非空时 effective_right_bound <= as_of 且等于裁剪后最大 datetime；为空返回 None。
# Validates: Requirements 2.4, 2.5
@settings(max_examples=100)
@given(offsets=_offsets, as_of_offset=_offset)
def test_property2_effective_right_bound(offsets: list[int], as_of_offset: int) -> None:
    as_of = _dt(as_of_offset)
    df = _frame_from_offsets(offsets)

    bound = effective_right_bound(df, as_of)
    clipped_dts = [dt for dt in (_dt(m) for m in offsets) if dt <= as_of]

    if not clipped_dts:
        # 裁剪后为空：返回 None
        assert bound is None
    else:
        # 裁剪后非空：<= as_of 且等于裁剪后最大时间
        assert bound is not None
        assert bound <= as_of
        assert bound == max(clipped_dts)


# Feature: symbol-profiling, Property 3: 比例类指标恒落在 [0, 1] —
# gap_ratio / zero_volume_ratio / alignment_coverage 的 value 恒在闭区间 [0, 1]。
# Validates: Requirements 3.2, 3.3, 4.1
@settings(max_examples=100)
@given(
    rows=_vol_rows,
    interval=_interval,
    session=_session,
    others_offsets=st.lists(_offsets, min_size=0, max_size=4),
)
def test_property3_ratio_metrics_within_unit_interval(
    rows: list[tuple[int, float | None]],
    interval: str,
    session: str,
    others_offsets: list[list[int]],
) -> None:
    df = _frame_with_volume(rows)

    gap = gap_ratio(df, interval, session).value
    assert isinstance(gap, float)
    assert 0.0 <= gap <= 1.0

    zero = zero_volume_ratio(df).value
    assert isinstance(zero, float)
    assert 0.0 <= zero <= 1.0

    others = [_frame_from_offsets(o) for o in others_offsets]
    coverage = alignment_coverage(df, others).value
    assert isinstance(coverage, float)
    assert 0.0 <= coverage <= 1.0


# Feature: symbol-profiling, Property 4: 对齐覆盖率的单调性与边界 —
# 所有观测覆盖目标全部时间戳时为 1.0；向观测集合追加任一 frame 后覆盖率单调非增；
# 目标为空返回 0.0。
# Validates: Requirements 3.3, 13.1
@settings(max_examples=100)
@given(
    target_offsets=_offsets,
    others_offsets=st.lists(_offsets, min_size=0, max_size=4),
    extra_offsets=_offsets,
)
def test_property4_alignment_coverage_monotonic_and_bounds(
    target_offsets: list[int],
    others_offsets: list[list[int]],
    extra_offsets: list[int],
) -> None:
    target = _frame_from_offsets(target_offsets)
    others = [_frame_from_offsets(o) for o in others_offsets]
    extra = _frame_from_offsets(extra_offsets)

    cov_base = alignment_coverage(target, others).value
    cov_ext = alignment_coverage(target, others + [extra]).value

    # 边界：目标为空时返回 0.0
    if target.height == 0:
        assert cov_base == 0.0

    # 单调非增：向观测集合追加任一 frame，覆盖率不增（交集只会收缩）
    assert cov_ext <= cov_base + 1e-9

    # 全覆盖：每个观测都包含目标的全部时间戳时，覆盖率为 1.0（目标非空）
    if target.height > 0:
        full_observers = [_frame_from_offsets(target_offsets) for _ in range(3)]
        cov_full = alignment_coverage(target, full_observers).value
        assert math.isclose(cov_full, 1.0, abs_tol=1e-9)


# Feature: symbol-profiling, Property 5: 波动 / ATR 类指标非负 —
# realized_volatility / atr_ratio / amplitude_quantiles 的数值恒非负。
# Validates: Requirements 5.1
@settings(max_examples=100)
@given(rows=_ohlc_rows, window=st.integers(min_value=-10, max_value=60))
def test_property5_volatility_metrics_are_non_negative(
    rows: list[tuple[int, float, float]],
    window: int,
) -> None:
    df = _ohlc_frame(rows)

    rv = realized_volatility(df).value
    assert isinstance(rv, float)
    assert rv >= 0.0

    atr = atr_ratio(df, window).value
    assert isinstance(atr, float)
    assert atr >= 0.0

    quantiles = amplitude_quantiles(df, [0.5, 0.9]).value
    assert isinstance(quantiles, dict)
    assert all(value >= 0.0 for value in quantiles.values())
