"""批次合并提速（batch-merge-speedup）的等价性测试。

核心立场：纯性能改造，行为必须与改造前逐项一致。本文件内联保留「参考实现」
（逐组 unique 循环、逐根 _batch_minutes_expected 循环）作为黄金对照，断言向量化
helper 的结果与之完全相等；并用端到端字面用例钉住 _build_merge_plan 的决策与产物。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.alpha.lab import AlphaLab, BarData, _interval_minutes

# 仅用其纯方法 _batch_minutes_expected（不依赖 self 状态）。
_LAB = AlphaLab.__new__(AlphaLab)

_VALUE_COLS = ["open", "high", "low", "close", "volume", "turnover", "open_interest"]


# ---------- 参考实现（改造前逻辑，黄金对照） ----------

def _ref_conflict_count(combined: pl.DataFrame, value_columns: list[str]) -> int:
    """改造前的逐组冲突计数。"""
    n = 0
    for _, group in combined.group_by("datetime", maintain_order=True):
        if len(group) <= 1:
            continue
        if group.select(value_columns).unique().height > 1:
            n += 1
    return n


def _ref_first_gap(sorted_dts: list[datetime], interval: str):
    """改造前的逐根连续性：返回首个断档 (prev, curr)，无则 None。"""
    for prev_dt, curr_dt in zip(sorted_dts, sorted_dts[1:]):
        if not _LAB._batch_minutes_expected(prev_dt, curr_dt, interval):
            return (prev_dt, curr_dt)
    return None


# ---------- 数据生成 ----------

def _trading_grid(days: int, step: int) -> list[datetime]:
    """生成连续 A 股交易分钟网格（上午 09:30–11:30、下午 13:00–15:00）。"""
    base = datetime(2024, 1, 2)
    out: list[datetime] = []
    for dd in range(days):
        day = base + timedelta(days=dd)
        for start_h, start_m, end_h, end_m in [(9, 30, 11, 30), (13, 0, 15, 0)]:
            cur = day.replace(hour=start_h, minute=start_m)
            end = day.replace(hour=end_h, minute=end_m)
            while cur < end:
                out.append(cur)
                cur += timedelta(minutes=step)
    return out


@st.composite
def _minute_series(draw):
    """从交易网格里随机保留部分点（制造断档/跨日/跨小节各种组合）。"""
    interval = draw(st.sampled_from(["1m", "5m", "15m"]))
    step = _interval_minutes(interval)
    grid = _trading_grid(draw(st.integers(1, 3)), step)
    flags = draw(st.lists(st.booleans(), min_size=len(grid), max_size=len(grid)))
    series = [g for g, keep in zip(grid, flags) if keep]
    return series, interval


# ---------- Property 1: 冲突计数等价 ----------

@settings(max_examples=100, deadline=None)
@given(
    n_part=st.integers(min_value=1, max_value=4),
    n_dt=st.integers(min_value=1, max_value=40),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property1_conflict_count_equivalent(n_part: int, n_dt: int, seed: int) -> None:
    # Feature: batch-merge-speedup, Property 1: 冲突计数等价
    from aitrade.alpha.lab import _count_value_conflicts

    rng = (seed * 2654435761) & 0xFFFFFFFF
    dts = [datetime(2024, 1, 2) + timedelta(minutes=i) for i in range(n_dt)]
    parts = []
    for p in range(n_part):
        # 随机让部分时间点的 close 在不同参与方间不一致 → 制造冲突
        closes = [10.0 + (1.0 if ((rng >> (i % 31)) & 1 and p % 2) else 0.0) for i in range(n_dt)]
        parts.append(pl.DataFrame({
            "datetime": dts, "open": [10.0] * n_dt, "high": [10.5] * n_dt, "low": [9.5] * n_dt,
            "close": closes, "volume": [100.0] * n_dt, "turnover": [1000.0] * n_dt,
            "open_interest": [0.0] * n_dt, "_batch_order": [p] * n_dt,
        }))
    combined = pl.concat(parts, how="vertical_relaxed")
    assert _count_value_conflicts(combined, _VALUE_COLS) == _ref_conflict_count(combined, _VALUE_COLS)


def test_property1_conflict_empty_value_columns() -> None:
    # Feature: batch-merge-speedup, Property 1: 空值列返回 0
    from aitrade.alpha.lab import _count_value_conflicts

    combined = pl.DataFrame({"datetime": [datetime(2024, 1, 2)], "_batch_order": [0]})
    assert _count_value_conflicts(combined, []) == 0


# ---------- Property 2: 连续性判定等价 ----------

@settings(max_examples=150, deadline=None)
@given(data=_minute_series())
def test_property2_session_gap_equivalent(data) -> None:
    # Feature: batch-merge-speedup, Property 2: 连续性判定等价（随机断档/跨日/跨小节）
    from aitrade.alpha.lab import _session_gap

    series, interval = data
    assert _session_gap(series, interval) == _ref_first_gap(series, interval)


def test_property2_session_gap_edge_cases() -> None:
    # Feature: batch-merge-speedup, Property 2: 边界用例（午休/隔夜/错频/重复/收盘点）
    from aitrade.alpha.lab import _session_gap

    cases = [
        # 连续上午 → 无断档
        [datetime(2024, 1, 2, 9, 30), datetime(2024, 1, 2, 9, 31), datetime(2024, 1, 2, 9, 32)],
        # 同小节缺根 → 断档
        [datetime(2024, 1, 2, 9, 30), datetime(2024, 1, 2, 9, 32)],
        # 午休跨小节 → 合法
        [datetime(2024, 1, 2, 11, 29), datetime(2024, 1, 2, 13, 0)],
        # 隔夜跨日 → 合法
        [datetime(2024, 1, 2, 14, 59), datetime(2024, 1, 3, 9, 30)],
        # 错频（2 分钟当 1m）→ 断档
        [datetime(2024, 1, 2, 9, 30) + timedelta(minutes=2 * i) for i in range(5)],
        # 收盘点 15:00（盘外）→ 合法（与参考一致的宽松边界）
        [datetime(2024, 1, 2, 14, 58), datetime(2024, 1, 2, 14, 59), datetime(2024, 1, 2, 15, 0)],
        # 重复时间点 → 参考按 curr<=prev 判断档
        [datetime(2024, 1, 2, 9, 30), datetime(2024, 1, 2, 9, 30)],
        # 单点 / 空
        [datetime(2024, 1, 2, 9, 30)],
        [],
    ]
    for series in cases:
        assert _session_gap(series, "1m") == _ref_first_gap(series, "1m"), series


# ---------- Property 3: 会话分钟不溢出 ----------

def test_property3_no_int8_overflow_in_session() -> None:
    # Feature: batch-merge-speedup, Property 3: time-of-day 分钟数不溢出（09:30→570 而非 Int8 回绕）
    from aitrade.alpha.lab import _session_gap

    # 若 hour*60 用 Int8 溢出（570→58），09:30 会被误判为盘外 → 缺根无法识别。
    # 09:30→09:32 在上午小节内缺一根，必须判为断档。
    series = [datetime(2024, 1, 2, 9, 30), datetime(2024, 1, 2, 9, 32)]
    assert _session_gap(series, "1m") == (series[0], series[1])
    # 下午 14:58→15:00 缺 14:59，但 14:58/15:00 与参考一致地处理
    pm = [datetime(2024, 1, 2, 14, 57), datetime(2024, 1, 2, 14, 58), datetime(2024, 1, 2, 14, 59)]
    assert _session_gap(pm, "1m") == _ref_first_gap(pm, "1m")


# ---------- Property 4 / 5: 端到端字面特性（钉住 _build_merge_plan 行为不变） ----------

def _bar(day_offset: int, *, close: float = 10.1, interval: str = "d") -> BarData:
    return BarData(
        symbol="000001", exchange="SZSE",
        datetime=datetime(2024, 1, 2) + timedelta(days=day_offset),
        interval=interval, open_price=10.0, high_price=10.2, low_price=9.8,
        close_price=close, volume=100.0, turnover=1000.0,
    )


def test_property4_single_batch_promote(tmp_path) -> None:
    # Feature: batch-merge-speedup, Property 4: 单批次+无官方直连晋级，冲突恒 0
    lab = AlphaLab(tmp_path)
    batch = lab.save_bars_as_import_batch([_bar(i) for i in range(3)])
    plan = lab.preview_merge_import_batches(kind="raw_bar", keys=[batch["key"]])
    assert plan["can_merge"] is True
    assert plan["conflict_count"] == 0
    assert plan["estimated_rows"] == 3
    assert plan["has_official"] is False


def test_property5_consistent_overlap_merges(tmp_path) -> None:
    # Feature: batch-merge-speedup, Property 5: 一致重叠可合并
    lab = AlphaLab(tmp_path)
    a = lab.save_bars_as_import_batch([_bar(0), _bar(1), _bar(2)])
    b = lab.save_bars_as_import_batch([_bar(1), _bar(2), _bar(3)])  # 与 a 在 day1/2 完全一致
    plan = lab.preview_merge_import_batches(kind="raw_bar", keys=[a["key"], b["key"]])
    assert plan["can_merge"] is True
    assert plan["conflict_count"] == 0
    assert plan["estimated_rows"] == 4  # day0..day3 去重


def test_property5_conflicting_overlap_rejected(tmp_path) -> None:
    # Feature: batch-merge-speedup, Property 5: 重叠区不一致被拒
    lab = AlphaLab(tmp_path)
    a = lab.save_bars_as_import_batch([_bar(0), _bar(1, close=10.1)])
    b = lab.save_bars_as_import_batch([_bar(1, close=99.9), _bar(2)])  # day1 冲突
    plan = lab.preview_merge_import_batches(kind="raw_bar", keys=[a["key"], b["key"]])
    assert plan["can_merge"] is False
    assert plan["conflict_count"] >= 1
    assert "不一致" in plan["reason"]
