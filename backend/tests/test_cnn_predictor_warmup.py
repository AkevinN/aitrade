"""
盘中监控决策（Intraday Monitoring Decision）— predictor warm-up 换算属性测试（Wave 2）。

历史 bug：`extended_start = start - timedelta(days=lookback*2.5)` 把 lookback（bar 数）
当日历日数，对分钟模型造成数百日分钟数据的过度拉取。修复后按 input_interval 折算。
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.cnn.predictor import _BARS_PER_DAY, warmup_days

_lookbacks = st.integers(min_value=1, max_value=512)


# Feature: intraday-monitoring-decision, Property 7: warm-up 换算
# "d" 下回退天数与旧公式 int(lookback*2.5) 等价（±1 天，ceil vs int）。
# Validates: Requirements 4.2
@settings(max_examples=100)
@given(lookback=_lookbacks)
def test_property_p7_daily_equivalent_to_legacy(lookback):
    legacy = int(lookback * 2.5)
    assert abs(warmup_days(lookback, "d") - max(5, legacy)) <= 1


# Feature: intraday-monitoring-decision, Property 7: warm-up 换算
# 分钟频下 warm_days * bars_per_day >= lookback（覆盖回看窗口）且 <= 旧公式（不再过度拉取）。
# Validates: Requirements 4.2
@settings(max_examples=100)
@given(lookback=_lookbacks, interval=st.sampled_from(["1m", "5m", "10m", "15m", "30m", "60m"]))
def test_property_p7_intraday_covers_lookback_without_overfetch(lookback, interval):
    days = warmup_days(lookback, interval)
    bars_per_day = _BARS_PER_DAY[interval]
    # 覆盖性：按交易日折算（2.5 倍日历裕量中至少 1 倍是交易日）
    assert days >= math.ceil(lookback / bars_per_day)
    assert days * bars_per_day >= lookback
    # 不过度：不超过旧公式（bar 数当天数）
    assert days <= max(5, math.ceil(lookback * 2.5))


def test_p7_examples() -> None:
    # 日频 lookback=30 → 75 天（与历史一致）
    assert warmup_days(30, "d") == 75
    # 5m 模型 lookback=96（2 个交易日的 bar）→ 5 天下限，而非旧公式的 240 天
    assert warmup_days(96, "5m") == 5
    # 未知周期保守按日频处理
    assert warmup_days(30, "w") == 75


def test_p7_minimum_floor() -> None:
    assert warmup_days(1, "1m") == 5
