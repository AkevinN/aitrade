"""cnn/intervals.py bars-per-day 单一事实源测试（按天换算 P2）。"""

from __future__ import annotations

import math

import pytest

from aitrade.cnn.intervals import BARS_PER_TRADING_DAY, bars_per_day, bars_to_days


def test_bars_per_day_known_intervals():
    assert bars_per_day("d") == 1
    assert bars_per_day("60m") == 4
    assert bars_per_day("30m") == 8
    assert bars_per_day("1m") == 240


def test_bars_per_day_unknown_falls_back_to_one():
    assert bars_per_day("3m") == 1
    assert bars_per_day("weekly") == 1


def test_bars_to_days_daily_is_identity():
    # 日线（每日 1 根）下 bar 数恒等于交易日数。
    for n in (1, 5, 30, 120):
        assert bars_to_days(n, "d") == n


def test_bars_to_days_minute_ceil():
    # 30m 每日 8 根：10 根 → ceil(10/8)=2 个交易日。
    assert bars_to_days(10, "30m") == 2
    assert bars_to_days(8, "30m") == 1
    assert bars_to_days(16, "30m") == 2
    assert bars_to_days(17, "30m") == 3


def test_bars_to_days_zero_is_zero():
    # 0 表示"无持有期/不限"（与 OCO max_hold=0 语义一致）。
    assert bars_to_days(0, "30m") == 0
    assert bars_to_days(-5, "30m") == 0


@pytest.mark.parametrize("interval", list(BARS_PER_TRADING_DAY))
def test_bars_to_days_matches_ceil_formula(interval: str):
    bpd = BARS_PER_TRADING_DAY[interval]
    for n in (1, 7, 13, 100):
        assert bars_to_days(n, interval) == max(1, math.ceil(n / bpd))
