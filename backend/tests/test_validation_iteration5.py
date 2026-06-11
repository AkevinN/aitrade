"""
迭代 5 验收测试：回测验证工具（成本敏感性 + 样本外切分）。

覆盖：
1. cost_sensitivity_table：集成 run_scheme_backtest，佣金×2 → 手续费更高、净盈亏更低。
2. time_series_holdout：按时间顺序切分、保持顺序、非法比例抛错。
3. walk_forward_windows：窗口数量、test 紧接 train（无样本外泄漏）、非法参数抛错。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

import aitrade.cnn.strategy  # noqa: F401  触发 cnn_signal 注册
from aitrade.backtest.scheme import (
    CostConfig,
    PredictorConfig,
    Scheme,
    StrategyConfig,
    run_scheme_backtest,
)
from aitrade.backtest.types import BarData
from aitrade.backtest.validation import (
    cost_sensitivity_table,
    time_series_holdout,
    walk_forward_windows,
)

SYMBOL = "TEST.SZSE"
START = datetime(2026, 1, 5)


class FakeLoader:
    def __init__(self, bars: list[BarData]) -> None:
        self._bars = bars

    def load_bar_data(self, vt_symbol, interval, start, end) -> list[BarData]:
        return list(self._bars)

    def load_contract_settings(self) -> dict:
        return {SYMBOL: {"long_rate": 3e-4, "short_rate": 3e-4, "stamp_duty": 0.0,
                         "slippage": 0.0, "size": 1, "pricetick": 0.01}}


def _bars(n: int):
    days = [START + timedelta(days=i) for i in range(n)]
    closes = [100.0 + i for i in range(n)]
    out = []
    for i, c in enumerate(closes):
        prev = closes[i - 1] if i > 0 else c
        out.append(BarData(symbol="TEST", exchange="SZSE", datetime=days[i], interval="d",
                           open_price=prev, high_price=max(prev, c) + 1, low_price=min(prev, c) - 1,
                           close_price=c, volume=1_000_000))
    return out, days


# ---------------------------------------------------------------------------
# 1. 成本敏感性
# ---------------------------------------------------------------------------
def test_cost_sensitivity_commission_doubles_increases_cost() -> None:
    bars, days = _bars(8)
    loader = FakeLoader(bars)
    signal = pl.DataFrame({
        "datetime": days, "vt_symbol": [SYMBOL] * 8,
        "signal": [0.9, 0.5, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5],
    })
    start, end = days[0], days[-1] + timedelta(days=1)
    base_cost = CostConfig(commission_rate=0.0003, stamp_duty=0.001, slippage=0.0)

    def make_scheme(cost: CostConfig) -> Scheme:
        return Scheme(
            name="s", vt_symbols=[SYMBOL], interval="d",
            predictor=PredictorConfig(type="cnn"),
            strategy=StrategyConfig(name="cnn_signal", params={
                "buy_threshold": 0.6, "exit_mode": "fixed_hold", "hold_days": 1}),
            cost=cost,
        )

    def run_with_cost(cost: CostConfig) -> dict:
        return run_scheme_backtest(make_scheme(cost), loader, signal, start, end)["statistics"]

    table = cost_sensitivity_table(run_with_cost, base_cost)
    assert [r["scenario"] for r in table] == ["基准", "佣金×2", "滑点+5bp"]

    base_row = table[0]
    comm2_row = table[1]
    assert comm2_row["total_commission"] > base_row["total_commission"]
    assert comm2_row["total_net_pnl"] < base_row["total_net_pnl"]


# ---------------------------------------------------------------------------
# 2. 样本外切分
# ---------------------------------------------------------------------------
def test_time_series_holdout_preserves_order() -> None:
    items = list(range(10))
    train, test = time_series_holdout(items, 0.7)
    assert train == [0, 1, 2, 3, 4, 5, 6]
    assert test == [7, 8, 9]
    # 拼接还原、无重叠、不打乱
    assert train + test == items


def test_time_series_holdout_invalid_ratio() -> None:
    with pytest.raises(ValueError):
        time_series_holdout([1, 2, 3], 1.5)


# ---------------------------------------------------------------------------
# 3. walk-forward 窗口
# ---------------------------------------------------------------------------
def test_walk_forward_windows_no_leakage() -> None:
    windows = walk_forward_windows(
        start=date(2024, 1, 1), end=date(2024, 12, 31),
        train_days=90, test_days=30,
    )
    assert len(windows) >= 3
    for w in windows:
        train_start, train_end = w["train"]
        test_start, test_end = w["test"]
        # test 紧接 train，样本外不含训练期
        assert test_start == train_end
        assert test_start < test_end
        assert train_start < train_end


def test_walk_forward_windows_invalid_params() -> None:
    with pytest.raises(ValueError):
        walk_forward_windows(date(2024, 1, 1), date(2024, 6, 1), train_days=0, test_days=30)
