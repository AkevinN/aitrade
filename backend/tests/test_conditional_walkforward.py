"""走查 + 成交假设验证：ConditionalTickPolicy(gap_rules) 走「2023 标定→2025 评估」。

Feature: conditional-tick-policy, Requirement 5.2/6.1/6.2/6.4 · Property 7
（验证只认样本外 + FillPolicy 网格，不认样本内最优）。

不新增 runner 代码，仅复用既有 ``T0BacktestRunner.run``：
- 标定档位仅用 2023 切片（样本内），评估只在 2025 窗（样本外）；
- 报告必须带样本外 eval_window + 跨 FillPolicy 的成交敏感性区间；
- runner 的「标定窗必须严格早于评估窗」守护对任意带 ``profile.window`` 的档位策略生效。
"""

from __future__ import annotations

from datetime import date, datetime, time
from types import SimpleNamespace

import polars as pl
import pytest

from aitrade.backtest.types import BarData, FillPolicy
from aitrade.backtest.t0.runner import T0BacktestRunner
from aitrade.backtest.t0.tick_policy import ConditionalTickPolicy, gap_rules


class _MemLoader:
    def __init__(self, bars): self._bars = bars
    def load_bar_data(self, *a): return list(self._bars)
    def load_contract_settings(self): return {}


def _day_bars(d: date, open_px: float, hi: float, lo: float, close: float):
    """造一天 3 根 1m bar：09:30 开盘 / 10:30 日内高低 / 15:00 收盘。"""
    base = datetime.combine(d, time(0, 0))
    mk = lambda hh, mm, o, h, l, c: BarData(
        symbol="AAA", exchange="SSE", datetime=base.replace(hour=hh, minute=mm),
        interval="1m", open_price=o, high_price=h, low_price=l, close_price=c, volume=10000)
    return [mk(9, 30, open_px, open_px, open_px, open_px),
            mk(10, 30, open_px, hi, lo, open_px),
            mk(15, 0, close, close, close, close)]


def _slice(year: int, n: int = 6, drift: float = 0.02):
    """造某年 n 个交易日：每日相对昨收跳空 ±0.6%，日内 close=open+drift。

    Returns:
        (1m bars 列表, 日线 DataFrame[d/open/high/low/close])。
    """
    bars, drows = [], []
    prev_close = 10.0
    for i in range(n):
        d = date(year, 3, 1) + (date(year, 3, 1 + i) - date(year, 3, 1))
        gap = 0.006 if i % 2 == 0 else -0.006
        o = round(prev_close * (1 + gap), 2)
        c = round(o + drift, 2)
        hi, lo = max(o, c) + 0.05, min(o, c) - 0.05
        bars.extend(_day_bars(d, o, hi, lo, c))
        drows.append({"d": d, "open": o, "high": hi, "low": lo, "close": c})
        prev_close = c
    return bars, pl.DataFrame(drows)


def _calibrate_gap_ticks(daily_2023: pl.DataFrame):
    """仅用 2023 数据标定跳空日的档位倾斜（样本内，不看 2025）。

    跳空日（|open/prev_close−1|>0.3%）若日内续涨（close>open），高开日倾向「卖远买近」。

    Returns:
        (up, down) 两个 (sell_tick, buy_tick) 元组，供 gap_rules 使用。
    """
    o = daily_2023["open"].to_numpy()
    c = daily_2023["close"].to_numpy()
    up_drift = float((c - o).mean())
    if up_drift > 0:                       # 跳空后日内续涨 → 卖远买近
        return (0.07, 0.01), (0.01, 0.07)
    return (0.01, 0.07), (0.07, 0.01)      # 否则反向


def test_conditional_walkforward_oos_report_has_fill_range() -> None:
    """2023 标定档位 → 2025 评估，报告带样本外窗口 + 跨成交假设区间（Property 7）。"""
    _, daily_2023 = _slice(2023)
    up, down = _calibrate_gap_ticks(daily_2023)            # 仅用 2023

    eval_bars, daily_2025 = _slice(2025)                   # 样本外评估数据
    runner = T0BacktestRunner(data_loader=_MemLoader(eval_bars))
    pol = ConditionalTickPolicy(rules=gap_rules(thresh=0.003, up=up, down=down), default=(0.03, 0.03))

    start, end = date(2025, 3, 1), date(2025, 3, 6)
    report = runner.run("AAA.SSE", start, end, daily_2025,
                        tick_policies=[("gap_oos", pol)],
                        fill_grid=[FillPolicy(0.0, 1.0), FillPolicy(0.01, 1.0), FillPolicy(0.0, 0.5)])

    # 样本外口径：评估窗严格晚于 2023 标定窗
    assert report.eval_window == (start, end)
    assert start.year == 2025
    # 成交敏感性区间：3 种成交假设各一行、互不相同
    fs = report.fill_sensitivity()
    assert len(fs) == 3
    assert len({tuple(sorted(x["fill"].items())) for x in fs}) == 3
    d = report.to_dict()
    assert d["eval_window"] == ["2025-03-01", "2025-03-06"]
    assert "fill_sensitivity" in d and len(d["results"]) == 3


def test_runner_rejects_calibration_overlapping_eval() -> None:
    """OOS 守护对任意带 profile.window 的档位策略生效：标定窗触及评估窗即报错（Req 6.4）。"""
    eval_bars, daily_2025 = _slice(2025)
    runner = T0BacktestRunner(data_loader=_MemLoader(eval_bars))
    pol = ConditionalTickPolicy(rules=gap_rules(), default=(0.03, 0.03))
    # 模拟「标定窗跨入评估窗」的违规策略：附一个 window 触及 2025-03 的 profile
    pol.profile = SimpleNamespace(window=(date(2024, 1, 1), date(2025, 3, 3)))

    with pytest.raises(ValueError, match="标定窗"):
        runner.run("AAA.SSE", date(2025, 3, 1), date(2025, 3, 6), daily_2025,
                   tick_policies=[("leaky", pol)], fill_grid=[FillPolicy(0.0, 1.0)])
