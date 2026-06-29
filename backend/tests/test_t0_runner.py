"""T0BacktestRunner 冒烟测试：合成多日 1m 行情，验证报告结构与成交敏感性区间。

Feature: half-position-t0-backtest
"""

from __future__ import annotations

from datetime import datetime, date

import polars as pl

from aitrade.backtest.types import BarData, FillPolicy
from aitrade.backtest.t0.runner import T0BacktestRunner
from aitrade.backtest.t0.tick_policy import FixedTick


class _MemLoader:
    def __init__(self, bars): self._bars = bars
    def load_bar_data(self, *a): return list(self._bars)
    def load_contract_settings(self): return {}


def _bars_and_daily(n_days=6):
    bars, drows = [], []
    for d in range(1, n_days + 1):
        base = datetime(2024, 1, d)
        o = 10.0 + 0.1 * (d % 3)            # 轻微波动
        hi, lo, c = o + 0.05, o - 0.05, o + 0.02
        for hh, mm, bo, bh, bl, bc in [(9, 30, o, o, o, o), (10, 30, o, hi, lo, o), (15, 0, c, c, c, c)]:
            bars.append(BarData(symbol="AAA", exchange="SSE", datetime=base.replace(hour=hh, minute=mm),
                                interval="1m", open_price=bo, high_price=bh, low_price=bl,
                                close_price=bc, volume=10000))
        drows.append({"d": date(2024, 1, d), "open": o, "high": hi, "low": lo, "close": c})
    return bars, pl.DataFrame(drows)


def _runner_and_args():
    bars, daily = _bars_and_daily()
    runner = T0BacktestRunner(data_loader=_MemLoader(bars))
    return runner, daily


def test_runner_report_has_fill_sensitivity_range() -> None:
    runner, daily = _runner_and_args()
    report = runner.run("AAA.SSE", date(2024, 1, 1), date(2024, 1, 6), daily,
                        tick_policies=[("fixed_2fen", FixedTick(0.02, 0.02))],
                        fill_grid=[FillPolicy(0.0, 1.0), FillPolicy(0.01, 1.0), FillPolicy(0.0, 0.5)])
    assert len(report.results) == 3                       # 1 档位 × 3 成交假设
    fs = report.fill_sensitivity()
    assert len(fs) == 3
    assert {tuple(sorted(x["fill"].items())) for x in fs} .__len__() == 3   # 三种成交假设各异
    d = report.to_dict()
    assert d["symbol"] == "AAA.SSE" and "fill_sensitivity" in d


def test_runner_single_result_structure() -> None:
    runner, daily = _runner_and_args()
    report = runner.run("AAA.SSE", date(2024, 1, 1), date(2024, 1, 6), daily,
                        tick_policies=[("fixed", FixedTick(0.02, 0.02))],
                        fill_grid=[FillPolicy(0.0, 1.0)])
    r = report.results[0]
    assert r.yearly and "excess_vs_bh" in r.yearly[0]
    assert set(r.hit_dist) == {"both", "onlyS", "onlyB", "none"}
    assert abs(sum(r.hit_dist.values()) - 1.0) < 1e-6
    assert isinstance(r.total_return, float)


# ---- T3：signal_provider 端到端贯通（条件信号规则真正影响成交） ----

def test_runner_threads_signal_provider_end_to_end() -> None:
    """注入 DictSignalProvider + signal 左值条件策略：高信号命中→卖档巨大不成交→换手更低。

    Feature: t0-conditional-tick-frontend, Property 5（多策略/信号接线）
    """
    from aitrade.backtest.t0.policy_spec import ConditionalCfg, RuleCfg, compile_tick_policy
    from aitrade.backtest.t0.signals import DictSignalProvider

    bars, daily = _bars_and_daily(8)
    cfg = ConditionalCfg(
        label="sig", default_sell_tick=0.01, default_buy_tick=0.01,
        rules=[RuleCfg(lhs="signal", op="gt", threshold=0.5, signal_name="s",
                       sell_tick=0.50, buy_tick=0.01)])   # 命中→卖档0.5元(永不成交)
    _, pol, names = compile_tick_policy(cfg)
    assert names == ("s",)

    # 高信号(0.9)：每个评估日都命中 → 抑制卖腿
    table = {("AAA.SSE", d, "s"): 0.9 for d in daily["d"].to_list()}
    sp = DictSignalProvider(table)
    rep_hi = T0BacktestRunner(data_loader=_MemLoader(bars)).run(
        "AAA.SSE", date(2024, 1, 1), date(2024, 1, 8), daily,
        tick_policies=[("sig", pol)], fill_grid=[FillPolicy(0.0, 1.0)], signal_provider=sp)
    # 无 provider：signals 恒空 → 规则不命中 → default 卖档0.01成交
    rep_lo = T0BacktestRunner(data_loader=_MemLoader(bars)).run(
        "AAA.SSE", date(2024, 1, 1), date(2024, 1, 8), daily,
        tick_policies=[("sig", pol)], fill_grid=[FillPolicy(0.0, 1.0)])

    assert rep_hi.results[0].turnover_annual < rep_lo.results[0].turnover_annual
