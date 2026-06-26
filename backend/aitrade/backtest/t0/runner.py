"""做 T 回测编排（T0BacktestRunner）：扫 {TickPolicy × FillPolicy} 跑引擎，出区间报告。

核心产物是 **成交敏感性区间**——同一标的、同一档位策略，在不同成交假设（FillPolicy）下
策略收益/超额/Sharpe/回撤的取值区间，用以替代"单一数字"。同时给逐年/逐月超额（vs 满仓
买入持有 与 每日再平衡半仓）、换手、命中分布。撮合/成本/T+1 复用引擎单一事实源；做 T 限价
单无价格滑点，故默认 ``slippage=0``，真实摩擦由 FillPolicy 表达。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl

from ...config import ALPHA_LAB_PATH
from ..engine import BacktestingEngine
from ..types import FillPolicy
from .strategy import HalfPositionT0Strategy
from .tick_policy import FixedTick, TickPolicy

_TRADING_DAYS = 244


@dataclass
class T0RunResult:
    """单个 (TickPolicy × FillPolicy) 组合的回测结果。"""

    tick_label: str
    fill: dict[str, float]
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    turnover_annual: float
    yearly: list[dict[str, Any]] = field(default_factory=list)
    monthly_excess: list[dict[str, Any]] = field(default_factory=list)
    hit_dist: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的 dict。"""
        return {
            "tick_label": self.tick_label, "fill": self.fill,
            "total_return": self.total_return, "cagr": self.cagr, "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown, "turnover_annual": self.turnover_annual,
            "yearly": self.yearly, "monthly_excess": self.monthly_excess, "hit_dist": self.hit_dist,
        }


@dataclass
class T0Report:
    """做 T 回测总报告：多组合结果 + 成交敏感性区间。"""

    symbol: str
    eval_window: tuple[date, date]
    results: list[T0RunResult] = field(default_factory=list)

    def fill_sensitivity(self) -> list[dict[str, Any]]:
        """按 (tick_label, fill) 汇总各组合的 total_return/sharpe/mdd，便于看区间。"""
        return [{"tick_label": r.tick_label, "fill": r.fill, "total_return": r.total_return,
                 "sharpe": r.sharpe, "max_drawdown": r.max_drawdown} for r in self.results]

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的 dict。"""
        return {"symbol": self.symbol,
                "eval_window": [self.eval_window[0].isoformat(), self.eval_window[1].isoformat()],
                "fill_sensitivity": self.fill_sensitivity(),
                "results": [r.to_dict() for r in self.results]}


def _benchmarks(daily: pl.DataFrame) -> pl.DataFrame:
    """由日线 close 计算满仓持有与每日再平衡半仓的净值（期初=1）。

    Args:
        daily: 含 d(date), close 列，按日期升序。

    Returns:
        追加 bh(满仓净值)、half_bh(再平衡半仓净值) 两列的 DataFrame。
    """
    close = daily["close"].to_numpy()
    bh = close / close[0]
    r = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    half = np.cumprod(1.0 + 0.5 * r)
    return daily.with_columns([pl.Series("bh", bh), pl.Series("half_bh", half)])


def _hit_distribution(daily: pl.DataFrame, sell_tick: float, buy_tick: float) -> dict[str, float]:
    """日线口径下"开盘±档位是否被触及"的命中分布（both/onlyS/onlyB/none 占比）。"""
    o = daily["open"].to_numpy(); hi = daily["high"].to_numpy(); lo = daily["low"].to_numpy()
    S = hi >= o + sell_tick; B = lo <= o - buy_tick
    n = max(len(o), 1)
    return {"both": float((S & B).mean()), "onlyS": float((S & ~B).mean()),
            "onlyB": float((B & ~S).mean()), "none": float((~S & ~B).mean())}


class T0BacktestRunner:
    """做 T 回测编排器。"""

    def __init__(self, data_loader: Any = None) -> None:
        """初始化。

        Args:
            data_loader: 引擎用的 BarDataLoader；缺省用 AlphaLab(ALPHA_LAB_PATH)。
                测试可注入内存 loader。
        """
        if data_loader is None:
            from ...alpha.lab import AlphaLab
            data_loader = AlphaLab(ALPHA_LAB_PATH)
        self.loader = data_loader

    def run_single(self, symbol: str, start: date, end: date, tick_policy: TickPolicy,
                   fill_policy: FillPolicy, daily: pl.DataFrame, capital: int = 1_000_000,
                   commission_rate: float = 0.0003, stamp_duty: float = 0.0005,
                   swing_frac: float = 1.0, base_weight: float = 0.5,
                   tick_label: str = "") -> T0RunResult:
        """跑单个组合并汇总指标 + 逐年/逐月超额 + 命中分布。

        Args:
            symbol: 标的 vt_symbol，如 "000415.SZSE"。
            start: 评估窗起（含）。
            end: 评估窗止（含）。
            tick_policy: 档位策略。
            fill_policy: 成交保真度策略。
            daily: 评估窗的日线 OHLC（d/open/high/low/close），供基准与命中分布。
            capital: 初始资金。
            commission_rate: 单边佣金率。
            stamp_duty: 卖出印花税率。
            swing_frac: 做 T 摆动幅度占半仓比例。
            base_weight: 半仓锚权重。
            tick_label: 该档位策略的标签（报告用）。

        Returns:
            T0RunResult。
        """
        engine = BacktestingEngine(data_loader=self.loader)
        engine.set_parameters([symbol], "1m",
                              datetime.combine(start, datetime.min.time()),
                              datetime.combine(end, datetime.max.time()), capital=capital)
        if symbol not in engine.sizes:
            engine.sizes[symbol] = 1
            engine.priceticks[symbol] = 0.01
        engine.long_rates[symbol] = commission_rate
        engine.short_rates[symbol] = commission_rate
        engine.stamp_duties[symbol] = stamp_duty
        engine.slippages[symbol] = 0.0            # 限价单无价格滑点（Property 8）
        engine.t_plus1 = True
        engine.fill_policy = fill_policy
        engine.add_strategy(HalfPositionT0Strategy, {
            "vt_symbol": symbol, "tick_policy": tick_policy,
            "swing_frac": swing_frac, "base_weight": base_weight,
        }, None)
        engine.load_data()
        engine.run_backtesting()

        daily_res = engine.calculate_result()
        bench = _benchmarks(daily.sort("d"))
        result = self._summarize(daily_res, bench, capital, tick_policy, tick_label, fill_policy, engine)
        return result

    def _summarize(self, daily_res, bench, capital, tick_policy, tick_label, fill_policy, engine) -> T0RunResult:
        """把引擎日结果 + 基准压成 T0RunResult（指标/逐年/逐月/命中）。"""
        # 策略净值（balance = capital + cumsum(net_pnl)），按日期对齐基准
        if daily_res is None or daily_res.height == 0:
            eq = np.array([float(capital)])
            dates = [bench["d"][0]]
        else:
            net = daily_res["net_pnl"].to_numpy()
            eq = np.cumsum(net) + capital
            dates = list(daily_res["date"])
        equity = eq / eq[0]
        daily_ret = np.concatenate([[0.0], equity[1:] / equity[:-1] - 1.0])
        n = len(equity)
        yrs = n / _TRADING_DAYS
        total = float(equity[-1] - 1.0)
        cagr = float(equity[-1] ** (1 / yrs) - 1.0) if yrs > 0 and equity[-1] > 0 else float("nan")
        vol = float(np.std(daily_ret, ddof=1) * np.sqrt(_TRADING_DAYS)) if n > 1 else float("nan")
        sharpe = float(np.mean(daily_ret) * _TRADING_DAYS / vol) if vol and vol > 0 else float("nan")
        peak = np.maximum.accumulate(equity)
        mdd = float(np.min(equity / peak - 1.0))
        trades = engine.get_all_trades()
        turn = len(trades) / max(yrs, 1e-9)

        # 逐年/逐月超额：把策略净值与基准净值都按年/月切（用基准的日期轴）
        sell_tick, buy_tick = tick_policy.ticks_for(bench["d"][-1], _full_hist(bench))
        yearly = _by_period(equity, dates, bench, "year")
        monthly = _by_period(equity, dates, bench, "month")
        return T0RunResult(
            tick_label=tick_label or type(tick_policy).__name__,
            fill={"penetration": fill_policy.fill_penetration, "ratio": fill_policy.fill_ratio},
            total_return=round(total, 4), cagr=round(cagr, 4) if cagr == cagr else cagr,
            sharpe=round(sharpe, 3) if sharpe == sharpe else sharpe,
            max_drawdown=round(mdd, 4), turnover_annual=round(turn, 1),
            yearly=yearly,
            monthly_excess=[{"ym": m["ym"], "excess_vs_bh": m["excess_vs_bh"]} for m in monthly],
            hit_dist=_hit_distribution(bench, sell_tick, buy_tick),
        )

    def run(self, symbol: str, start: date, end: date, daily: pl.DataFrame,
            tick_policies: list[tuple[str, TickPolicy]] | None = None,
            fill_grid: list[FillPolicy] | None = None, **kw) -> T0Report:
        """扫 {TickPolicy × FillPolicy} 网格，汇总成 T0Report（含成交敏感性区间）。"""
        if tick_policies is None:
            tick_policies = [("fixed_2fen", FixedTick(0.02, 0.02))]
        if fill_grid is None:
            fill_grid = [FillPolicy(0.0, 1.0), FillPolicy(0.01, 1.0), FillPolicy(0.0, 0.5)]
        # Property 6 / Req 6.3：用画像标定的档位，其标定窗必须严格早于评估窗（杜绝样本内拟合）
        for label, tp in tick_policies:
            prof = getattr(tp, "profile", None)
            win = getattr(prof, "window", None)
            if win is not None and win[1] >= start:
                raise ValueError(
                    f"档位策略 {label!r} 的标定窗 {win} 必须严格早于评估窗起点 {start}（杜绝样本内拟合/前视）")
        report = T0Report(symbol=symbol, eval_window=(start, end))
        for label, tp in tick_policies:
            for fp in fill_grid:
                report.results.append(
                    self.run_single(symbol, start, end, tp, fp, daily, tick_label=label, **kw))
        return report


def _full_hist(bench: pl.DataFrame):
    """从 benchmark 日线构造一个完整 DailyHistory（仅用于给档位策略取代表性档位）。"""
    from .tick_policy import DailyBar, DailyHistory
    h = DailyHistory()
    for row in bench.iter_rows(named=True):
        h.append(DailyBar(row["d"], row["open"], row["high"], row["low"], row["close"]))
    return h


def _by_period(equity: np.ndarray, dates: list, bench: pl.DataFrame, gran: str) -> list[dict]:
    """把策略净值与基准净值按年/月切片算各自收益与超额。"""
    # 对齐：策略净值 dates 与基准 bench["d"] 可能长度一致（同一回测窗）
    bdates = list(bench["d"])
    bh = bench["bh"].to_numpy(); half = bench["half_bh"].to_numpy()
    # 用基准日期为准，策略净值按索引对齐（同窗等长）；不等长时按 min 截断
    m = min(len(equity), len(bdates))
    eq = equity[:m]; bh = bh[:m]; half = half[:m]; ds = bdates[:m]
    key = (lambda d: d.year) if gran == "year" else (lambda d: f"{d.year}-{d.month:02d}")
    groups: dict = {}
    for i, d in enumerate(ds):
        groups.setdefault(key(d), []).append(i)
    out = []
    for g, idx in groups.items():
        i0, i1 = idx[0], idx[-1]
        s = eq[i1] / eq[i0] - 1.0
        b = bh[i1] / bh[i0] - 1.0
        hb = half[i1] / half[i0] - 1.0
        row = {"strat": round(s, 4), "bh": round(b, 4), "half_bh": round(hb, 4),
               "excess_vs_bh": round(s - b, 4), "excess_vs_half_bh": round(s - hb, 4)}
        if gran == "year":
            row = {"year": g, **row}
        else:
            row = {"ym": g, **row}
        out.append(row)
    return out
