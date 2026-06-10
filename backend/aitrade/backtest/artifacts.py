"""把回测引擎内存中的成交与逐日净值序列化为可 JSON 化结构。

仅做数据搬运与字段裁剪，不参与任何回测计算。CNN 与 Alpha 两条回测路径共用。
"""

from __future__ import annotations

from typing import Any

import polars as pl

from .types import TradeData


def serialize_trades(
    trades: dict[str, TradeData] | list[TradeData] | None,
) -> list[dict[str, Any]]:
    """把 ``engine.trades`` 序列化为成交列表，按 ``datetime`` 升序。

    入参可为 ``dict[str, TradeData]``（引擎内部存储形态）或 ``list[TradeData]``。
    无成交（``None`` / 空）返回空列表，不报错。
    """
    if not trades:
        return []

    # dict 取其 values，list 直接迭代
    items = trades.values() if isinstance(trades, dict) else trades

    out: list[dict[str, Any]] = []
    for t in items:
        out.append(
            {
                # datetime 输出为 ISO 字符串，供前端直接解析
                "datetime": t.datetime.isoformat()
                if hasattr(t.datetime, "isoformat")
                else str(t.datetime),
                "vt_symbol": t.vt_symbol,
                "direction": str(t.direction),  # long / short
                "offset": str(t.offset),        # open / close
                "price": float(t.price),
                "volume": float(t.volume),
            }
        )

    # 按时间升序，保证前端标注顺序稳定
    out.sort(key=lambda r: r["datetime"])
    return out


def serialize_equity_curve(daily_df: pl.DataFrame | None) -> list[dict[str, Any]]:
    """把 ``daily_df`` 序列化为逐日净值序列。

    ``daily_df`` 需为 ``calculate_statistics`` 之后的 DataFrame（此时才补入
    ``balance`` / ``drawdown`` / ``ddpercent`` 列）。

    以下情况返回空列表（不报错）：
    - ``daily_df`` 为 ``None`` 或空 DataFrame；
    - 缺少 ``balance`` 列（爆仓场景下 ``calculate_statistics`` 未补净值列）。
    """
    if daily_df is None or daily_df.is_empty():
        return []

    # 爆仓：calculate_statistics 判定 positive_balance=False 时不补净值列
    if "balance" not in daily_df.columns:
        return []

    out: list[dict[str, Any]] = []
    for row in daily_df.iter_rows(named=True):
        d = row["date"]
        out.append(
            {
                # date 输出为 YYYY-MM-DD 字符串
                "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
                "balance": float(row["balance"]),
                "drawdown": float(row.get("drawdown", 0.0) or 0.0),
                "ddpercent": float(row.get("ddpercent", 0.0) or 0.0),
                "net_pnl": float(row.get("net_pnl", 0.0) or 0.0),
            }
        )

    return out


def extract_benchmark_prices(
    daily_results: dict[Any, Any] | None,
    benchmark_symbol: str | None,
) -> dict[str, float]:
    """从 ``engine.daily_results`` 提取基准标的（买入持有的股票）逐日收盘价。

    返回 ``{YYYY-MM-DD: close}``。``daily_results`` 为引擎逐日盯市结果
    （``dict[date, PortfolioDailyResult]``），每个结果的 ``close_prices`` 含各
    合约当日收盘价。缺基准标的、收盘价无效（None/<=0）的交易日被跳过。

    无 ``benchmark_symbol`` 或无 ``daily_results`` 时返回空字典（不报错）。
    """
    if not benchmark_symbol or not daily_results:
        return {}

    prices: dict[str, float] = {}
    for d, daily_result in daily_results.items():
        close_prices = getattr(daily_result, "close_prices", None)
        if not close_prices:
            continue
        close = close_prices.get(benchmark_symbol)
        if close is None or close <= 0:
            continue
        key = d.isoformat() if hasattr(d, "isoformat") else str(d)
        prices[key] = float(close)
    return prices


def attach_benchmark_returns(
    equity_curve: list[dict[str, Any]],
    benchmark_prices: dict[str, float],
    capital: float,
) -> list[dict[str, Any]]:
    """为逐日净值序列叠加基准（买入持有标的）累计收益与超额收益（均为百分比）。

    逐行新增三列：
    - ``strategy_return``：策略累计收益 ``(balance / capital - 1) * 100``；
    - ``benchmark_return``：基准累计收益 ``(close / close_0 - 1) * 100``；
    - ``excess_return``：超额收益 ``strategy_return - benchmark_return``。

    无基准价、首个有效基准价 <=0 或初始资金 <=0 时，``benchmark_return`` 与
    ``excess_return`` 置 ``None``（前端跳过绘制），``strategy_return`` 仍照常计算。
    入参与返回均为同一份列表的原地增列结果。
    """
    if not equity_curve:
        return equity_curve

    base_price: float | None = None
    if benchmark_prices:
        for row in equity_curve:
            price = benchmark_prices.get(row.get("date"))
            if price is not None and price > 0:
                base_price = price
                break

    can_benchmark = base_price is not None and base_price > 0

    for row in equity_curve:
        if capital and capital > 0:
            row["strategy_return"] = (float(row["balance"]) / capital - 1.0) * 100.0
        else:
            row["strategy_return"] = None

        price = benchmark_prices.get(row.get("date")) if benchmark_prices else None
        if can_benchmark and price is not None and price > 0:
            bench_ret = (price / base_price - 1.0) * 100.0  # type: ignore[operator]
            row["benchmark_return"] = bench_ret
            if row["strategy_return"] is not None:
                row["excess_return"] = row["strategy_return"] - bench_ret
            else:
                row["excess_return"] = None
        else:
            # 当日基准缺价：保持曲线连续性，留空由前端按需跳过/补点
            row["benchmark_return"] = None
            row["excess_return"] = None

    return equity_curve


def summarize_benchmark(
    equity_curve: list[dict[str, Any]],
    benchmark_symbol: str | None,
) -> dict[str, Any]:
    """汇总基准与超额收益（百分比），用于回测统计卡片展示。

    取净值曲线中最后一个含有效基准收益的交易日作为期末口径：
    - ``benchmark_symbol``：基准标的；
    - ``benchmark_return``：基准期末累计收益（%）；
    - ``excess_return``：策略相对基准的超额收益（%）。

    无有效基准数据时返回空字典（调用方据此不展示超额收益卡片）。
    """
    last_bench: float | None = None
    last_excess: float | None = None
    for row in equity_curve:
        if row.get("benchmark_return") is not None:
            last_bench = row["benchmark_return"]
            last_excess = row.get("excess_return")

    if last_bench is None:
        return {}

    return {
        "benchmark_symbol": benchmark_symbol,
        "benchmark_return": last_bench,
        "excess_return": last_excess,
    }
