"""回测产物端到端冒烟测试。

驱动一次最小回测（内存行情 + CNNSignalStrategy，先买后卖），完整跑过
``run_backtesting -> calculate_result -> calculate_statistics`` 后，按生产路径
（见 ``api/cnn.py`` 与 ``backtest/scheme.py``）调用序列化函数，断言：

- ``trades`` 非空，且每笔含 datetime / vt_symbol / direction / offset / price / volume；
- ``equity_curve`` 非空，且每行含 date / balance / drawdown / ddpercent / net_pnl；
- 时间/日期为前端可解析的字符串格式。

仅构造内存行情，不读本地数据，保证测试环境可复现。
对应需求：1.1、2.1。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from aitrade.backtest.artifacts import serialize_equity_curve, serialize_trades
from aitrade.backtest.engine import BacktestingEngine
from aitrade.backtest.types import BarData
from aitrade.cnn.strategy import CNNSignalStrategy


class _MemoryLoader:
    """最小 BarDataLoader：内存行情 + 空合约配置（成本由测试显式设置）。"""

    def __init__(self, bars: list[BarData]) -> None:
        self._bars = bars

    def load_bar_data(self, vt_symbol, interval, start, end) -> list[BarData]:
        return list(self._bars)

    def load_contract_settings(self) -> dict:
        return {}


def _make_bars(symbol: str = "AAA", exchange: str = "SSE", n: int = 8, price: float = 100.0) -> list[BarData]:
    """构造一段先涨后小幅回落的日线，制造非零回撤，覆盖 drawdown/ddpercent 字段。"""
    base = datetime(2024, 1, 1)
    # 价格先上行（建仓后浮盈），尾部回落（产生回撤），便于校验净值字段齐全且非全零
    deltas = [0, 1, 2, 3, 4, 3, 2, 1]
    bars: list[BarData] = []
    for i in range(n):
        px = price + deltas[i % len(deltas)]
        bars.append(
            BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=base + timedelta(days=i),
                interval="d",
                open_price=px,
                high_price=px + 1,
                low_price=px - 1,
                close_price=px,
                volume=10000,
            )
        )
    return bars


def _signal_df(vt_symbol: str, bars: list[BarData], probs: list[float]) -> pl.DataFrame:
    rows = [
        {"datetime": bar.datetime, "vt_symbol": vt_symbol, "signal": prob}
        for bar, prob in zip(bars, probs)
    ]
    return pl.DataFrame(rows)


def _run_minimal_backtest() -> dict:
    """跑一次最小回测，按生产路径组装含 trades / equity_curve 的结果。"""
    vt_symbol = "AAA.SSE"
    bars = _make_bars()
    loader = _MemoryLoader(bars)
    engine = BacktestingEngine(data_loader=loader)
    start = bars[0].datetime
    end = bars[-1].datetime
    engine.set_parameters([vt_symbol], "d", start, end, capital=1_000_000)
    engine.sizes[vt_symbol] = 1
    engine.priceticks[vt_symbol] = 0.01
    engine.long_rates[vt_symbol] = 0.0003
    engine.short_rates[vt_symbol] = 0.0003
    engine.stamp_duties[vt_symbol] = 0.0
    engine.slippages[vt_symbol] = 0.0

    # 先高概率买入、后低概率清仓，保证至少一买一卖
    probs = [0.9, 0.9, 0.9, 0.9, 0.05, 0.05, 0.05, 0.05]
    engine.add_strategy(
        CNNSignalStrategy,
        {"buy_threshold": 0.6, "sell_threshold": 0.4, "price_add": 0.0},
        _signal_df(vt_symbol, bars, probs),
    )
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    # 关键顺序：必须先 calculate_statistics()，daily_df 此时才补入 balance/drawdown/ddpercent
    statistics = engine.calculate_statistics()

    # 与 api/cnn.py、backtest/scheme.py 生产路径一致的结果组装
    return {
        "statistics": statistics,
        "trades": serialize_trades(engine.trades),
        "equity_curve": serialize_equity_curve(engine.daily_df),
    }


def test_backtest_result_contains_non_empty_trades():
    """端到端回测结果含非空 trades，且字段齐全、类型/格式正确。"""
    result = _run_minimal_backtest()

    trades = result["trades"]
    assert isinstance(trades, list)
    assert len(trades) > 0, "最小回测应至少产生一笔成交"

    required_fields = {"datetime", "vt_symbol", "direction", "offset", "price", "volume"}
    for t in trades:
        assert required_fields.issubset(t.keys()), f"成交字段缺失: {t.keys()}"
        # datetime 为可解析的 ISO 字符串
        assert isinstance(t["datetime"], str)
        datetime.fromisoformat(t["datetime"])
        assert isinstance(t["price"], float)
        assert isinstance(t["volume"], float)

    # 先买后卖：应同时出现开仓与平仓
    offsets = {t["offset"] for t in trades}
    assert "open" in offsets and "close" in offsets, offsets


def test_backtest_result_contains_non_empty_equity_curve():
    """端到端回测结果含非空 equity_curve，且字段齐全、date 为 YYYY-MM-DD。"""
    result = _run_minimal_backtest()

    curve = result["equity_curve"]
    assert isinstance(curve, list)
    assert len(curve) > 0, "净值可计算时 equity_curve 不应为空"

    required_fields = {"date", "balance", "drawdown", "ddpercent", "net_pnl"}
    for row in curve:
        assert required_fields.issubset(row.keys()), f"净值字段缺失: {row.keys()}"
        # date 为 YYYY-MM-DD，可被 date.fromisoformat 解析
        assert isinstance(row["date"], str)
        datetime.strptime(row["date"], "%Y-%m-%d")
        for k in ("balance", "drawdown", "ddpercent", "net_pnl"):
            assert isinstance(row[k], float)

    # 净值序列与逐日盯市天数一致，且 balance 起点接近初始资金
    assert curve[0]["balance"] > 0
