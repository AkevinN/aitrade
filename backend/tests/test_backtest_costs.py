"""回测成本回归测试：验证滑点与卖出印花税被正确计入成交价与净盈亏。

仅依赖共享回测引擎 + CNNSignalStrategy，构造内存行情，不读本地数据。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

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


def _make_bars(symbol="AAA", exchange="SSE", n=6, price=100.0) -> list[BarData]:
    base = datetime(2024, 1, 1)
    bars: list[BarData] = []
    for i in range(n):
        # 价格平稳上行，open≈close，保证撮合稳定、便于对账
        px = price + i
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
    rows = []
    for bar, prob in zip(bars, probs):
        rows.append({"datetime": bar.datetime, "vt_symbol": vt_symbol, "signal": prob})
    return pl.DataFrame(rows)


def _run(stamp_duty: float, slippage: float, commission: float = 0.0003) -> dict:
    vt_symbol = "AAA.SSE"
    bars = _make_bars()
    loader = _MemoryLoader(bars)
    engine = BacktestingEngine(data_loader=loader)
    start = bars[0].datetime
    end = bars[-1].datetime
    engine.set_parameters([vt_symbol], "d", start, end, capital=1_000_000)
    engine.sizes[vt_symbol] = 1
    engine.priceticks[vt_symbol] = 0.01
    engine.long_rates[vt_symbol] = commission
    engine.short_rates[vt_symbol] = commission
    engine.stamp_duties[vt_symbol] = stamp_duty
    engine.slippages[vt_symbol] = slippage

    # 先买后卖：前几根高概率买入，后几根低概率清仓
    probs = [0.9, 0.9, 0.9, 0.05, 0.05, 0.05]
    engine.add_strategy(
        CNNSignalStrategy,
        {"buy_threshold": 0.6, "sell_threshold": 0.4, "price_add": 0.0},
        _signal_df(vt_symbol, bars, probs),
    )
    engine.load_data()
    engine.run_backtesting()
    daily_df = engine.calculate_result()
    # 直接从逐日盯市表对账成本，避免依赖 calculate_statistics（零回撤时会触发既有除零边界）
    total_commission = float(daily_df["commission"].sum())
    total_net_pnl = float(daily_df["net_pnl"].sum())
    trades = engine.get_all_trades()
    return {
        "total_commission": total_commission,
        "total_net_pnl": total_net_pnl,
        "trades": trades,
    }


def test_slippage_worsens_fill_price() -> None:
    """同一撮合，买入价随滑点上升、卖出价随滑点下降。"""
    no_slip = _run(stamp_duty=0.0, slippage=0.0)
    with_slip = _run(stamp_duty=0.0, slippage=0.01)

    def first(direction: str, trades):
        return next(t for t in trades if t.direction == direction)

    buy_no = first("long", no_slip["trades"]).price
    buy_slip = first("long", with_slip["trades"]).price
    sell_no = first("short", no_slip["trades"]).price
    sell_slip = first("short", with_slip["trades"]).price

    assert buy_slip > buy_no, (buy_slip, buy_no)
    assert sell_slip < sell_no, (sell_slip, sell_no)


def test_stamp_duty_increases_total_cost() -> None:
    """加上卖出印花税后，总成本（commission 字段含税费）上升、净盈亏下降。"""
    no_stamp = _run(stamp_duty=0.0, slippage=0.0)
    with_stamp = _run(stamp_duty=0.001, slippage=0.0)

    cost_no = no_stamp["total_commission"]
    cost_stamp = with_stamp["total_commission"]
    assert cost_stamp > cost_no, (cost_stamp, cost_no)

    # 印花税只对卖出收取，加税后总成本上升、净盈亏下降
    assert (cost_stamp - cost_no) > 0
    assert with_stamp["total_net_pnl"] < no_stamp["total_net_pnl"]
