"""
迭代 4 验收测试：OCO 止盈/止损出场（路径依赖，保守假设止损先到）。

入场固定：D0 概率达标 → D1 开盘建仓 @100。区别全在出场触发：
1. 止盈：后续 bar high 触及止盈价 → 按止盈价当根成交。
2. 止损：后续 bar low 触及止损价 → 按止损价当根成交。
3. 同一根 bar 同时触及止盈与止损 → 保守按止损价成交。
4. 都不触发 → 最大持有期回退（下一根成交）。

用显式 OHLC 合成行情，slippage/stamp=0 以便精确断言触发价。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from aitrade.backtest.engine import BacktestingEngine
from aitrade.backtest.types import BarData, Direction
from aitrade.cnn.strategy import CNNSignalStrategy

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


def _bars(ohlc: list[tuple[float, float, float, float]]) -> tuple[list[BarData], list[datetime]]:
    days = [START + timedelta(days=i) for i in range(len(ohlc))]
    out = [
        BarData(symbol="TEST", exchange="SZSE", datetime=days[i], interval="d",
                open_price=o, high_price=h, low_price=l, close_price=c, volume=1_000_000)
        for i, (o, h, l, c) in enumerate(ohlc)
    ]
    return out, days


def _signal(days, probs) -> pl.DataFrame:
    return pl.DataFrame({"datetime": days, "vt_symbol": [SYMBOL] * len(days), "signal": probs})


def _run_oco(ohlc, probs, setting) -> BacktestingEngine:
    bars, days = _bars(ohlc)
    engine = BacktestingEngine(data_loader=FakeLoader(bars))
    engine.set_parameters([SYMBOL], "d", days[0], days[-1] + timedelta(days=1), capital=1_000_000)
    engine.add_strategy(CNNSignalStrategy, {"buy_threshold": 0.6, "exit_mode": "oco", **setting},
                        _signal(days, probs))
    engine.load_data()
    engine.run_backtesting()
    return engine, days


def _trades(engine):
    return sorted(engine.get_all_trades(), key=lambda t: int(t.tradeid))


# ---------------------------------------------------------------------------
def test_oco_take_profit() -> None:
    # D1 建仓@100；D2 high=103 触及止盈 102 → @102 平仓
    ohlc = [
        (100, 101, 99, 100),     # D0 买入决策
        (100, 100.5, 99.5, 100), # D1 建仓 @100，未触发
        (101, 103, 100, 102),    # D2 high 103 ≥ tp102 → 止盈
        (102, 103, 101, 102),
    ]
    engine, days = _run_oco(ohlc, [0.9, 0.5, 0.5, 0.5],
                            {"take_profit": 0.02, "stop_loss": 0.03, "hold_days": 10})
    trades = _trades(engine)
    assert len(trades) == 2
    assert trades[0].direction == Direction.LONG and trades[0].price == 100.0
    assert trades[1].direction == Direction.SHORT and trades[1].price == 102.0
    assert trades[1].datetime.date() == days[2].date()


def test_oco_stop_loss() -> None:
    # D2 low=96 触及止损 97 → @97 平仓
    ohlc = [
        (100, 101, 99, 100),
        (100, 100.5, 99.5, 100),
        (99, 99.5, 96, 97),      # D2 low 96 ≤ sl97 → 止损
        (97, 98, 96, 97),
    ]
    engine, days = _run_oco(ohlc, [0.9, 0.5, 0.5, 0.5],
                            {"take_profit": 0.02, "stop_loss": 0.03, "hold_days": 10})
    trades = _trades(engine)
    assert len(trades) == 2
    assert trades[1].direction == Direction.SHORT and trades[1].price == 97.0


def test_oco_stop_loss_first_when_both_hit() -> None:
    # D2 同时触及止盈(103≥102)与止损(96≤97) → 保守按止损 97
    ohlc = [
        (100, 101, 99, 100),
        (100, 100.5, 99.5, 100),
        (100, 103, 96, 100),     # 同根同时触发
        (100, 101, 99, 100),
    ]
    engine, days = _run_oco(ohlc, [0.9, 0.5, 0.5, 0.5],
                            {"take_profit": 0.02, "stop_loss": 0.03, "hold_days": 10})
    trades = _trades(engine)
    assert len(trades) == 2
    assert trades[1].price == 97.0, "同根同时触发应保守按止损价成交"


def test_oco_max_hold_fallback() -> None:
    # 止盈止损都设很大永不触发；hold_days=1 → 最大持有回退，下一根(D2)成交
    ohlc = [
        (100, 101, 99, 100),
        (100, 100.5, 99.5, 100),
        (100, 101, 99, 100),     # D2 限价卖单成交（回退）
        (100, 101, 99, 100),
    ]
    engine, days = _run_oco(ohlc, [0.9, 0.5, 0.5, 0.5],
                            {"take_profit": 0.5, "stop_loss": 0.5, "hold_days": 1})
    trades = _trades(engine)
    assert len(trades) == 2
    assert trades[0].direction == Direction.LONG
    assert trades[1].direction == Direction.SHORT
    # 回退是下一根撮合（非止盈止损触发价）→ 成交在 D2
    assert trades[1].datetime.date() == days[2].date()
