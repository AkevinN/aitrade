"""
迭代 0 验收测试：CNN 回测「label ↔ 策略出场 ↔ 撮合成交」一致性与成本补全。

覆盖：
1. fixed_hold 固定持有出场：信号触发建仓、固定持有 hold_days 个交易日后强制平仓，
   出场不依赖信号，确定性往返（与固定持有期 label 对齐）。
2. 印花税真正计入 PnL（仅卖出收取）。
3. T+1 卖出限制：当日买入的持仓不可当日卖出。
4. 阈值（threshold）默认模式向后兼容。

均使用合成行情，不依赖任何已下载数据。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from aitrade.backtest.engine import BacktestingEngine
from aitrade.backtest.types import BarData, Direction, Offset, OrderData
from aitrade.cnn.strategy import CNNSignalStrategy

SYMBOL = "TEST.SZSE"
START = datetime(2026, 1, 5)


class FakeLoader:
    """实现 BarDataLoader 协议的合成数据源。"""

    def __init__(self, bars: list[BarData], stamp_duty: float = 0.0, slippage: float = 0.0) -> None:
        self._bars = bars
        self._stamp_duty = stamp_duty
        self._slippage = slippage

    def load_bar_data(self, vt_symbol: str, interval: str, start, end) -> list[BarData]:
        return list(self._bars)

    def load_contract_settings(self) -> dict:
        return {
            SYMBOL: {
                "long_rate": 0.0003,
                "short_rate": 0.0003,
                "stamp_duty": self._stamp_duty,
                "slippage": self._slippage,
                "size": 1,
                "pricetick": 0.01,
            }
        }


def _build_bars(closes: list[float]) -> tuple[list[BarData], list[datetime]]:
    """构造满足撮合条件的合成日线：每根 bar 的最低价低于前收，保证限价单可成交。"""
    days = [START + timedelta(days=i) for i in range(len(closes))]
    bars: list[BarData] = []
    for i, close in enumerate(closes):
        prev_close = closes[i - 1] if i > 0 else close
        open_price = prev_close
        high_price = max(open_price, close) + 1.0
        low_price = min(open_price, close) - 1.0
        bars.append(
            BarData(
                symbol="TEST",
                exchange="SZSE",
                datetime=days[i],
                interval="d",
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close,
                volume=1_000_000,
            )
        )
    return bars, days


def _signal_df(days: list[datetime], probs: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": days,
            "vt_symbol": [SYMBOL] * len(days),
            "signal": probs,
        }
    )


def _run(loader: FakeLoader, days: list[datetime], signal_df: pl.DataFrame, setting: dict,
         *, t_plus1: bool = False) -> BacktestingEngine:
    engine = BacktestingEngine(data_loader=loader)
    engine.set_parameters(
        vt_symbols=[SYMBOL],
        interval="d",
        start=days[0],
        end=days[-1] + timedelta(days=1),
        capital=1_000_000,
    )
    engine.t_plus1 = t_plus1
    engine.add_strategy(CNNSignalStrategy, setting, signal_df)
    engine.load_data()
    engine.run_backtesting()
    return engine


def _ordered_trades(engine: BacktestingEngine) -> list:
    return sorted(engine.get_all_trades(), key=lambda t: int(t.tradeid))


# ---------------------------------------------------------------------------
# 1. fixed_hold 固定持有出场
# ---------------------------------------------------------------------------
def test_fixed_hold_round_trip_exits_after_hold_days() -> None:
    closes = [100.0 + i for i in range(8)]
    bars, days = _build_bars(closes)
    # 仅在 D0、D5 给出强买入信号；其余为中性，验证出场不依赖信号
    probs = [0.9, 0.5, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5]
    loader = FakeLoader(bars)
    signal_df = _signal_df(days, probs)

    engine = _run(
        loader, days, signal_df,
        {"buy_threshold": 0.6, "exit_mode": "fixed_hold", "hold_days": 1},
    )

    trades = _ordered_trades(engine)
    # 两次完整往返：买、卖、买、卖
    assert engine.trade_count == 4, f"期望 4 笔成交，实际 {engine.trade_count}"
    assert [t.direction for t in trades] == [
        Direction.LONG, Direction.SHORT, Direction.LONG, Direction.SHORT,
    ]
    # 建仓 D1 成交、持有 1 个交易日后于 D2 平仓（确定性，与 label 出场口径一致）
    assert trades[0].datetime.date() == days[1].date()
    assert trades[1].datetime.date() == days[2].date()
    assert trades[2].datetime.date() == days[6].date()
    assert trades[3].datetime.date() == days[7].date()
    # 回测结束应为空仓
    assert engine.strategy.get_pos(SYMBOL) == 0


def test_fixed_hold_two_day_holding() -> None:
    closes = [100.0 + i for i in range(8)]
    bars, days = _build_bars(closes)
    probs = [0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    loader = FakeLoader(bars)
    signal_df = _signal_df(days, probs)

    engine = _run(
        loader, days, signal_df,
        {"buy_threshold": 0.6, "exit_mode": "fixed_hold", "hold_days": 2},
    )

    trades = _ordered_trades(engine)
    assert engine.trade_count == 2
    assert trades[0].direction == Direction.LONG
    assert trades[1].direction == Direction.SHORT
    # 建仓 D1 成交；持有计数 D1→1、D2→2，第 2 日触发出场决策，下一根 D3 成交平仓
    assert trades[0].datetime.date() == days[1].date()
    assert trades[1].datetime.date() == days[3].date()


# ---------------------------------------------------------------------------
# 2. 印花税计入 PnL（仅卖出）
# ---------------------------------------------------------------------------
def test_stamp_duty_reduces_net_pnl() -> None:
    closes = [100.0 + i for i in range(8)]
    bars, days = _build_bars(closes)
    probs = [0.9, 0.5, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5]
    setting = {"buy_threshold": 0.6, "exit_mode": "fixed_hold", "hold_days": 1}

    eng_no = _run(FakeLoader(bars, stamp_duty=0.0), days, _signal_df(days, probs), dict(setting))
    eng_yes = _run(FakeLoader(bars, stamp_duty=0.001), days, _signal_df(days, probs), dict(setting))

    eng_no.calculate_result()
    eng_yes.calculate_result()
    stat_no = eng_no.calculate_statistics()
    stat_yes = eng_yes.calculate_statistics()

    # 成交一致，唯一差异是卖出印花税：手续费更高、净盈亏更低
    assert eng_no.trade_count == eng_yes.trade_count == 4
    assert stat_yes["total_commission"] > stat_no["total_commission"]
    assert stat_yes["total_net_pnl"] < stat_no["total_net_pnl"]


# ---------------------------------------------------------------------------
# 3. T+1 卖出限制
# ---------------------------------------------------------------------------
def test_t_plus_1_blocks_same_day_sell() -> None:
    closes = [100.0, 100.0, 100.0]
    bars, days = _build_bars(closes)
    loader = FakeLoader(bars)

    engine = BacktestingEngine(data_loader=loader)
    engine.set_parameters([SYMBOL], "d", days[0], days[-1] + timedelta(days=1), capital=1_000_000)
    engine.add_strategy(CNNSignalStrategy, {"exit_mode": "fixed_hold"}, _signal_df(days, [0.5, 0.5, 0.5]))
    engine.t_plus1 = True

    # 模拟：当日已买入，挂出一张卖单，尝试当日撮合
    engine.datetime = days[1]
    engine.buy_dates[SYMBOL] = days[1].date()
    engine.bars[SYMBOL] = bars[1]
    engine.pre_closes[SYMBOL] = 100.0
    sell_order = OrderData(
        symbol="TEST", exchange="SZSE", orderid="1",
        direction=Direction.SHORT, offset=Offset.CLOSE,
        price=99.0, volume=100, status="nottraded",
        datetime=days[1], gateway_name=engine.gateway_name,
    )
    engine.active_limit_orders[sell_order.vt_orderid] = sell_order

    engine.cross_order()
    assert engine.trade_count == 0, "T+1 应拦截当日买入当日卖出"
    assert sell_order.vt_orderid in engine.active_limit_orders, "被拦截的卖单应保留待次日撮合"

    # 次日：不再是当日买入，应可成交
    engine.datetime = days[2]
    engine.bars[SYMBOL] = bars[2]
    engine.cross_order()
    assert engine.trade_count == 1, "次日应允许卖出"


# ---------------------------------------------------------------------------
# 4. 阈值模式向后兼容
# ---------------------------------------------------------------------------
def test_threshold_mode_buys_and_sells_on_signal() -> None:
    closes = [100.0 + i for i in range(6)]
    bars, days = _build_bars(closes)
    # 高概率买入，随后低概率触发阈值平仓
    probs = [0.9, 0.9, 0.1, 0.1, 0.5, 0.5]
    loader = FakeLoader(bars)
    signal_df = _signal_df(days, probs)

    engine = _run(
        loader, days, signal_df,
        {"buy_threshold": 0.6, "sell_threshold": 0.4, "exit_mode": "threshold"},
    )

    trades = _ordered_trades(engine)
    assert engine.trade_count >= 2
    assert trades[0].direction == Direction.LONG
    # 出现过卖出（阈值触发平仓）
    assert any(t.direction == Direction.SHORT for t in trades)
    assert engine.strategy.get_pos(SYMBOL) == 0
