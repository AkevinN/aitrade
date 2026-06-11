"""
回测撮合成交价口径（fill_price_mode）验收：与训练 label 的 price_ref 一一对齐。

覆盖：
1. fill_price_mode=close：限价单按撮合 bar 的「收盘价」成交（对齐 next_close / MOC）。
2. fill_price_mode=vwap：按撮合 bar 的「均价 成交额/成交量」成交（对齐 next_vwap / VWAP）。
3. fill_price_mode=vwap 在量为 0 / 无成交额时回退到收盘价（与 dataset 口径一致）。
4. 默认 fill_price_mode=open 保持旧行为（按开盘价成交，对齐 next_open）。

均使用合成行情，不依赖任何已下载数据。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from aitrade.backtest.engine import BacktestingEngine, round_to
from aitrade.backtest.types import BarData, Direction
from aitrade.cnn.strategy import CNNSignalStrategy

SYMBOL = "TEST.SZSE"
START = datetime(2026, 1, 5)


class FakeLoader:
    def __init__(self, bars: list[BarData]) -> None:
        self._bars = bars

    def load_bar_data(self, vt_symbol: str, interval: str, start, end) -> list[BarData]:
        return list(self._bars)

    def load_contract_settings(self) -> dict:
        return {
            SYMBOL: {
                "long_rate": 0.0003,
                "short_rate": 0.0003,
                "stamp_duty": 0.0,
                "slippage": 0.0,
                "size": 1,
                "pricetick": 0.01,
            }
        }


def _build_bars(
    closes: list[float],
    *,
    vwap_delta: float = 5.0,
    volume: float = 1_000.0,
    with_turnover: bool = True,
) -> tuple[list[BarData], list[datetime]]:
    """构造合成日线：open=前收，vwap(=turnover/volume)=close+vwap_delta，三价互不相同。"""
    days = [START + timedelta(days=i) for i in range(len(closes))]
    bars: list[BarData] = []
    for i, close in enumerate(closes):
        prev_close = closes[i - 1] if i > 0 else close
        open_price = prev_close
        vwap = close + vwap_delta
        bars.append(
            BarData(
                symbol="TEST",
                exchange="SZSE",
                datetime=days[i],
                interval="d",
                open_price=open_price,
                high_price=max(open_price, close, vwap) + 1.0,
                low_price=min(open_price, close, vwap) - 1.0,
                close_price=close,
                volume=volume,
                turnover=(vwap * volume) if with_turnover else 0.0,
            )
        )
    return bars, days


def _signal_df(days: list[datetime], probs: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"datetime": days, "vt_symbol": [SYMBOL] * len(days), "signal": probs}
    )


def _run(bars: list[BarData], days: list[datetime], probs: list[float], fill_mode: str):
    engine = BacktestingEngine(data_loader=FakeLoader(bars))
    engine.set_parameters(
        vt_symbols=[SYMBOL], interval="d",
        start=days[0], end=days[-1] + timedelta(days=1), capital=1_000_000,
    )
    engine.fill_price_mode = fill_mode
    engine.add_strategy(
        CNNSignalStrategy,
        {"buy_threshold": 0.6, "exit_mode": "fixed_hold", "hold_days": 1},
        _signal_df(days, probs),
    )
    engine.load_data()
    engine.run_backtesting()
    return engine


def _ordered_trades(engine) -> list:
    return sorted(engine.get_all_trades(), key=lambda t: int(t.tradeid))


# D0 强买入信号、hold_days=1 → D1 建仓、D2 平仓。
_CLOSES = [100.0, 101.0, 102.0, 103.0]
_PROBS = [0.9, 0.5, 0.5, 0.5]


def test_fill_mode_close_uses_bar_close() -> None:
    bars, days = _build_bars(_CLOSES)
    engine = _run(bars, days, _PROBS, "close")
    trades = _ordered_trades(engine)

    assert engine.trade_count == 2
    assert trades[0].direction == Direction.LONG
    assert trades[1].direction == Direction.SHORT
    # 建仓按 D1 收盘价、平仓按 D2 收盘价成交（对齐 next_close）
    assert trades[0].price == round_to(101.0, 0.01)
    assert trades[1].price == round_to(102.0, 0.01)


def test_fill_mode_vwap_uses_turnover_over_volume() -> None:
    bars, days = _build_bars(_CLOSES, vwap_delta=5.0)
    engine = _run(bars, days, _PROBS, "vwap")
    trades = _ordered_trades(engine)

    assert engine.trade_count == 2
    # vwap = close + 5 → 建仓 D1=106、平仓 D2=107（对齐 next_vwap）
    assert trades[0].price == round_to(106.0, 0.01)
    assert trades[1].price == round_to(107.0, 0.01)


def test_fill_mode_vwap_falls_back_to_close_without_turnover() -> None:
    bars, days = _build_bars(_CLOSES, with_turnover=True, volume=0.0)
    engine = _run(bars, days, _PROBS, "vwap")
    trades = _ordered_trades(engine)

    assert engine.trade_count == 2
    # 量为 0 → vwap 无法计算，回退收盘价
    assert trades[0].price == round_to(101.0, 0.01)
    assert trades[1].price == round_to(102.0, 0.01)


def test_fill_mode_open_is_default_behavior() -> None:
    bars, days = _build_bars(_CLOSES)
    engine = _run(bars, days, _PROBS, "open")
    trades = _ordered_trades(engine)

    assert engine.trade_count == 2
    # open=前收 → 建仓 D1 开盘=close[D0]=100、平仓 D2 开盘=close[D1]=101（旧行为，对齐 next_open）
    assert trades[0].price == round_to(100.0, 0.01)
    assert trades[1].price == round_to(101.0, 0.01)
