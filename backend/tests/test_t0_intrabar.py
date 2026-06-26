"""引擎 buy_to_open_intrabar 测试：当根 bar 内按给定价直接买入（做T收盘/开盘回半仓用）。

对称于既有 sell_to_close_intrabar：立即结算、含成本、更新现金与持仓、记 T+1 建仓日。

Feature: half-position-t0-backtest
"""

from __future__ import annotations

from datetime import datetime, timedelta

from aitrade.backtest.engine import BacktestingEngine
from aitrade.backtest.strategy import BaseStrategy
from aitrade.backtest.types import BarData


class _NoopStrategy(BaseStrategy):
    def on_init(self) -> None: ...
    def on_bars(self, bars) -> None: ...
    def on_trade(self, trade) -> None: ...


def _engine_with_one_bar():
    vt = "AAA.SSE"
    bar = BarData(symbol="AAA", exchange="SSE", datetime=datetime(2024, 1, 1), interval="d",
                  open_price=10.0, high_price=10.0, low_price=10.0, close_price=10.0, volume=10000)

    class _L:
        def load_bar_data(self, *a): return [bar]
        def load_contract_settings(self): return {}

    eng = BacktestingEngine(data_loader=_L())
    eng.set_parameters([vt], "d", bar.datetime, bar.datetime, capital=1_000_000)
    eng.sizes[vt] = 1
    eng.priceticks[vt] = 0.01
    eng.long_rates[vt] = 0.0003
    eng.short_rates[vt] = 0.0003
    eng.stamp_duties[vt] = 0.0
    eng.slippages[vt] = 0.0
    eng.add_strategy(_NoopStrategy, {}, None)
    eng.load_data()
    eng.datetime = bar.datetime
    eng.bars[vt] = bar
    return eng, vt


def test_buy_to_open_intrabar_increases_position_and_spends_cash() -> None:
    eng, vt = _engine_with_one_bar()
    cash0 = eng.cash
    eng.buy_to_open_intrabar(vt, price=10.0, volume=1000)
    assert eng.strategy.get_pos(vt) == 1000
    # 现金减少 = 成交额 + 佣金（买入无印花税）
    assert eng.cash < cash0
    spent = cash0 - eng.cash
    assert abs(spent - (10.0 * 1000 * (1 + 0.0003))) < 1e-6


def test_buy_to_open_intrabar_records_t1_buy_date() -> None:
    eng, vt = _engine_with_one_bar()
    eng.buy_to_open_intrabar(vt, price=10.0, volume=1000)
    assert eng.buy_dates.get(vt) == eng.datetime.date()
