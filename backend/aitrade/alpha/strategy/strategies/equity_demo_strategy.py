from collections import defaultdict

import polars as pl

from ...lab import BarData
from ..template import AlphaStrategy, TradeData, Direction


class EquityDemoStrategy(AlphaStrategy):
    """Equity Long-Only Demo Strategy"""

    top_k: int = 50
    n_drop: int = 5
    min_days: int = 3
    cash_ratio: float = 0.95
    min_volume: int = 100
    open_rate: float = 0.0005
    close_rate: float = 0.0015
    min_commission: int = 5
    price_add: float = 0.05

    def on_init(self) -> None:
        """Strategy initialization callback"""
        self.holding_days: defaultdict = defaultdict(int)

        self.write_log("Strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        """Trade execution callback"""
        if trade.direction == Direction.SHORT:
            self.holding_days.pop(trade.vt_symbol, None)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K-line slice callback"""
        last_signal: pl.DataFrame = self.get_signal()
        last_signal = last_signal.sort("signal", descending=True)

        pos_symbols: list[str] = [vt_symbol for vt_symbol, pos in self.pos_data.items() if pos]

        for vt_symbol in pos_symbols:
            self.holding_days[vt_symbol] += 1

        active_symbols: set[str] = set(last_signal["vt_symbol"][:self.top_k])
        active_symbols.update(pos_symbols)
        active_df: pl.DataFrame = last_signal.filter(pl.col("vt_symbol").is_in(active_symbols))

        component_symbols: set[str] = set(last_signal["vt_symbol"])
        sell_symbols: set[str] = set(pos_symbols).difference(component_symbols)

        for vt_symbol in active_df["vt_symbol"][-self.n_drop:]:
            if vt_symbol in pos_symbols:
                sell_symbols.add(vt_symbol)

        buyable_df: pl.DataFrame = last_signal.filter(~pl.col("vt_symbol").is_in(pos_symbols))
        buy_quantity: int = len(sell_symbols) + self.top_k - len(pos_symbols)
        buy_symbols: list = list(buyable_df[:buy_quantity]["vt_symbol"])

        cash: float = self.get_cash_available()

        for vt_symbol in sell_symbols:
            if self.holding_days[vt_symbol] < self.min_days:
                continue

            bar: BarData | None = bars.get(vt_symbol)
            if not bar:
                continue
            sell_price: float = bar.close_price

            sell_volume: float = self.get_pos(vt_symbol)

            self.set_target(vt_symbol, target=0)

            turnover: float = sell_price * sell_volume
            cost: float = max(turnover * self.close_rate, self.min_commission)
            cash += turnover - cost

        if buy_symbols:
            buy_value: float = cash * self.cash_ratio / len(buy_symbols)

            for vt_symbol in buy_symbols:
                buy_price: float = bars[vt_symbol].close_price
                if not buy_price:
                    continue

                buy_volume: float = round(buy_value / buy_price / self.min_volume) * self.min_volume

                self.set_target(vt_symbol, buy_volume)

        self.execute_trading(bars, price_add=self.price_add)
