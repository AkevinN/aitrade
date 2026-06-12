"""A 股多头 Demo 策略——因子信号驱动的持仓换仓示例。"""

from collections import defaultdict

import polars as pl

from ...lab import BarData
from ..template import AlphaStrategy, TradeData, Direction


class EquityDemoStrategy(AlphaStrategy):
    """A 股纯多头演示策略，基于因子信号排名进行持仓轮动。

    策略逻辑：
    1. 每个 K 线切片取最新因子信号，按信号值降序选出 top_k 只股票为目标持仓；
    2. 末尾排名倒数 n_drop 且当前持仓的股票被列为卖出候选；
    3. 不在成分池的持仓股也强制卖出；
    4. 持仓天数不足 min_days 的股票暂不卖出（防止频繁换仓）；
    5. 卖出后腾出仓位，按剩余现金均仓买入新标的。

    类级参数说明（可子类覆盖）：
        top_k: 目标持仓数量，默认 50。
        n_drop: 每次最多卖出信号末尾名次数，默认 5。
        min_days: 最短持仓天数，默认 3。
        cash_ratio: 可用现金使用比例，默认 0.95。
        min_volume: 最小交易单位（手），默认 100。
        open_rate: 买入佣金率，默认 0.05%。
        close_rate: 卖出佣金率（含印花税），默认 0.15%。
        min_commission: 最低佣金金额（元），默认 5。
        price_add: 限价单价格附加额（元），默认 0.05。
    """

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
        """策略初始化回调，创建持仓天数计数器并写入初始化日志。"""
        self.holding_days: defaultdict = defaultdict(int)

        self.write_log("Strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        """成交回调：卖出成交时清除对应标的的持仓计时。

        Args:
            trade: 成交数据对象，含方向（Direction.LONG/SHORT）与 vt_symbol。
        """
        if trade.direction == Direction.SHORT:
            self.holding_days.pop(trade.vt_symbol, None)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K 线切片回调，执行一次完整的选股→卖出→买入→下单流程。

        每个交易日调用一次，流程如下：
        1. 取最新信号并按信号值降序排列；
        2. 对已持仓标的递增持仓天数计数；
        3. 确定需卖出的标的（末尾排名/不在成分池），跳过持仓不足 min_days 的；
        4. 按剩余现金均仓计算买入量（取整到 min_volume），调用 set_target 设置目标仓位；
        5. 最后调用 execute_trading 提交委托。

        Args:
            bars: {vt_symbol: BarData} 字典，包含本切片所有标的的最新 K 线。
        """
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
