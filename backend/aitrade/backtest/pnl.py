"""
Shared PnL calculation — ContractDailyResult, PortfolioDailyResult.

Extracted from alpha/strategy/backtesting.py for cross-module reuse.
"""

from datetime import date

from .types import Direction, TradeData


class ContractDailyResult:
    """单合约单日盯市损益结果，汇聚当日所有成交并计算交易/持有 PnL。"""

    def __init__(self, result_date: date, close_price: float) -> None:
        """初始化单日合约结果。

        Args:
            result_date: 该记录对应的交易日。
            close_price: 该合约当日收盘价（元），用于估算持仓盈亏。
        """
        self.date: date = result_date
        self.close_price: float = close_price
        self.pre_close: float = 0

        self.trades: list[TradeData] = []
        self.trade_count: int = 0

        self.start_pos: float = 0
        self.end_pos: float = 0

        self.turnover: float = 0
        self.commission: float = 0

        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """添加一笔当日成交记录。

        Args:
            trade: 待加入的 TradeData 成交对象。
        """
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: float,
        long_rate: float,
        short_rate: float,
        stamp_rate: float = 0.0
    ) -> None:
        """计算当日交易盈亏、持仓盈亏与净盈亏。

        计算公式（与 vnpy 标准口径对齐）：
        - 持仓盈亏 = start_pos × (close - pre_close) × size；
        - 每笔成交：trading_pnl += pos_change × (close - price) × size；
          turnover += price × volume × size；
          commission += turnover × rate + stamp（仅卖出收印花税）；
        - total_pnl = trading_pnl + holding_pnl；net_pnl = total_pnl - commission。

        Args:
            pre_close: 前一交易日收盘价（元）；为 0 时持仓盈亏为 0（首日无前收盘）。
            start_pos: 日初持仓量（手/股）。
            size: 合约乘数（A 股为 1）。
            long_rate: 买入佣金率（如 0.0003）。
            short_rate: 卖出佣金率（如 0.0003）。
            stamp_rate: 卖出印花税率（A 股为 0.0005，默认 0）。
        """
        if pre_close:
            self.pre_close = pre_close

        self.start_pos = start_pos
        self.end_pos = start_pos

        self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size

        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change: float = trade.volume
                rate: float = long_rate
                stamp: float = 0.0
            else:
                pos_change = -trade.volume
                rate = short_rate

            self.end_pos += pos_change

            turnover: float = trade.volume * size * trade.price
            # 印花税仅卖出收取；与佣金一起计入交易成本（commission 字段）
            stamp = turnover * stamp_rate if trade.direction != Direction.LONG else 0.0

            self.trading_pnl += pos_change * (self.close_price - trade.price) * size
            self.turnover += turnover
            self.commission += turnover * rate + stamp

        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission

    def update_close_price(self, close_price: float) -> None:
        """更新当日收盘价（用于停牌前向填充等场景）。

        Args:
            close_price: 新的收盘价（元）。
        """
        self.close_price = close_price


class PortfolioDailyResult:
    """多合约组合单日盯市结果，聚合所有合约的当日 ContractDailyResult。"""

    def __init__(self, result_date: date, close_prices: dict[str, float]) -> None:
        """初始化组合日结果，并为每个合约创建 ContractDailyResult。

        Args:
            result_date: 该记录对应的交易日。
            close_prices: 各合约当日收盘价字典，key 为 vt_symbol。
        """
        self.date: date = result_date
        self.close_prices: dict[str, float] = close_prices
        self.pre_closes: dict[str, float] = {}
        self.start_poses: dict[str, float] = {}
        self.end_poses: dict[str, float] = {}

        self.contract_results: dict[str, ContractDailyResult] = {}

        for vt_symbol, close_price in close_prices.items():
            self.contract_results[vt_symbol] = ContractDailyResult(result_date, close_price)

        self.trade_count: int = 0
        self.turnover: float = 0
        self.commission: float = 0
        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """将成交记录分发到对应合约的 ContractDailyResult。

        Args:
            trade: 当日发生的 TradeData 成交对象。
        """
        contract_result: ContractDailyResult = self.contract_results[trade.vt_symbol]
        contract_result.add_trade(trade)

    def calculate_pnl(
        self,
        pre_closes: dict[str, float],
        start_poses: dict[str, float],
        sizes: dict[str, float],
        long_rates: dict[str, float],
        short_rates: dict[str, float],
        stamp_rates: dict[str, float] | None = None
    ) -> None:
        """驱动所有合约的 ContractDailyResult.calculate_pnl()，并汇聚组合级数据。

        汇聚后的组合 trade_count / turnover / commission / pnl 字段为各合约之和。
        期末持仓 end_poses 将用于下一交易日的 start_poses。

        Args:
            pre_closes: 各合约前收盘价字典（key 为 vt_symbol）。
            start_poses: 各合约日初持仓字典（key 为 vt_symbol）。
            sizes: 各合约乘数字典。
            long_rates: 各合约买入佣金率字典。
            short_rates: 各合约卖出佣金率字典。
            stamp_rates: 各合约印花税率字典；None 时等同空字典（税率均为 0）。
        """
        self.pre_closes = pre_closes
        self.start_poses = start_poses
        stamp_rates = stamp_rates or {}

        for vt_symbol, contract_result in self.contract_results.items():
            contract_result.calculate_pnl(
                pre_closes.get(vt_symbol, 0),
                start_poses.get(vt_symbol, 0),
                sizes[vt_symbol],
                long_rates[vt_symbol],
                short_rates[vt_symbol],
                stamp_rates.get(vt_symbol, 0.0)
            )

            self.trade_count += contract_result.trade_count
            self.turnover += contract_result.turnover
            self.commission += contract_result.commission
            self.trading_pnl += contract_result.trading_pnl
            self.holding_pnl += contract_result.holding_pnl
            self.total_pnl += contract_result.total_pnl
            self.net_pnl += contract_result.net_pnl

            self.end_poses[vt_symbol] = contract_result.end_pos

    def update_close_prices(self, close_prices: dict[str, float]) -> None:
        """更新组合中各合约的收盘价（含新增合约）。

        对 close_prices 中每个 vt_symbol：
        - 若合约结果已存在，更新其 close_price；
        - 若为新合约（如首次出现），自动创建 ContractDailyResult。

        Args:
            close_prices: 各合约最新收盘价字典，key 为 vt_symbol。
        """
        self.close_prices.update(close_prices)

        for vt_symbol, close_price in close_prices.items():
            contract_result: ContractDailyResult | None = self.contract_results.get(vt_symbol, None)
            if contract_result:
                contract_result.update_close_price(close_price)
            else:
                self.contract_results[vt_symbol] = ContractDailyResult(self.date, close_price)
