"""
Shared backtesting engine — order matching, daily mark-to-market, statistics.

Extracted from alpha/strategy/backtesting.py for cross-module reuse.
The engine accepts a BarDataLoader protocol so that both Alpha and CNN
modules can plug in their own data sources.
"""

from collections import defaultdict
from datetime import date, datetime
from copy import copy
from typing import Protocol, cast, runtime_checkable
import logging
import traceback

import numpy as np
import polars as pl
from tqdm import tqdm

from .types import Direction, Offset, OrderData, TradeData, BarData
from .strategy import BaseStrategy
from .pnl import ContractDailyResult, PortfolioDailyResult
from .oco import check_oco_trigger

logger = logging.getLogger(__name__)

# Status constants
STATUS_SUBMITTING: str = "submitting"
STATUS_NOTTRADED: str = "nottraded"
STATUS_ALLTRADED: str = "alltraded"
STATUS_CANCELLED: str = "cancelled"


@runtime_checkable
class BarDataLoader(Protocol):
    """Protocol for loading bar data — any module can implement this."""

    def load_bar_data(
        self, vt_symbol: str, interval: str, start, end
    ) -> list[BarData]: ...

    def load_contract_settings(self) -> dict: ...


def round_to(value: float, pricetick: float) -> float:
    """Round price to nearest tick"""
    return round(value / pricetick) * pricetick


class BacktestingEngine:
    """Shared strategy backtesting engine"""

    gateway_name: str = "BACKTESTING"

    def __init__(self, data_loader: BarDataLoader) -> None:
        """Constructor"""
        self.data_loader: BarDataLoader = data_loader

        self.vt_symbols: list[str] = []
        self.start: datetime
        self.end: datetime

        self.long_rates: dict[str, float] = {}
        self.short_rates: dict[str, float] = {}
        self.stamp_duties: dict[str, float] = {}   # 卖出印花税率（A股）
        self.slippages: dict[str, float] = {}      # 每笔成交不利滑点率
        self.sizes: dict[str, float] = {}
        self.priceticks: dict[str, float] = {}

        self.capital: float = 0
        self.risk_free: float = 0
        self.annual_days: int = 0

        self.strategy_class: type[BaseStrategy]
        self.strategy: BaseStrategy
        self.bars: dict[str, BarData] = {}
        self.datetime: datetime | None = None

        self.interval: str = ""
        self.history_data: dict[tuple, BarData] = {}
        self.dts: set[datetime] = set()

        self.limit_order_count: int = 0
        self.limit_orders: dict[str, OrderData] = {}
        self.active_limit_orders: dict[str, OrderData] = {}

        self.trade_count: int = 0
        self.trades: dict[str, TradeData] = {}

        self.logs: list[str] = []

        self.daily_results: dict[date, PortfolioDailyResult] = {}
        self.daily_df: pl.DataFrame

        self.pre_closes: defaultdict = defaultdict(float)

        self.cash: float = 0

        self.signal_df: pl.DataFrame = pl.DataFrame()

        # T+1：当日买入不可当日卖出（默认关闭，按需开启）
        self.t_plus1: bool = False
        self.buy_dates: dict[str, date] = {}

        # 限价单成交价口径（与训练 label 的 price_ref 对齐，杜绝回测↔label 背离）：
        #   open  = 撮合 bar 的开盘价（默认/旧行为，对齐 next_open）
        #   close = 撮合 bar 的收盘价（对齐 next_close，市价化 MOC 成交）
        #   vwap  = 撮合 bar 的均价 成交额/成交量（对齐 next_vwap，VWAP 成交）
        # close/vwap 为市价化成交：成交价取参考价本身，不受委托限价封顶；
        # 仍保留涨跌停封板保护与 T+1 卖出限制。
        self.fill_price_mode: str = "open"

    def set_parameters(
        self,
        vt_symbols: list[str],
        interval: str,
        start: datetime,
        end: datetime,
        capital: int = 1_000_000,
        risk_free: float = 0,
        annual_days: int = 240
    ) -> None:
        """Set parameters"""
        self.vt_symbols = vt_symbols
        self.interval = interval

        self.start = start
        self.end = end
        self.capital = capital
        self.risk_free = risk_free
        self.annual_days = annual_days

        self.cash = capital

        contract_settings: dict = self.data_loader.load_contract_settings()
        for vt_symbol in vt_symbols:
            setting: dict | None = contract_settings.get(vt_symbol, None)
            if not setting:
                logger.warning(f"找不到合约{vt_symbol}的交易配置，请检查！")
                continue

            self.long_rates[vt_symbol] = setting["long_rate"]
            self.short_rates[vt_symbol] = setting["short_rate"]
            self.stamp_duties[vt_symbol] = setting.get("stamp_duty", 0.0)
            self.slippages[vt_symbol] = setting.get("slippage", 0.0)
            self.sizes[vt_symbol] = setting["size"]
            self.priceticks[vt_symbol] = setting["pricetick"]

    def add_strategy(self, strategy_class: type, setting: dict, signal_df: pl.DataFrame) -> None:
        """Add strategy"""
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, copy(self.vt_symbols), setting
        )
        self.signal_df = signal_df

    def load_data(self) -> None:
        """Load historical data"""
        logger.info("开始加载历史数据")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            logger.info("起始日期必须小于结束日期")
            return

        self.history_data.clear()
        self.dts.clear()

        empty_symbols: list[str] = []
        for vt_symbol in tqdm(self.vt_symbols, total=len(self.vt_symbols)):
            data: list[BarData] = self.data_loader.load_bar_data(
                vt_symbol,
                self.interval,
                self.start,
                self.end
            )

            for bar in data:
                self.dts.add(bar.datetime)
                self.history_data[(bar.datetime, vt_symbol)] = bar

            data_count = len(data)
            if not data_count:
                empty_symbols.append(vt_symbol)

        if empty_symbols:
            logger.info(f"部分合约历史数据为空：{empty_symbols}")

        logger.info("所有历史数据加载完成")

    def run_backtesting(self) -> None:
        """Start backtesting"""
        self.strategy.on_init()
        logger.info("策略初始化完成")

        dts: list = list(self.dts)
        dts.sort()

        logger.info("开始回放历史数据")
        for dt in dts:
            try:
                self.new_bars(dt)
            except Exception:
                logger.info("触发异常，回测终止")
                logger.info(traceback.format_exc())
                return

        logger.info("历史数据回放结束")

    def calculate_result(self) -> pl.DataFrame | None:
        """Calculate daily results"""
        logger.info("开始计算逐日盯市结果")

        if not self.daily_results:
            logger.info("无每日结果，无法计算")
            return None

        for trade in self.trades.values():
            d: date = trade.datetime.date()
            daily_result: PortfolioDailyResult | None = self.daily_results.get(d, None)
            if daily_result:
                daily_result.add_trade(trade)

        pre_closes: dict[str, float] = {}
        start_poses: dict[str, float] = {}

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_closes,
                start_poses,
                self.sizes,
                self.long_rates,
                self.short_rates,
                self.stamp_duties,
            )

            pre_closes = daily_result.close_prices
            start_poses = daily_result.end_poses

        results: list[dict] = []
        for daily_result in self.daily_results.values():
            results.append({
                "date": daily_result.date,
                "trade_count": daily_result.trade_count,
                "turnover": daily_result.turnover,
                "commission": daily_result.commission,
                "trading_pnl": daily_result.trading_pnl,
                "holding_pnl": daily_result.holding_pnl,
                "total_pnl": daily_result.total_pnl,
                "net_pnl": daily_result.net_pnl
            })

        self.daily_df = pl.DataFrame(results)

        logger.info("逐日盯市计算完成")
        return self.daily_df

    def calculate_statistics(self) -> dict:
        """Calculate strategy statistics"""
        logger.info("开始计算策略统计指标")

        start_date: str = ""
        end_date: str = ""
        total_days: int = 0
        profit_days: int = 0
        loss_days: int = 0
        end_balance: float = 0
        max_drawdown: float = 0
        max_ddpercent: float = 0
        max_drawdown_duration: int = 0
        total_net_pnl: float = 0
        daily_net_pnl: float = 0
        total_commission: float = 0
        daily_commission: float = 0
        total_turnover: float = 0
        daily_turnover: float = 0
        total_trade_count: int = 0
        daily_trade_count: float = 0
        total_return: float = 0
        annual_return: float = 0
        daily_return: float = 0
        return_std: float = 0
        sharpe_ratio: float = 0
        return_drawdown_ratio: float = 0

        positive_balance: bool = False

        df: pl.DataFrame = self.daily_df

        if df is not None:
            df = df.with_columns(
                balance=pl.col("net_pnl").cum_sum() + self.capital
            ).with_columns(
                pl.col("balance").pct_change().fill_null(0).alias("return"),
                highlevel=pl.col("balance").cum_max()
            ).with_columns(
                drawdown=pl.col("balance") - pl.col("highlevel"),
                ddpercent=(pl.col("balance") / pl.col("highlevel") - 1) * 100
            )

            positive_balance = (df["balance"] > 0).all()
            if not positive_balance:
                logger.info("回测中出现爆仓（资金小于等于0），无法计算策略统计指标")

            self.daily_df = df

        if positive_balance:
            start_date = df["date"][0]
            end_date = df["date"][-1]

            total_days = len(df)
            profit_days = df.filter(pl.col("net_pnl") > 0).height
            loss_days = df.filter(pl.col("net_pnl") < 0).height

            end_balance = df["balance"][-1]
            max_drawdown = cast(float, df["drawdown"].min())
            max_ddpercent = cast(float, df["ddpercent"].min())

            max_drawdown_end_idx = cast(int, df["drawdown"].arg_min())
            max_drawdown_end = df["date"][max_drawdown_end_idx]

            if isinstance(max_drawdown_end, date):
                max_drawdown_start_idx = cast(int, df.slice(0, max_drawdown_end_idx + 1)["balance"].arg_max())
                max_drawdown_start = df["date"][max_drawdown_start_idx]
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration = 0

            total_net_pnl = df["net_pnl"].sum()
            daily_net_pnl = total_net_pnl / total_days

            total_commission = df["commission"].sum()
            daily_commission = total_commission / total_days

            total_turnover = df["turnover"].sum()
            daily_turnover = total_turnover / total_days

            total_trade_count = cast(int, df["trade_count"].sum())
            daily_trade_count = total_trade_count / total_days

            total_return = (end_balance / self.capital - 1) * 100
            annual_return = total_return / total_days * self.annual_days
            daily_return = cast(float, df["return"].mean()) * 100
            return_std = cast(float, df["return"].std()) * 100

            if return_std:
                daily_risk_free = self.risk_free / np.sqrt(self.annual_days)
                sharpe_ratio = (daily_return - daily_risk_free) / return_std * np.sqrt(self.annual_days)
            else:
                sharpe_ratio = 0

            # 零回撤（如单调上行的合成/样本外区间）时分母为 0，安全返回 0，
            # 避免成本压测 / walk-forward 批量回测时触发 ZeroDivisionError
            return_drawdown_ratio = (-total_net_pnl / max_drawdown) if max_drawdown else 0.0

        logger.info("-" * 30)
        logger.info(f"首个交易日：  {start_date}")
        logger.info(f"最后交易日：  {end_date}")

        logger.info(f"总交易日：  {total_days}")
        logger.info(f"盈利交易日：  {profit_days}")
        logger.info(f"亏损交易日：  {loss_days}")

        logger.info(f"起始资金：  {self.capital:,.2f}")
        logger.info(f"结束资金：  {end_balance:,.2f}")

        logger.info(f"总收益率：  {total_return:,.2f}%")
        logger.info(f"年化收益：  {annual_return:,.2f}%")
        logger.info(f"最大回撤:   {max_drawdown:,.2f}")
        logger.info(f"百分比最大回撤: {max_ddpercent:,.2f}%")
        logger.info(f"最长回撤天数:   {max_drawdown_duration}")

        logger.info(f"总盈亏：  {total_net_pnl:,.2f}")
        logger.info(f"总手续费：  {total_commission:,.2f}")
        logger.info(f"总成交金额：  {total_turnover:,.2f}")
        logger.info(f"总成交笔数：  {total_trade_count}")

        logger.info(f"日均盈亏：  {daily_net_pnl:,.2f}")
        logger.info(f"日均手续费：  {daily_commission:,.2f}")
        logger.info(f"日均成交金额：  {daily_turnover:,.2f}")
        logger.info(f"日均成交笔数：  {daily_trade_count}")

        logger.info(f"日均收益率：  {daily_return:,.2f}%")
        logger.info(f"收益标准差：  {return_std:,.2f}%")
        logger.info(f"Sharpe Ratio：  {sharpe_ratio:,.2f}")
        logger.info(f"收益回撤比：  {return_drawdown_ratio:,.2f}")

        statistics: dict = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        for key, value in statistics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                if value in (np.inf, -np.inf):
                    value = 0
                value = np.nan_to_num(value)
                if isinstance(value, np.generic):
                    value = value.item()
            statistics[key] = value

        logger.info("策略统计指标计算完成")
        return statistics

    def update_daily_close(self, bars: dict[str, BarData], dt: datetime) -> None:
        """Update daily closing price"""
        d: date = dt.date()

        close_prices: dict[str, float] = {}
        for bar in bars.values():
            if not bar.close_price:
                close_prices[bar.vt_symbol] = self.pre_closes[bar.vt_symbol]
            else:
                close_prices[bar.vt_symbol] = bar.close_price

        daily_result: PortfolioDailyResult | None = self.daily_results.get(d, None)

        if daily_result:
            daily_result.update_close_prices(close_prices)
        else:
            self.daily_results[d] = PortfolioDailyResult(d, close_prices)

    def new_bars(self, dt: datetime) -> None:
        """Push historical data"""
        self.datetime = dt

        bars: dict[str, BarData] = {}
        for vt_symbol in self.vt_symbols:
            last_bar = self.bars.get(vt_symbol, None)
            if last_bar:
                if last_bar.close_price:
                    self.pre_closes[vt_symbol] = last_bar.close_price

            bar: BarData | None = self.history_data.get((dt, vt_symbol), None)

            if bar:
                self.bars[vt_symbol] = bar
                bars[vt_symbol] = bar
            elif vt_symbol in self.bars:
                old_bar: BarData = self.bars[vt_symbol]

                fill_bar: BarData = BarData(
                    symbol=old_bar.symbol,
                    exchange=old_bar.exchange,
                    datetime=dt,
                    interval=old_bar.interval,
                    open_price=old_bar.close_price,
                    high_price=old_bar.close_price,
                    low_price=old_bar.close_price,
                    close_price=old_bar.close_price,
                )
                self.bars[vt_symbol] = fill_bar

        self.cross_order()
        self.strategy.on_bars(bars)

        self.update_daily_close(self.bars, dt)

    def _bar_reference_price(self, bar: BarData) -> float:
        """按 fill_price_mode 返回撮合参考价：open / close / vwap(成交额/成交量)。

        vwap 在量为 0 或无成交额时回退到收盘价，与 dataset 的 next_vwap 口径一致。
        """
        mode = self.fill_price_mode
        if mode == "close":
            return bar.close_price
        if mode == "vwap":
            volume = float(getattr(bar, "volume", 0.0) or 0.0)
            turnover = float(getattr(bar, "turnover", 0.0) or 0.0)
            if volume > 0 and turnover > 0:
                return turnover / volume
            return bar.close_price
        return bar.open_price

    def cross_order(self) -> None:
        """Match orders（先按 OCO 分组撮合止盈止损，再撮合普通限价单）"""
        # OCO 止盈止损括号单优先撮合（含「止损先到」保守假设）
        self._cross_oco_orders()

        market_fill = self.fill_price_mode in ("close", "vwap")

        for order in list(self.active_limit_orders.values()):
            # OCO 腿由 _cross_oco_orders 专门处理，普通循环跳过
            if order.oco_group is not None:
                continue

            bar: BarData = self.bars[order.vt_symbol]

            long_cross_price: float = bar.low_price
            short_cross_price: float = bar.high_price
            long_best_price: float = bar.open_price
            short_best_price: float = bar.open_price

            if order.status == STATUS_SUBMITTING:
                order.status = STATUS_NOTTRADED
                self.strategy.update_order(order)

            pricetick: float = self.priceticks[order.vt_symbol]
            pre_close: float = self.pre_closes.get(order.vt_symbol, 0)

            limit_up: float = round_to(pre_close * 1.1, pricetick)
            limit_down: float = round_to(pre_close * 0.9, pricetick)

            if market_fill:
                # 市价化成交（close/vwap）：以参考价成交，委托限价不封顶（对齐 label）。
                # 撮合门槛仅保留「当根有成交 + 未涨/跌停封板」，与实盘 MOC/VWAP 委托一致。
                ref_price: float = self._bar_reference_price(bar)
                long_cross = (
                    order.direction == Direction.LONG
                    and ref_price > 0
                    and bar.low_price > 0
                    and bar.low_price < limit_up
                )
                short_cross = (
                    order.direction == Direction.SHORT
                    and ref_price > 0
                    and bar.high_price > 0
                    and bar.high_price > limit_down
                )
            else:
                long_cross = (
                    order.direction == Direction.LONG
                    and order.price >= long_cross_price
                    and long_cross_price > 0
                    and bar.low_price < limit_up
                )
                short_cross = (
                    order.direction == Direction.SHORT
                    and order.price <= short_cross_price
                    and short_cross_price > 0
                    and bar.high_price > limit_down
                )

            # T+1：当日买入的持仓不可当日卖出，跳过本根撮合（订单保留，次日继续尝试）
            if (
                short_cross
                and self.t_plus1
                and self.datetime is not None
                and self.buy_dates.get(order.vt_symbol) == self.datetime.date()
            ):
                continue

            if not long_cross and not short_cross:
                continue

            if market_fill:
                # 市价化：买卖均按参考价成交（滑点在 _settle_fill 内统一叠加）
                raw_price = self._bar_reference_price(bar)
            elif long_cross:
                # 买入成交价取 min(委托价, 开盘价)（滑点在 _settle_fill 内统一叠加）
                raw_price = min(order.price, long_best_price)
            else:
                # 卖出成交价取 max(委托价, 开盘价)
                raw_price = max(order.price, short_best_price)

            self._settle_fill(order, raw_price)

    def _cross_oco_orders(self) -> None:
        """撮合 OCO 止盈止损括号单。

        红线：同一根 bar 同时触发止盈与止损、先后未知时，**保守假设止损先到**
        （见 backtest/oco.check_oco_trigger）。一腿成交即撤销另一腿。
        """
        # 按 oco_group 归集尚在挂单中的腿
        groups: dict[str, list[OrderData]] = {}
        for order in self.active_limit_orders.values():
            if order.oco_group is not None:
                groups.setdefault(order.oco_group, []).append(order)

        for legs in groups.values():
            bar: BarData = self.bars[legs[0].vt_symbol]

            for o in legs:
                if o.status == STATUS_SUBMITTING:
                    o.status = STATUS_NOTTRADED
                    self.strategy.update_order(o)

            tp_leg = next((o for o in legs if o.order_type == "limit"), None)
            sl_leg = next((o for o in legs if o.order_type == "stop"), None)
            if tp_leg is None or sl_leg is None:
                continue

            # T+1：当日买入不可当日卖出，本根跳过（OCO 腿保留待次日）
            if (
                self.t_plus1
                and self.datetime is not None
                and self.buy_dates.get(bar.vt_symbol) == self.datetime.date()
            ):
                continue

            # 跌停封死（最高价仍 ≤ 跌停价）时卖不出，与主撮合 short 口径一致
            pricetick: float = self.priceticks[bar.vt_symbol]
            pre_close: float = self.pre_closes.get(bar.vt_symbol, 0)
            limit_down: float = round_to(pre_close * 0.9, pricetick)
            if pre_close and bar.high_price <= limit_down:
                continue

            trig = check_oco_trigger(
                bar.open_price, bar.high_price, bar.low_price,
                tp_leg.price, sl_leg.price,
            )
            if trig is None:
                continue

            reason, fill_price = trig
            winner = sl_leg if reason == "sl" else tp_leg
            loser = tp_leg if reason == "sl" else sl_leg

            self._settle_fill(winner, fill_price)
            # 一腿成交，撤销另一腿（OCO 语义）
            self.cancel_order(self.strategy, loser.vt_orderid)

    def _settle_fill(self, order: OrderData, raw_price: float) -> TradeData:
        """成交结算：叠加滑点、记录成交、更新现金与持仓。买卖共用，杜绝两处各算。"""
        pricetick: float = self.priceticks[order.vt_symbol]
        slippage: float = self.slippages.get(order.vt_symbol, 0.0)

        if order.direction == Direction.LONG:
            # 买入叠加不利滑点（买更贵）
            trade_price = round_to(raw_price * (1 + slippage), pricetick)
        else:
            # 卖出叠加不利滑点（卖更便宜）
            trade_price = round_to(raw_price * (1 - slippage), pricetick)

        order.traded = order.volume
        order.status = STATUS_ALLTRADED
        self.strategy.update_order(order)

        if order.vt_orderid in self.active_limit_orders:
            self.active_limit_orders.pop(order.vt_orderid)

        self.trade_count += 1

        trade: TradeData = TradeData(
            symbol=order.symbol,
            exchange=order.exchange,
            orderid=order.orderid,
            tradeid=str(self.trade_count),
            direction=order.direction,
            offset=order.offset,
            price=trade_price,
            volume=order.volume,
            datetime=self.datetime,
            gateway_name=self.gateway_name,
        )

        size: float = self.sizes[trade.vt_symbol]
        trade_turnover: float = trade.price * trade.volume * size

        if trade.direction == Direction.LONG:
            trade_commission: float = trade_turnover * self.long_rates[trade.vt_symbol]
            trade_stamp: float = 0.0
            self.cash -= trade_turnover
        else:
            trade_commission = trade_turnover * self.short_rates[trade.vt_symbol]
            # 印花税仅卖出收取（A股现行规则）
            trade_stamp = trade_turnover * self.stamp_duties.get(trade.vt_symbol, 0.0)
            self.cash += trade_turnover

        self.cash -= trade_commission + trade_stamp

        self.strategy.update_trade(trade)
        self.trades[trade.vt_tradeid] = trade

        # 记录建仓成交日，供 T+1 卖出限制判定
        if trade.direction == Direction.LONG and self.datetime is not None:
            self.buy_dates[trade.vt_symbol] = self.datetime.date()

        return trade

    def sell_to_close_intrabar(self, vt_symbol: str, price: float, volume: float) -> None:
        """当根 bar 内按给定价直接平多（卖出），用于止盈/止损等路径依赖出场。

        区别于限价单的「下一根撮合」：止盈/止损是触发价当根成交，这里直接结算一笔
        SHORT/CLOSE 成交（含佣金、卖出印花税、现金更新），保守由调用方决定触发价。
        """
        if volume <= 0:
            return
        pricetick: float = self.priceticks[vt_symbol]
        trade_price: float = round_to(price, pricetick)
        symbol, exchange = vt_symbol.rsplit(".", 1)

        self.trade_count += 1
        trade: TradeData = TradeData(
            symbol=symbol,
            exchange=exchange,
            orderid=f"oco{self.trade_count}",
            tradeid=str(self.trade_count),
            direction=Direction.SHORT,
            offset=Offset.CLOSE,
            price=trade_price,
            volume=volume,
            datetime=self.datetime,
            gateway_name=self.gateway_name,
        )

        size: float = self.sizes[vt_symbol]
        turnover: float = trade_price * volume * size
        commission: float = turnover * self.short_rates[vt_symbol]
        stamp: float = turnover * self.stamp_duties.get(vt_symbol, 0.0)

        self.cash += turnover
        self.cash -= commission + stamp

        self.strategy.update_trade(trade)
        self.trades[trade.vt_tradeid] = trade

    def get_signal(self) -> pl.DataFrame:
        """Get model prediction signal for current time"""
        if not self.datetime:
            self.write_log("尚未开始数据回放，无法加载模型预测值")
            return pl.DataFrame()

        dt: datetime = self.datetime.replace(tzinfo=None)
        signal: pl.DataFrame = self.signal_df.filter(pl.col("datetime") == dt)

        if signal.is_empty():
            self.write_log(f"找不到{dt}对应的信号模型预测值")

        return signal

    def send_order(
        self,
        strategy: BaseStrategy,
        vt_symbol: str,
        direction: str,
        offset: str,
        price: float,
        volume: float,
    ) -> list[str]:
        """Send order"""
        price = round_to(price, self.priceticks[vt_symbol])
        symbol, exchange = vt_symbol.rsplit(".", 1)

        self.limit_order_count += 1

        order: OrderData = OrderData(
            symbol=symbol,
            exchange=exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=STATUS_SUBMITTING,
            datetime=self.datetime,
            gateway_name=self.gateway_name,
        )

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order

        return [order.vt_orderid]

    def send_oco(
        self,
        strategy: BaseStrategy,
        vt_symbol: str,
        tp_price: float,
        sl_price: float,
        volume: float,
    ) -> list[str]:
        """挂出一组 OCO 止盈止损卖出括号单（止盈=限价腿、止损=触发腿）。

        两腿共享同一 oco_group，由 _cross_oco_orders 撮合，一腿成交即撤另一腿。
        返回 [止盈腿 id, 止损腿 id]。
        """
        pricetick: float = self.priceticks[vt_symbol]
        tp_price = round_to(tp_price, pricetick)
        sl_price = round_to(sl_price, pricetick)
        symbol, exchange = vt_symbol.rsplit(".", 1)

        group_id = f"OCO.{self.limit_order_count + 1}"
        order_ids: list[str] = []
        for price, order_type in ((tp_price, "limit"), (sl_price, "stop")):
            self.limit_order_count += 1
            order: OrderData = OrderData(
                symbol=symbol,
                exchange=exchange,
                orderid=str(self.limit_order_count),
                direction=Direction.SHORT,
                offset=Offset.CLOSE,
                price=price,
                volume=volume,
                status=STATUS_SUBMITTING,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
                order_type=order_type,
                oco_group=group_id,
            )
            self.active_limit_orders[order.vt_orderid] = order
            self.limit_orders[order.vt_orderid] = order
            order_ids.append(order.vt_orderid)

        return order_ids

    def cancel_order(self, strategy: BaseStrategy, vt_orderid: str) -> None:
        """Cancel order"""
        if vt_orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(vt_orderid)

        order.status = STATUS_CANCELLED
        self.strategy.update_order(order)

    def write_log(self, msg: str, strategy: BaseStrategy | None = None) -> None:
        """Output log message"""
        msg = f"{self.datetime}  {msg}"
        self.logs.append(msg)

    def get_all_trades(self) -> list[TradeData]:
        """Get all trade information"""
        return list(self.trades.values())

    def get_all_orders(self) -> list[OrderData]:
        """Get all order information"""
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> list[PortfolioDailyResult]:
        """Get all daily profit and loss information"""
        return list(self.daily_results.values())

    def get_cash_available(self) -> float:
        """Get current available cash"""
        return self.cash

    def get_holding_value(self) -> float:
        """Get current holding market value"""
        holding_value: float = 0

        for vt_symbol, pos in self.strategy.pos_data.items():
            bar: BarData = self.bars[vt_symbol]
            size: float = self.sizes[vt_symbol]

            holding_value += bar.close_price * pos * size

        return holding_value
