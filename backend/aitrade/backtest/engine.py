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
from .pnl import PortfolioDailyResult
from .oco import check_oco_trigger
from .instrument import infer_limit_ratio, infer_t_plus1
from .t1 import is_t1_locked

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
    """将价格对齐到最小价格跳动（pricetick）的整数倍。

    Args:
        value: 待对齐的原始价格或价格增量。
        pricetick: 品种最小价格变动单位，如 0.01（A 股）。

    Returns:
        经四舍五入后距离 value 最近的 pricetick 整数倍价格。

    Example:
        >>> round_to(10.234, 0.01)
        10.23
    """
    return round(value / pricetick) * pricetick


class BacktestingEngine:
    """Shared strategy backtesting engine"""

    gateway_name: str = "BACKTESTING"

    def __init__(self, data_loader: BarDataLoader) -> None:
        """初始化回测引擎，绑定数据加载器并清空内部状态。

        Args:
            data_loader: 实现 BarDataLoader 协议的数据加载对象，负责提供历史 K 线
                与合约配置；Alpha/CNN 模块可各自注入不同实现。
        """
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
        # 涨跌停比例：None 表示无限制（转债），float 表示单边比例（如 0.1=10%）。
        # 由 infer_limit_ratio 自动推断，contract_settings 中的 limit_ratio 优先覆盖。
        self.limit_ratios: dict[str, float | None] = {}

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
        # T+1 豁免标的：contract setting 中 t_plus1=false 的品种（如可转债）。
        # 豁免标的即使全局 t_plus1=True 也允许当日买当日卖。
        # 注：全局 t_plus1=False 时本集合不起效果（_t_plus1_locked 短路返回 False）。
        # 注：contract setting t_plus1=true 且全局 t_plus1=False 的组合不作处理（YAGNI）。
        self.t_plus1_exempt: set[str] = set()

        # 停牌无量门槛：load_data 时统计至少有一根 volume>0 bar 的标的。
        # 撮合时若标的在此集合内且当根 bar.volume<=0，视为停牌/合成 bar，跳过撮合。
        # 从未出现过 volume>0 的标的（如 parquet 缺 volume 列兜底 0.0）不加入，
        # 豁免门槛，避免此类数据全程零成交且无任何报错。
        self.volume_supported: set[str] = set()

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
        """设置回测全局参数：标的列表、时间范围、资金与成本配置。

        从 data_loader 加载合约配置（contract.json），自动填充各标的的佣金率、
        滑点、涨跌停比例与 T+1 豁免集合；未找到合约配置时按品种规则推断并告警。

        Args:
            vt_symbols: 回测标的列表，如 ``["000001.SZSE", "600000.SSE"]``。
            interval: K 线周期，``"d"`` 为日线，``"1m"``/``"30m"`` 为分钟线。
            start: 回测起始时间（含）。
            end: 回测截止时间（含）。
            capital: 初始资金（元），默认 1,000,000。
            risk_free: 无风险年化利率（小数），用于 Sharpe Ratio 计算，默认 0。
            annual_days: 年化交易日数，默认 240（A 股）。
        """
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
                # 无合约配置时仍填充推断的涨跌停比例，避免撮合时缺键
                if vt_symbol not in self.limit_ratios:
                    self.limit_ratios[vt_symbol] = infer_limit_ratio(vt_symbol)
                # 无合约配置时按品种推断 T+1 豁免（转债 T+0）
                if not infer_t_plus1(vt_symbol):
                    self.t_plus1_exempt.add(vt_symbol)
                continue

            self.long_rates[vt_symbol] = setting["long_rate"]
            self.short_rates[vt_symbol] = setting["short_rate"]
            self.stamp_duties[vt_symbol] = setting.get("stamp_duty", 0.0)
            self.slippages[vt_symbol] = setting.get("slippage", 0.0)
            self.sizes[vt_symbol] = setting["size"]
            self.priceticks[vt_symbol] = setting["pricetick"]
            # limit_ratio：contract.json 显式配置优先，否则按品种推断
            if vt_symbol not in self.limit_ratios:
                self.limit_ratios[vt_symbol] = setting.get(
                    "limit_ratio", infer_limit_ratio(vt_symbol)
                )
            # t_plus1 豁免：contract setting 显式配置优先；缺失时按品种推断（转债 T+0）。
            # setting.get("t_plus1", infer_t_plus1(vt_symbol)) is False 判定豁免：
            #   - setting["t_plus1"]=False        → 显式配置豁免
            #   - 键缺失 + infer_t_plus1=False    → 推断豁免（如转债无合约配置）
            if setting.get("t_plus1", infer_t_plus1(vt_symbol)) is False:
                self.t_plus1_exempt.add(vt_symbol)

    def add_strategy(self, strategy_class: type, setting: dict, signal_df: pl.DataFrame) -> None:
        """实例化策略并注入信号 DataFrame。

        Args:
            strategy_class: BaseStrategy 子类，将以 self 为引擎、setting 为参数被实例化。
            setting: 策略参数字典，注入策略实例的同名属性（如 ``exit_mode``/``hold_days``）。
            signal_df: 外部预计算的模型信号表，列至少含 ``[datetime, vt_symbol, signal]``；
                策略通过 ``get_signal()`` 在回放时按当前时间戳检索对应行。
        """
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, copy(self.vt_symbols), setting
        )
        self.signal_df = signal_df

    def load_data(self) -> None:
        """加载所有标的的历史 K 线到内存，并初始化成交量支持集合。

        按 vt_symbols 逐只调用 data_loader.load_bar_data，将每根 bar 按
        (datetime, vt_symbol) 键写入 self.history_data；同时收集所有出现过的 datetime
        到 self.dts（用于 run_backtesting 的时间轴排序）。

        副作用：
        - 对任何历史区间内出现过 volume > 0 的标的，加入 self.volume_supported；
          全程无量的标的自动豁免停牌不成交门槛并写告警日志。
        - 无数据的标的记入告警。
        """
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
                # 只要有一根 bar 的成交量大于 0，即认为该标的数据含有量信息
                if bar.volume > 0:
                    self.volume_supported.add(vt_symbol)

            data_count = len(data)
            if not data_count:
                empty_symbols.append(vt_symbol)

        if empty_symbols:
            logger.info(f"部分合约历史数据为空：{empty_symbols}")

        # 有 bar 数据但从未出现 volume>0 的标的：自动豁免停牌不成交门槛，并告警
        symbols_with_bars = {vt_symbol for vt_symbol in self.vt_symbols
                             if any((bar_dt, vt_symbol) in self.history_data
                                    for bar_dt in self.dts)}
        no_volume_symbols = symbols_with_bars - self.volume_supported
        if no_volume_symbols:
            logger.warning(
                f"以下标的全程无 volume 信息，已豁免停牌不成交门槛：{sorted(no_volume_symbols)}"
            )

        logger.info("所有历史数据加载完成")

    def run_backtesting(self) -> None:
        """按时间顺序回放历史 K 线，驱动策略回调与订单撮合。

        先调用 strategy.on_init()，再对 dts 中所有时间戳升序调用 new_bars()。
        任何异常（含策略层抛出的）都会中止回测并打印 traceback。
        本方法不返回值；结果存于 self.trades / self.daily_results 供后续统计使用。
        """
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
        """汇总逐日盯市损益，返回每日统计 DataFrame。

        将 self.trades 中的每笔成交分配到对应交易日，依次调用各日
        PortfolioDailyResult.calculate_pnl()（前收盘价与期末持仓向后传递），
        最终汇总为逐日 DataFrame 写入 self.daily_df。

        Returns:
            列为 [date, trade_count, turnover, commission, trading_pnl, holding_pnl,
            total_pnl, net_pnl] 的 polars DataFrame；无任何交易日结果时返回 None。
        """
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
        """在 daily_df 基础上计算全套策略统计指标并写回 self.daily_df（补充净值列）。

        在 calculate_result() 之后调用。计算流程：
        1. 用 net_pnl 累加还原每日净值（balance）、最高净值（highlevel）；
        2. 推导 drawdown / ddpercent / return；
        3. 若任一交易日资金 <= 0（爆仓），打印告警并返回全零统计；
        4. 计算 Sharpe Ratio、年化收益、最大回撤等 22 个指标；
           零回撤时 return_drawdown_ratio 安全返回 0（避免 ZeroDivisionError）。

        Returns:
            含以下键的 dict（所有 inf / NaN 被规整为 0）：
            start_date / end_date / total_days / profit_days / loss_days /
            capital / end_balance / max_drawdown / max_ddpercent /
            max_drawdown_duration / total_net_pnl / daily_net_pnl /
            total_commission / daily_commission / total_turnover / daily_turnover /
            total_trade_count / daily_trade_count / total_return / annual_return /
            daily_return / return_std / sharpe_ratio / return_drawdown_ratio。
        """
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
        """用当前 bar 的收盘价更新或新建当日盯市结果。

        在每个时间戳回放结束时调用，将 bars 中各标的的 close_price 写入对应
        PortfolioDailyResult.close_prices；当日尚无结果时自动新建。
        收盘价为 0 的 bar（停牌/合成 fill_bar）回退到 pre_closes 的前收盘价，
        避免停牌日价格归零拉低持仓估值。

        Args:
            bars: 当前时间戳的活跃 bar 字典，key 为 vt_symbol。
            dt: 当前回放时间戳（用于确定日期键）。
        """
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
        """推送单个时间戳的 K 线切片，触发撮合与策略回调。

        对当前时间戳 dt 在 history_data 中查询每个标的的 bar；
        若无对应 bar 则以前收盘价合成一根平行 bar（量为 0，用于维持价格向前填充）。
        流程：cross_order() → strategy.on_bars(bars) → update_daily_close(bars, dt)。

        Args:
            dt: 当前推送的时间戳，必须在 self.dts 集合中。
        """
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
        """按 fill_price_mode 返回撮合参考价：open / close / vwap。

        三种模式对应三种成交价口径，须与训练 label 的 price_ref 对齐（杜绝回测↔label 背离）：
        - ``open``：撮合 bar 的开盘价（默认/旧行为，对齐 next_open）。
        - ``close``：撮合 bar 的收盘价（市价化 MOC，对齐 next_close）。
        - ``vwap``：成交额 / 成交量（对齐 next_vwap）；量或额为 0 时回退到收盘价。

        Args:
            bar: 当前撮合 bar。

        Returns:
            大于等于 0 的参考价格浮点数。
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

    def is_t1_locked(self, vt_symbol: str) -> bool:
        """公开的 T+1 锁定查询：当根 bar 该标的是否因当日买入而不可卖出。

        供策略层（如 :class:`CNNSignalStrategy` 的出场预检）复用同一事实源，
        避免各路径自造 T+1 判断导致豁免口径漂移。判定委托给
        :func:`aitrade.backtest.t1.is_t1_locked`。

        Args:
            vt_symbol: 待判定的合约代码，如 ``"000001.SZSE"``。

        Returns:
            True 表示当日买入不可当日卖出；False 表示允许卖出。
        """
        today = self.datetime.date() if self.datetime is not None else None
        return is_t1_locked(
            vt_symbol,
            self.buy_dates,
            today,
            enabled=self.t_plus1,
            exempt=self.t_plus1_exempt,
        )

    def _t_plus1_locked(self, vt_symbol: str) -> bool:
        """内部别名，保留原撮合路径调用点；等价于 :meth:`is_t1_locked`。"""
        return self.is_t1_locked(vt_symbol)

    def cross_order(self) -> None:
        """对活跃订单执行撮合，先处理 OCO 括号单，再处理普通限价单。

        撮合优先级：
        1. _cross_oco_orders()：止盈/止损括号单（保守假设止损先到）；
        2. 遍历 active_limit_orders 中非 OCO 腿的限价单；
        3. 对每笔订单检查停牌门槛（有量标的在零量 bar 不撮合）、涨跌停封板、
           T+1 限制；通过后调用 _settle_fill() 完成成交结算。
        fill_price_mode 为 close/vwap 时为市价化成交，委托限价不封顶成交价。
        """
        # OCO 止盈止损括号单优先撮合（含「止损先到」保守假设）
        self._cross_oco_orders()

        market_fill = self.fill_price_mode in ("close", "vwap")

        for order in list(self.active_limit_orders.values()):
            # OCO 腿由 _cross_oco_orders 专门处理，普通循环跳过
            if order.oco_group is not None:
                continue

            bar: BarData = self.bars[order.vt_symbol]

            # 停牌门槛：有量数据标的（volume_supported）在无量 bar（停牌/合成 fill_bar）
            # 上不撮合，订单留存到下一根真实有量 bar 再成交。
            # 全程无量的标的（如 parquet 缺 volume 列）豁免此检查，照常撮合。
            if order.vt_symbol in self.volume_supported and bar.volume <= 0:
                continue

            long_cross_price: float = bar.low_price
            short_cross_price: float = bar.high_price
            long_best_price: float = bar.open_price
            short_best_price: float = bar.open_price

            if order.status == STATUS_SUBMITTING:
                order.status = STATUS_NOTTRADED
                self.strategy.update_order(order)

            pricetick: float = self.priceticks[order.vt_symbol]
            pre_close: float = self.pre_closes.get(order.vt_symbol, 0)

            # 品种化涨跌停：ratio=None（转债）→ 无限制（inf/0 使所有比较退化为"允许"）
            _ratio: float | None = self.limit_ratios.get(
                order.vt_symbol, infer_limit_ratio(order.vt_symbol)
            )
            if _ratio is None:
                # ratio=None 时不依赖 pre_close，首根 bar（pre_close=0）即可成交；
                # 有涨跌停品种首根 bar 维持旧行为（limit_up=0 不成交）
                limit_up: float = float("inf")
                limit_down: float = 0.0
            else:
                limit_up = round_to(pre_close * (1 + _ratio), pricetick)
                limit_down = round_to(pre_close * (1 - _ratio), pricetick)

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
            if short_cross and self._t_plus1_locked(order.vt_symbol):
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

            # 停牌门槛：有量数据标的在无量 bar 上不撮合 OCO 腿
            if legs[0].vt_symbol in self.volume_supported and bar.volume <= 0:
                continue

            # T+1：当日买入不可当日卖出，本根跳过（OCO 腿保留待次日）
            if self._t_plus1_locked(bar.vt_symbol):
                continue

            # 跌停封死（最高价仍 ≤ 跌停价）时卖不出，与主撮合 short 口径一致
            pricetick: float = self.priceticks[bar.vt_symbol]
            pre_close: float = self.pre_closes.get(bar.vt_symbol, 0)
            # 品种化跌停价：ratio=None（转债）→ 0.0，high_price > 0.0 永远满足（无限制）
            _oco_ratio: float | None = self.limit_ratios.get(
                bar.vt_symbol, infer_limit_ratio(bar.vt_symbol)
            )
            if _oco_ratio is None:
                oco_limit_down: float = 0.0
            else:
                oco_limit_down = round_to(pre_close * (1 - _oco_ratio), pricetick)
            if pre_close and bar.high_price <= oco_limit_down:
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
        """成交结算：叠加滑点、更新订单状态、记录成交、更新现金与持仓。

        买入方向叠加不利滑点（价格更高）；卖出方向叠加不利滑点（价格更低）。
        成交后：
        - 从 active_limit_orders 移除已成交订单；
        - 创建 TradeData 并通知 strategy.update_trade()；
        - 更新 self.cash（含佣金与卖出印花税）；
        - 买入时记录 buy_dates，供 T+1 锁仓判定。
        买卖共用，避免两处各算导致口径不一致。

        Args:
            order: 待结算的限价单。
            raw_price: 结算前的原始参考价（滑点将在此基础上叠加）。

        Returns:
            已写入 self.trades 的 TradeData 对象。
        """
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

        T+1 保护：全局 t_plus1=True 且标的未豁免且当日有买入时，拒绝成交（直接返回）。
        此为引擎层防线，确保策略层 can_sell 遗漏时也不会绕过 T+1 约束。
        """
        if volume <= 0:
            return
        # T+1：当日买入不可当日 intrabar 平仓（豁免标的除外）
        if self._t_plus1_locked(vt_symbol):
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
        """返回当前时间戳对应的模型信号行（从 signal_df 中按 datetime 过滤）。

        Returns:
            过滤后的 polars DataFrame；当 datetime 为 None 或无匹配行时返回空 DataFrame。
        """
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
        """向撮合引擎挂出一笔限价单，返回 vt_orderid 列表（单笔）。

        价格先对齐到 pricetick；新建 OrderData 以 STATUS_SUBMITTING 状态写入
        active_limit_orders 和 limit_orders，等待下一个 cross_order() 时撮合。

        Args:
            strategy: 发出委托的策略实例（当前未使用，预留扩展）。
            vt_symbol: 合约代码，如 ``"000001.SZSE"``。
            direction: 委托方向，``Direction.LONG`` 或 ``Direction.SHORT``。
            offset: 开平标志，``Offset.OPEN`` 或 ``Offset.CLOSE``。
            price: 委托价格（元）。
            volume: 委托数量（手/股）。

        Returns:
            含单个 vt_orderid 字符串的列表，如 ``["BACKTESTING.1"]``。
        """
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
        """撤销指定活跃限价单，更新状态为 CANCELLED 并通知策略。

        若 vt_orderid 不在 active_limit_orders 中（已成交或不存在），静默返回。

        Args:
            strategy: 发出撤单请求的策略实例。
            vt_orderid: 要撤销的订单全局 ID，如 ``"BACKTESTING.3"``。
        """
        if vt_orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(vt_orderid)

        order.status = STATUS_CANCELLED
        self.strategy.update_order(order)

    def write_log(self, msg: str, strategy: BaseStrategy | None = None) -> None:
        """记录一条带时间戳前缀的日志消息到 self.logs 列表。

        Args:
            msg: 日志正文，将被拼接为 ``"{self.datetime}  {msg}"``。
            strategy: 调用方策略实例（当前仅保留兼容，未使用）。
        """
        msg = f"{self.datetime}  {msg}"
        self.logs.append(msg)

    def get_all_trades(self) -> list[TradeData]:
        """返回所有历史成交记录列表。

        Returns:
            按 vt_tradeid 键存储的全部 TradeData 对象列表（顺序不保证）。
        """
        return list(self.trades.values())

    def get_all_orders(self) -> list[OrderData]:
        """返回所有提交过的订单列表（含已成交、已撤销）。

        Returns:
            全部 OrderData 列表（包括已完成状态的订单）。
        """
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> list[PortfolioDailyResult]:
        """返回所有交易日的组合盯市结果列表。

        Returns:
            按日期顺序（dict 插入顺序）排列的 PortfolioDailyResult 列表。
        """
        return list(self.daily_results.values())

    def get_cash_available(self) -> float:
        """返回当前可用现金余额（已扣除已成交买入金额及手续费）。

        Returns:
            现金余额浮点数（元），可能为负（爆仓场景）。
        """
        return self.cash

    def get_holding_value(self) -> float:
        """按当前 bar 收盘价估算持仓市值。

        Returns:
            各标的 close_price × pos × size 加总的持仓市值（元）。
        """
        holding_value: float = 0

        for vt_symbol, pos in self.strategy.pos_data.items():
            bar: BarData = self.bars[vt_symbol]
            size: float = self.sizes[vt_symbol]

            holding_value += bar.close_price * pos * size

        return holding_value
