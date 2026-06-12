"""
Shared strategy base class for backtesting.

Extracted from alpha/strategy/template.py for cross-module reuse.
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

import polars as pl

from .types import BarData
from .types import Direction, Offset, OrderData, TradeData

if TYPE_CHECKING:
    from .engine import BacktestingEngine


class BaseStrategy(metaclass=ABCMeta):
    """回测策略基类，为各量化方案提供统一的订单管理与持仓跟踪接口。

    子类需实现 on_init / on_bars / on_trade 三个抽象方法；其余方法（buy/sell/
    send_oco/execute_trading 等）直接代理到 BacktestingEngine 完成实际撮合。
    持仓通过 pos_data 字典自动维护（每笔成交后在 update_trade 中更新）。
    """

    def __init__(
        self,
        strategy_engine: "BacktestingEngine",
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict
    ) -> None:
        """初始化策略，绑定引擎并将 setting 中的键值注入对应属性。

        Args:
            strategy_engine: 驱动该策略的 BacktestingEngine 实例。
            strategy_name: 策略名称字符串，用于日志标识。
            vt_symbols: 策略关注的合约列表，如 ``["000001.SZSE"]``。
            setting: 参数字典，其中已在子类声明为类属性的键会被 setattr 注入。
        """
        self.strategy_engine: "BacktestingEngine" = strategy_engine
        self.strategy_name: str = strategy_name
        self.vt_symbols: list[str] = vt_symbols

        self.pos_data: dict[str, float] = defaultdict(float)
        self.target_data: dict[str, float] = defaultdict(float)

        self.orders: dict[str, OrderData] = {}
        self.active_orderids: set[str] = set()

        for k, v in setting.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @abstractmethod
    def on_init(self) -> None:
        """策略初始化回调，在回测开始前由引擎调用一次。

        子类可在此加载预训练模型、设置初始参数或预热状态。
        """
        pass

    @abstractmethod
    def on_bars(self, bars: dict[str, BarData]) -> None:
        """每个时间戳推送一切片 bar 数据时的回调，策略核心逻辑入口。

        Args:
            bars: 当前时间戳各标的的 BarData，key 为 vt_symbol。
                  注意：非所有标的都必然出现（停牌或数据缺失时 key 不存在）。
        """
        pass

    @abstractmethod
    def on_trade(self, trade: TradeData) -> None:
        """每笔成交完成后的回调，可在此更新策略内部状态（如已成交仓位统计）。

        Args:
            trade: 本次成交的 TradeData 对象。
        """
        pass

    def update_trade(self, trade: TradeData) -> None:
        """根据成交方向更新 pos_data，并触发 on_trade 回调。

        买入时仓位增加，卖出时仓位减少。由引擎在 _settle_fill() 内调用，
        子类通常不需要直接调用。

        Args:
            trade: 已完成的 TradeData 对象。
        """
        if trade.direction == Direction.LONG:
            self.pos_data[trade.vt_symbol] += trade.volume
        else:
            self.pos_data[trade.vt_symbol] -= trade.volume

        self.on_trade(trade)

    def update_order(self, order: OrderData) -> None:
        """更新订单记录，并在订单已完结时从 active_orderids 中移除。

        由引擎在订单状态变化时调用（SUBMITTING→NOTTRADED→ALLTRADED/CANCELLED）。
        子类通常不直接调用。

        Args:
            order: 最新状态的 OrderData 对象。
        """
        self.orders[order.vt_orderid] = order

        if not order.is_active() and order.vt_orderid in self.active_orderids:
            self.active_orderids.remove(order.vt_orderid)

    def get_signal(self) -> pl.DataFrame:
        """获取当前时间戳的模型信号 DataFrame（代理到引擎）。

        Returns:
            当前时间点对应的信号行；空 DataFrame 表示无信号或未开始回放。
        """
        return self.strategy_engine.get_signal()

    def buy(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        """挂出多开买入限价单（LONG/OPEN），返回 vt_orderid 列表。

        Args:
            vt_symbol: 合约代码，如 ``"000001.SZSE"``。
            price: 委托价格（元）。
            volume: 委托数量（手/股）。

        Returns:
            含单个 vt_orderid 字符串的列表。
        """
        return self.send_order(vt_symbol, Direction.LONG, Offset.OPEN, price, volume)

    def sell(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        """挂出平多卖出限价单（SHORT/CLOSE），返回 vt_orderid 列表。

        Args:
            vt_symbol: 合约代码。
            price: 委托价格（元）。
            volume: 委托数量（手/股）。

        Returns:
            含单个 vt_orderid 字符串的列表。
        """
        return self.send_order(vt_symbol, Direction.SHORT, Offset.CLOSE, price, volume)

    def short(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        """挂出空开卖出限价单（SHORT/OPEN），返回 vt_orderid 列表。

        Args:
            vt_symbol: 合约代码。
            price: 委托价格（元）。
            volume: 委托数量（手/股）。

        Returns:
            含单个 vt_orderid 字符串的列表。
        """
        return self.send_order(vt_symbol, Direction.SHORT, Offset.OPEN, price, volume)

    def cover(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        """挂出平空买入限价单（LONG/CLOSE），返回 vt_orderid 列表。

        Args:
            vt_symbol: 合约代码。
            price: 委托价格（元）。
            volume: 委托数量（手/股）。

        Returns:
            含单个 vt_orderid 字符串的列表。
        """
        return self.send_order(vt_symbol, Direction.LONG, Offset.CLOSE, price, volume)

    def send_order(
        self,
        vt_symbol: str,
        direction: str,
        offset: str,
        price: float,
        volume: float
    ) -> list[str]:
        """向回测引擎提交限价单，并将新订单 id 加入 active_orderids。

        Args:
            vt_symbol: 合约代码，如 ``"000001.SZSE"``。
            direction: 委托方向，``Direction.LONG`` 或 ``Direction.SHORT``。
            offset: 开平标志，``Offset.OPEN`` 或 ``Offset.CLOSE``。
            price: 委托价格（元）。
            volume: 委托数量（手/股）。

        Returns:
            含单个 vt_orderid 字符串的列表。
        """
        vt_orderids: list = self.strategy_engine.send_order(
            self, vt_symbol, direction, offset, price, volume
        )

        for vt_orderid in vt_orderids:
            self.active_orderids.add(vt_orderid)

        return vt_orderids

    def send_oco(
        self,
        vt_symbol: str,
        tp_price: float,
        sl_price: float,
        volume: float,
    ) -> list[str]:
        """挂出 OCO 止盈止损括号单（止盈限价 + 止损触发），一腿成交即撤另一腿。"""
        vt_orderids: list = self.strategy_engine.send_oco(
            self, vt_symbol, tp_price, sl_price, volume
        )
        for vt_orderid in vt_orderids:
            self.active_orderids.add(vt_orderid)
        return vt_orderids

    def cancel_order(self, vt_orderid: str) -> None:
        """撤销指定活跃订单（代理到引擎）。

        Args:
            vt_orderid: 要撤销的订单全局 ID，如 ``"BACKTESTING.3"``。
        """
        self.strategy_engine.cancel_order(self, vt_orderid)

    def cancel_all(self) -> None:
        """撤销当前全部活跃订单（遍历 active_orderids 逐一撤单）。"""
        for vt_orderid in list(self.active_orderids):
            self.cancel_order(vt_orderid)

    def get_pos(self, vt_symbol: str) -> float:
        """查询指定标的当前净持仓（多为正、空为负）。

        Args:
            vt_symbol: 合约代码，如 ``"000001.SZSE"``。

        Returns:
            当前净持仓量（手/股），未持仓返回 0.0。
        """
        return self.pos_data[vt_symbol]

    def get_target(self, vt_symbol: str) -> float:
        """查询指定标的的目标仓位。

        Args:
            vt_symbol: 合约代码。

        Returns:
            目标持仓量（手/股），未设置时返回 0.0。
        """
        return self.target_data[vt_symbol]

    def set_target(self, vt_symbol: str, target: float) -> None:
        """设置指定标的的目标仓位，供 execute_trading() 对齐实际持仓。

        Args:
            vt_symbol: 合约代码。
            target: 目标持仓量（手/股）。
        """
        self.target_data[vt_symbol] = target

    def execute_trading(self, bars: dict[str, BarData], price_add: float) -> None:
        """根据 target_data 与 pos_data 的差值，自动生成调仓委托。

        先撤所有活跃订单，再遍历 bars 中每个标的计算 diff = target - pos：
        - diff > 0：先平空（cover），再买入（buy）；
        - diff < 0：先平多（sell），再做空（short）；
        委托价 = close_price × (1 ± price_add)，正数 price_add 表示买入超价、卖出割价。

        Args:
            bars: 当前时间戳的 bar 切片，用于获取参考价格。
            price_add: 价格追价幅度（小数），如 0.001 表示追价 0.1%。
        """
        self.cancel_all()

        for vt_symbol, bar in bars.items():
            target: float = self.get_target(vt_symbol)
            pos: float = self.get_pos(vt_symbol)
            diff: float = target - pos

            if diff > 0:
                order_price: float = bar.close_price * (1 + price_add)

                cover_volume: float = 0
                buy_volume: float = 0

                if pos < 0:
                    cover_volume = min(diff, abs(pos))
                    buy_volume = diff - cover_volume
                else:
                    buy_volume = diff

                if cover_volume:
                    self.cover(vt_symbol, order_price, cover_volume)

                if buy_volume:
                    self.buy(vt_symbol, order_price, buy_volume)
            elif diff < 0:
                order_price = bar.close_price * (1 - price_add)

                sell_volume: float = 0
                short_volume: float = 0

                if pos > 0:
                    sell_volume = min(abs(diff), pos)
                    short_volume = abs(diff) - sell_volume
                else:
                    short_volume = abs(diff)

                if sell_volume:
                    self.sell(vt_symbol, order_price, sell_volume)

                if short_volume:
                    self.short(vt_symbol, order_price, short_volume)

    def write_log(self, msg: str) -> None:
        """写入带时间戳前缀的日志（代理到引擎 write_log）。

        Args:
            msg: 日志内容字符串。
        """
        self.strategy_engine.write_log(msg, self)

    def get_cash_available(self) -> float:
        """查询当前可用现金余额（代理到引擎）。

        Returns:
            现金余额浮点数（元）。
        """
        return self.strategy_engine.get_cash_available()

    def get_holding_value(self) -> float:
        """查询当前持仓市值（代理到引擎，按各标的最新收盘价估算）。

        Returns:
            持仓市值浮点数（元）。
        """
        return self.strategy_engine.get_holding_value()

    def get_portfolio_value(self) -> float:
        """查询当前总组合价值 = 可用现金 + 持仓市值。

        Returns:
            组合总价值浮点数（元）。
        """
        return self.get_cash_available() + self.get_holding_value()

    def get_cash(self) -> float:
        """查询当前可用现金（兼容旧接口，等同于 get_cash_available）。

        Returns:
            现金余额浮点数（元）。
        """
        return self.get_cash_available()
