from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

import polars as pl

from ...lab import BarData
from ..logger import logger


Direction: str = "Direction"
Offset: str = "Offset"


class Direction:
    LONG: str = "long"
    SHORT: str = "short"


class Offset:
    OPEN: str = "open"
    CLOSE: str = "close"


class OrderData:
    """Order data"""
    def __init__(
        self,
        symbol: str,
        exchange: str,
        orderid: str,
        direction: str,
        offset: str,
        price: float,
        volume: float,
        status: str,
        datetime,
        gateway_name: str = "",
    ) -> None:
        self.symbol: str = symbol
        self.exchange: str = exchange
        self.orderid: str = orderid
        self.direction: str = direction
        self.offset: str = offset
        self.price: float = price
        self.volume: float = volume
        self.status: str = status
        self.datetime = datetime
        self.gateway_name: str = gateway_name
        self.traded: float = 0

    @property
    def vt_orderid(self) -> str:
        return f"{self.gateway_name}.{self.orderid}"

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange}"

    def is_active(self) -> bool:
        return self.status == "submitting" or self.status == "nottraded"


class TradeData:
    """Trade data"""
    def __init__(
        self,
        symbol: str,
        exchange: str,
        orderid: str,
        tradeid: str,
        direction: str,
        offset: str,
        price: float,
        volume: float,
        datetime,
        gateway_name: str = "",
    ) -> None:
        self.symbol: str = symbol
        self.exchange: str = exchange
        self.orderid: str = orderid
        self.tradeid: str = tradeid
        self.direction: str = direction
        self.offset: str = offset
        self.price: float = price
        self.volume: float = volume
        self.datetime = datetime
        self.gateway_name: str = gateway_name

    @property
    def vt_tradeid(self) -> str:
        return f"{self.gateway_name}.{self.tradeid}"

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange}"


if TYPE_CHECKING:
    from .backtesting import BacktestingEngine


class AlphaStrategy(metaclass=ABCMeta):
    """Alpha strategy template class"""

    def __init__(
        self,
        strategy_engine: "BacktestingEngine",
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict
    ) -> None:
        """Constructor"""
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
        """Initialization callback"""
        pass

    @abstractmethod
    def on_bars(self, bars: dict[str, BarData]) -> None:
        """Bar slice callback"""
        pass

    @abstractmethod
    def on_trade(self, trade: TradeData) -> None:
        """Trade callback"""
        pass

    def update_trade(self, trade: TradeData) -> None:
        """Update trade data"""
        if trade.direction == Direction.LONG:
            self.pos_data[trade.vt_symbol] += trade.volume
        else:
            self.pos_data[trade.vt_symbol] -= trade.volume

        self.on_trade(trade)

    def update_order(self, order: OrderData) -> None:
        """Update order data"""
        self.orders[order.vt_orderid] = order

        if not order.is_active() and order.vt_orderid in self.active_orderids:
            self.active_orderids.remove(order.vt_orderid)

    def get_signal(self) -> pl.DataFrame:
        """Get current signal"""
        return self.strategy_engine.get_signal()

    def buy(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        """Buy to open position"""
        return self.send_order(vt_symbol, Direction.LONG, Offset.OPEN, price, volume)

    def sell(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        """Sell to close position"""
        return self.send_order(vt_symbol, Direction.SHORT, Offset.CLOSE, price, volume)

    def short(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        """Sell to open position"""
        return self.send_order(vt_symbol, Direction.SHORT, Offset.OPEN, price, volume)

    def cover(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        """Buy to close position"""
        return self.send_order(vt_symbol, Direction.LONG, Offset.CLOSE, price, volume)

    def send_order(
        self,
        vt_symbol: str,
        direction: str,
        offset: str,
        price: float,
        volume: float
    ) -> list[str]:
        """Send order"""
        vt_orderids: list = self.strategy_engine.send_order(
            self, vt_symbol, direction, offset, price, volume
        )

        for vt_orderid in vt_orderids:
            self.active_orderids.add(vt_orderid)

        return vt_orderids

    def cancel_order(self, vt_orderid: str) -> None:
        """Cancel order"""
        self.strategy_engine.cancel_order(self, vt_orderid)

    def cancel_all(self) -> None:
        """Cancel all active orders"""
        for vt_orderid in list(self.active_orderids):
            self.cancel_order(vt_orderid)

    def get_pos(self, vt_symbol: str) -> float:
        """Query current position"""
        return self.pos_data[vt_symbol]

    def get_target(self, vt_symbol: str) -> float:
        """Query target position"""
        return self.target_data[vt_symbol]

    def set_target(self, vt_symbol: str, target: float) -> None:
        """Set target position"""
        self.target_data[vt_symbol] = target

    def execute_trading(self, bars: dict[str, BarData], price_add: float) -> None:
        """Execute position adjustment based on targets"""
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
        """Write log message"""
        self.strategy_engine.write_log(msg, self)

    def get_cash_available(self) -> float:
        """Get available cash"""
        return self.strategy_engine.get_cash_available()

    def get_holding_value(self) -> float:
        """Get holding market value"""
        return self.strategy_engine.get_holding_value()

    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        return self.get_cash_available() + self.get_holding_value()

    def get_cash(self) -> float:
        """Legacy compatibility method"""
        return self.get_cash_available()
