"""
Shared backtesting data types — Direction, Offset, OrderData, TradeData, BarData.

Extracted from alpha/strategy/template.py for cross-module reuse.
BarData is re-exported from alpha.lab so every module imports it from one place.
"""

from ..alpha.lab import BarData

__all__ = ["Direction", "Offset", "OrderData", "TradeData", "BarData"]


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
        order_type: str = "limit",
        oco_group: str | None = None,
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
        # order_type: limit（限价，默认）| stop（止损触发单，用于 OCO 止损腿）
        self.order_type: str = order_type
        # oco_group: 同一 OCO 括号单的两条腿共享同一标识；一腿成交即撤另一腿
        self.oco_group: str | None = oco_group

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
