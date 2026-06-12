"""
Shared backtesting data types — Direction, Offset, OrderData, TradeData, BarData.

Extracted from alpha/strategy/template.py for cross-module reuse.
BarData is re-exported from alpha.lab so every module imports it from one place.
"""

from ..alpha.lab import BarData

__all__ = ["Direction", "Offset", "OrderData", "TradeData", "BarData"]


class Direction:
    """委托方向常量：多头（买入）或空头（卖出）。"""

    LONG: str = "long"
    SHORT: str = "short"


class Offset:
    """开平仓标志常量。"""

    OPEN: str = "open"
    CLOSE: str = "close"


class OrderData:
    """单笔委托数据，贯穿从下单到成交/撤单的整个生命周期。"""

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
        """初始化委托数据。

        Args:
            symbol: 合约代码（不含交易所后缀），如 ``"000001"``。
            exchange: 交易所代码，如 ``"SZSE"``。
            orderid: 订单编号字符串（引擎内自增）。
            direction: 委托方向，``Direction.LONG`` 或 ``Direction.SHORT``。
            offset: 开平标志，``Offset.OPEN`` 或 ``Offset.CLOSE``。
            price: 委托价格（元）。
            volume: 委托数量（手/股）。
            status: 初始状态，通常为 ``"submitting"``。
            datetime: 下单时间戳。
            gateway_name: 网关名称，回测固定为 ``"BACKTESTING"``。
            order_type: 订单类型，``"limit"``（限价单）或 ``"stop"``（OCO 止损触发单）。
            oco_group: OCO 括号单组 ID；None 表示普通限价单。
        """
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
        """返回全局唯一订单 ID：``"{gateway_name}.{orderid}"``。"""
        return f"{self.gateway_name}.{self.orderid}"

    @property
    def vt_symbol(self) -> str:
        """返回合约全称：``"{symbol}.{exchange}"``，如 ``"000001.SZSE"``。"""
        return f"{self.symbol}.{self.exchange}"

    def is_active(self) -> bool:
        """判断订单是否仍在活跃挂单中（submitting 或 nottraded）。

        Returns:
            True 表示订单尚未成交也未撤销，False 表示已完结。
        """
        return self.status == "submitting" or self.status == "nottraded"


class TradeData:
    """单笔成交数据，由引擎在撮合成功后创建。"""

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
        """初始化成交数据。

        Args:
            symbol: 合约代码（不含交易所后缀）。
            exchange: 交易所代码，如 ``"SZSE"``。
            orderid: 对应委托编号。
            tradeid: 成交编号（引擎内自增）。
            direction: 成交方向，``Direction.LONG`` 或 ``Direction.SHORT``。
            offset: 开平标志，``Offset.OPEN`` 或 ``Offset.CLOSE``。
            price: 成交价格（含滑点调整后的最终价，元）。
            volume: 成交数量（手/股）。
            datetime: 成交时间戳。
            gateway_name: 网关名称，回测固定为 ``"BACKTESTING"``。
        """
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
        """返回全局唯一成交 ID：``"{gateway_name}.{tradeid}"``。"""
        return f"{self.gateway_name}.{self.tradeid}"

    @property
    def vt_symbol(self) -> str:
        """返回合约全称：``"{symbol}.{exchange}"``，如 ``"000001.SZSE"``。"""
        return f"{self.symbol}.{self.exchange}"
