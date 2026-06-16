"""
执行网关抽象（迭代 8）：统一下单/撤单/查询接口，回测与实盘只在此层不同。

任何券商实现 BrokerGateway 即可接入；策略/信号/风控复用回测同一套代码。
关键：每笔订单带唯一 client_order_id，幂等——重复提交不产生双重下单。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# 订单方向 / 开平 / 状态常量
DIRECTION_LONG = "long"
DIRECTION_SHORT = "short"
OFFSET_OPEN = "open"
OFFSET_CLOSE = "close"

STATUS_SUBMITTED = "submitted"
STATUS_FILLED = "filled"
STATUS_PARTIAL = "partial"
STATUS_CANCELLED = "cancelled"
STATUS_REJECTED = "rejected"


@dataclass
class OrderRequest:
    """委托请求（下单指令）。

    Attributes:
        client_order_id: 幂等键，由调用方生成（如 "{rebalance_id}:{vt_symbol}:buy"）；
                         同 id 重复提交时网关返回首单结果，不重复下单。
        vt_symbol:       合约代码，如 "000001.SZSE"。
        direction:       方向："long"（买/做多）| "short"（卖/做空）。
        offset:          开平："open"（开仓）| "close"（平仓）。
        volume:          委托数量（股数或手数）。
        price:           委托价格（限价单）；市价单时由网关忽略。
        order_type:      订单类型："limit"（限价，默认）| "market"（市价）。
    """

    client_order_id: str          # 幂等键
    vt_symbol: str
    direction: str                # long / short
    offset: str                   # open / close
    volume: float
    price: float
    order_type: str = "limit"     # limit / market


@dataclass
class OrderReport:
    """订单回报（网关响应）。

    Attributes:
        client_order_id: 与请求 client_order_id 一致，供对账使用。
        vt_symbol:       合约代码。
        direction:       方向（long / short）。
        status:          订单状态（submitted / filled / partial / cancelled / rejected）。
        volume:          委托数量。
        filled_volume:   已成交数量。
        avg_price:       成交均价；未成交时为 0.0。
        message:         附言（如拒单原因）。
    """

    client_order_id: str
    vt_symbol: str
    direction: str
    status: str
    volume: float
    filled_volume: float = 0.0
    avg_price: float = 0.0
    message: str = ""

    @property
    def is_active(self) -> bool:
        """订单是否仍处于活跃状态（已报/部分成交，尚未终结）。"""
        return self.status in (STATUS_SUBMITTED, STATUS_PARTIAL)


@runtime_checkable
class BrokerGateway(Protocol):
    """券商网关协议，所有网关实现（实盘/模拟盘/测试桩）必须满足此接口。

    策略/风控/LiveTrader 只依赖该协议，实现可任意替换。
    幂等红线：`send_order` 同 client_order_id 重复提交时必须返回首单结果，不产生双重下单。
    """

    def send_order(self, req: OrderRequest) -> OrderReport:
        """提交一笔委托，幂等。

        同 client_order_id 重复提交必须返回首单结果，不产生双重下单。

        Args:
            req: 委托请求，含 client_order_id 幂等键。

        Returns:
            OrderReport；被风控/券商拒单时 status=rejected，message 为原因。
        """
        ...

    def cancel_order(self, client_order_id: str) -> bool:
        """按 client_order_id 撤销一笔委托。

        Args:
            client_order_id: 要撤销的委托幂等键。

        Returns:
            True 表示撤单成功，False 表示订单不存在或已终结、不可撤。
        """
        ...

    def query_orders(self) -> list[OrderReport]:
        """查询全部订单回报。

        Returns:
            OrderReport 列表；无订单时返回空列表。
        """
        ...

    def query_positions(self) -> dict[str, float]:
        """查询当前持仓。

        Returns:
            vt_symbol → 持仓数量（股数/手数）的字典；无持仓时返回空字典。
        """
        ...

    def query_account(self) -> dict:
        """查询账户资金信息。

        Returns:
            账户资金字段字典（如可用资金、总权益等），具体键由各网关实现约定。
        """
        ...
