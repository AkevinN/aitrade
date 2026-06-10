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
    client_order_id: str          # 幂等键
    vt_symbol: str
    direction: str                # long / short
    offset: str                   # open / close
    volume: float
    price: float
    order_type: str = "limit"     # limit / market


@dataclass
class OrderReport:
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
        return self.status in (STATUS_SUBMITTED, STATUS_PARTIAL)


@runtime_checkable
class BrokerGateway(Protocol):
    def send_order(self, req: OrderRequest) -> OrderReport: ...
    def cancel_order(self, client_order_id: str) -> bool: ...
    def query_orders(self) -> list[OrderReport]: ...
    def query_positions(self) -> dict[str, float]: ...
    def query_account(self) -> dict: ...
