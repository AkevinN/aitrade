"""
真实券商网关适配桩（迭代 9）：定义对接形状，实际网络对接需券商连接器 + 凭据。

【集成点】A 股个人常用 QMT/miniqmt、Ptrade 等。要接实盘，注入一个实现了
send_order/cancel_order/query_orders/query_positions/query_account 的 connector
（封装券商 SDK 调用、超时与重试）。未注入 connector 时本类拒绝下单（防误用）。

注意：凭据用环境变量/密钥管理，切勿入库或写入仓库。
"""

from __future__ import annotations

from typing import Optional

from .gateway import BrokerGateway, OrderReport, OrderRequest


class StubLiveBroker(BrokerGateway):
    """实盘网关桩。connector 为实际券商连接器（需外部实现并注入）。"""

    def __init__(self, connector: Optional[object] = None) -> None:
        self._connector = connector

    def _require(self) -> object:
        if self._connector is None:
            raise NotImplementedError(
                "实盘网关未配置券商连接器（connector）。请注入 QMT/miniqmt 或 Ptrade 等"
                "连接器后再用于实盘；详见 docs/08 迭代 9。"
            )
        return self._connector

    def send_order(self, req: OrderRequest) -> OrderReport:
        return self._require().send_order(req)  # type: ignore[attr-defined]

    def cancel_order(self, client_order_id: str) -> bool:
        return self._require().cancel_order(client_order_id)  # type: ignore[attr-defined]

    def query_orders(self) -> list[OrderReport]:
        return self._require().query_orders()  # type: ignore[attr-defined]

    def query_positions(self) -> dict[str, float]:
        return self._require().query_positions()  # type: ignore[attr-defined]

    def query_account(self) -> dict:
        return self._require().query_account()  # type: ignore[attr-defined]
