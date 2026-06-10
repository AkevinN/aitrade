"""
风控闸门网关（迭代 9）：包裹任意 BrokerGateway，下单前过全局风控并审计。

职责分层：
- 全局闸门（kill-switch / 单日亏损熔断 / 对账阻断）在此强制，绕不过去；
- 仓位/单票上限等需组合上下文的检查在上游（信号服务/LiveTrader）完成；
- 每笔下单与结果写审计，保证可追溯。
"""

from __future__ import annotations

from typing import Optional

from .audit import AuditLog
from .gateway import STATUS_REJECTED, BrokerGateway, OrderReport, OrderRequest
from .risk import RiskManager


class RiskGuardedGateway(BrokerGateway):
    def __init__(
        self,
        inner: BrokerGateway,
        risk: RiskManager,
        audit: Optional[AuditLog] = None,
        version: str = "",
    ) -> None:
        self.inner = inner
        self.risk = risk
        self.audit = audit
        self.version = version
        # 对账阻断标志：对账差异过大时由外部置位，阻断后续自动下单
        self.blocked_by_reconcile: bool = False

    def block_for_reconcile(self, blocked: bool = True) -> None:
        self.blocked_by_reconcile = blocked

    def send_order(self, req: OrderRequest) -> OrderReport:
        ok, reason = self.risk.can_trade()
        if ok and self.blocked_by_reconcile:
            ok, reason = False, "对账差异过大，自动下单已阻断"

        if not ok:
            report = OrderReport(req.client_order_id, req.vt_symbol, req.direction,
                                 STATUS_REJECTED, req.volume, message=reason)
            if self.audit:
                self.audit.record("order_rejected", {
                    "client_order_id": req.client_order_id, "vt_symbol": req.vt_symbol,
                    "direction": req.direction, "volume": req.volume, "reason": reason,
                }, version=self.version)
            return report

        report = self.inner.send_order(req)
        if self.audit:
            self.audit.record("order", {
                "client_order_id": req.client_order_id, "vt_symbol": req.vt_symbol,
                "direction": req.direction, "volume": req.volume, "price": req.price,
                "status": report.status, "filled": report.filled_volume,
            }, version=self.version)
        return report

    def cancel_order(self, client_order_id: str) -> bool:
        return self.inner.cancel_order(client_order_id)

    def query_orders(self) -> list[OrderReport]:
        return self.inner.query_orders()

    def query_positions(self) -> dict[str, float]:
        return self.inner.query_positions()

    def query_account(self) -> dict:
        return self.inner.query_account()
