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
    """风控闸门网关：包裹任意 BrokerGateway，下单前过全局风控并写审计。

    职责分层：
    - kill-switch / 单日亏损熔断 / 对账阻断 在此强制，任何下单调用都绕不过去；
    - 仓位/单票上限等需组合上下文的检查在上游（SignalService/LiveTrader）完成。

    Example:
        >>> gw = RiskGuardedGateway(inner=paper_broker, risk=RiskManager(), audit=audit_log)
        >>> gw.send_order(OrderRequest("oid1", "000001.SZSE", "long", "open", 100, 12.50))
    """

    def __init__(
        self,
        inner: BrokerGateway,
        risk: RiskManager,
        audit: Optional[AuditLog] = None,
        version: str = "",
    ) -> None:
        """初始化风控网关。

        Args:
            inner:   被包裹的真实/模拟网关，通过风控后委托它执行。
            risk:    RiskManager 实例（全局闸门状态由其持有）。
            audit:   审计日志；None 时跳过审计写入。
            version: 模型/scheme 版本标签，随每条审计记录写入。
        """
        self.inner = inner
        self.risk = risk
        self.audit = audit
        self.version = version
        # 对账阻断标志：对账差异过大时由外部置位，阻断后续自动下单
        self.blocked_by_reconcile: bool = False

    def block_for_reconcile(self, blocked: bool = True) -> None:
        """设置对账阻断标志。

        Args:
            blocked: True 表示阻断（默认），False 表示解除阻断。
        """
        self.blocked_by_reconcile = blocked

    def send_order(self, req: OrderRequest) -> OrderReport:
        """下单（含风控前置检查与审计）。

        先过全局风控（kill-switch / 熔断 / 对账阻断），不通过则直接返回 REJECTED 回报
        并写审计（order_rejected）；通过则委托 inner 下单并写审计（order）。

        Args:
            req: 委托请求，含 client_order_id 幂等键。

        Returns:
            OrderReport；风控拦截时 status=rejected，message 为拦截原因。
        """
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
        """透传撤单到 inner（不经风控检查）。

        Args:
            client_order_id: 要撤销的委托 id。

        Returns:
            True 表示撤单成功，False 表示订单不存在或不可撤。
        """
        return self.inner.cancel_order(client_order_id)

    def query_orders(self) -> list[OrderReport]:
        """透传查询全部订单回报。"""
        return self.inner.query_orders()

    def query_positions(self) -> dict[str, float]:
        """透传查询当前持仓（vt_symbol → 股数）。"""
        return self.inner.query_positions()

    def query_account(self) -> dict:
        """透传查询账户资金信息。"""
        return self.inner.query_account()
