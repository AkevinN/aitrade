"""
模拟盘网关 PaperBroker（迭代 8）：实现 BrokerGateway，用于纸面/模拟盘验证闭环。

成交口径复用回测：买入扣佣金、卖出扣佣金+印花税；按委托价即时成交（市价/限价）。
幂等：同 client_order_id 重复提交返回首单结果，不重复成交。资金/持仓不足则拒单。
"""

from __future__ import annotations

from .gateway import (
    DIRECTION_LONG,
    STATUS_FILLED,
    STATUS_REJECTED,
    BrokerGateway,
    OrderReport,
    OrderRequest,
)


class PaperBroker(BrokerGateway):
    """模拟盘网关：内存撮合，用于纸面/沙盒验证，不连接任何真实交易所。

    成交口径：按委托价即时成交（限价/市价不区分）；
    买入扣佣金；卖出扣佣金 + 印花税；资金/持仓不足拒单；
    同 client_order_id 重复提交返回首单结果（幂等）。

    Example:
        >>> broker = PaperBroker(cash=500_000)
        >>> req = OrderRequest("oid1", "000001.SZSE", "long", "open", 100, 12.5)
        >>> report = broker.send_order(req)
        >>> report.status
        'filled'
    """

    def __init__(
        self,
        cash: float = 1_000_000.0,
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.0005,
        size: float = 1.0,
    ) -> None:
        """
        Args:
            cash:            初始资金（元）。
            commission_rate: 单边佣金率（买卖均收）。
            stamp_duty:      卖出印花税率（买入不收，默认 0.05%，2023 起 A 股标准）。
            size:            合约乘数（股票为 1，期货按品种设置）。
        """
        self.cash = cash
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.size = size
        self.positions: dict[str, float] = {}
        self._orders: dict[str, OrderReport] = {}   # client_order_id -> report
        self.trades: list[dict] = []

    # -- BrokerGateway --
    def send_order(self, req: OrderRequest) -> OrderReport:
        """下单并撮合（内存即时成交）。

        幂等：同 client_order_id 重复提交直接返回首单结果，不重复扣款/加仓。
        资金不足（买入）或持仓不足（卖出）时返回 REJECTED 回报。

        Args:
            req: 委托请求。

        Returns:
            OrderReport；成交时 status=filled，拒单时 status=rejected，message 含原因。
        """
        # 幂等：重复 client_order_id 直接返回首单结果
        if req.client_order_id in self._orders:
            return self._orders[req.client_order_id]

        turnover = req.price * req.volume * self.size
        commission = turnover * self.commission_rate

        if req.direction == DIRECTION_LONG:
            cost = turnover + commission
            if cost > self.cash + 1e-6:
                report = OrderReport(req.client_order_id, req.vt_symbol, req.direction,
                                     STATUS_REJECTED, req.volume, message="现金不足")
                self._orders[req.client_order_id] = report
                return report
            self.cash -= cost
            self.positions[req.vt_symbol] = self.positions.get(req.vt_symbol, 0.0) + req.volume
        else:
            holding = self.positions.get(req.vt_symbol, 0.0)
            if req.volume > holding + 1e-6:
                report = OrderReport(req.client_order_id, req.vt_symbol, req.direction,
                                     STATUS_REJECTED, req.volume, message="持仓不足")
                self._orders[req.client_order_id] = report
                return report
            stamp = turnover * self.stamp_duty
            self.cash += turnover - commission - stamp
            new_pos = holding - req.volume
            if new_pos == 0:
                self.positions.pop(req.vt_symbol, None)
            else:
                self.positions[req.vt_symbol] = new_pos

        report = OrderReport(req.client_order_id, req.vt_symbol, req.direction,
                             STATUS_FILLED, req.volume, filled_volume=req.volume,
                             avg_price=req.price)
        self._orders[req.client_order_id] = report
        self.trades.append({
            "client_order_id": req.client_order_id, "vt_symbol": req.vt_symbol,
            "direction": req.direction, "price": req.price, "volume": req.volume,
        })
        return report

    def cancel_order(self, client_order_id: str) -> bool:
        """撤销活跃订单（submitted / partial 状态），已终结订单无法撤销。

        Args:
            client_order_id: 委托 id。

        Returns:
            True 表示成功撤销，False 表示订单不存在或已终结。
        """
        report = self._orders.get(client_order_id)
        if report and report.is_active:
            report.status = "cancelled"
            return True
        return False

    def query_orders(self) -> list[OrderReport]:
        """返回全部历史订单回报列表（包含已成交/已撤/已拒单）。"""
        return list(self._orders.values())

    def query_positions(self) -> dict[str, float]:
        """返回当前持仓快照（vt_symbol → 股数，零持仓标的不含在内）。"""
        return dict(self.positions)

    def query_account(self) -> dict:
        """返回账户资金状态（含可用现金）。"""
        return {"cash": self.cash}
