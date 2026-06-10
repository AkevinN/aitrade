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
    def __init__(
        self,
        cash: float = 1_000_000.0,
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.001,
        size: float = 1.0,
    ) -> None:
        self.cash = cash
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.size = size
        self.positions: dict[str, float] = {}
        self._orders: dict[str, OrderReport] = {}   # client_order_id -> report
        self.trades: list[dict] = []

    # -- BrokerGateway --
    def send_order(self, req: OrderRequest) -> OrderReport:
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
        report = self._orders.get(client_order_id)
        if report and report.is_active:
            report.status = "cancelled"
            return True
        return False

    def query_orders(self) -> list[OrderReport]:
        return list(self._orders.values())

    def query_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def query_account(self) -> dict:
        return {"cash": self.cash}
