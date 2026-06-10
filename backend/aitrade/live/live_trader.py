"""
实盘交易器 LiveTrader（迭代 8）：把"目标仓位"经网关落为订单，幂等。

与回测的 execute_trading 同思路（diff = 目标 - 当前 → 买/卖），但执行层走 BrokerGateway。
幂等：client_order_id 由 (rebalance_id, vt_symbol) 派生，重复 rebalance 不重复下单。
状态恢复：当前持仓以网关 query_positions 为准（权威源），重启后重建一致。
"""

from __future__ import annotations

from .gateway import (
    DIRECTION_LONG,
    DIRECTION_SHORT,
    OFFSET_CLOSE,
    OFFSET_OPEN,
    BrokerGateway,
    OrderReport,
    OrderRequest,
)


class LiveTrader:
    def __init__(self, gateway: BrokerGateway, min_volume: int = 100) -> None:
        self.gateway = gateway
        self.min_volume = min_volume

    def current_positions(self) -> dict[str, float]:
        """权威持仓来自网关（支持重启恢复）。"""
        return self.gateway.query_positions()

    def rebalance_to_target(
        self,
        rebalance_id: str,
        targets: dict[str, float],
        prices: dict[str, float],
        price_add: float = 0.0,
    ) -> list[OrderReport]:
        """把当前持仓调整到目标仓位。rebalance_id 用于幂等。"""
        positions = self.current_positions()
        symbols = set(targets) | set(positions)
        reports: list[OrderReport] = []

        for vt_symbol in sorted(symbols):
            target = float(targets.get(vt_symbol, 0.0))
            current = float(positions.get(vt_symbol, 0.0))
            diff = target - current
            if abs(diff) < self.min_volume or vt_symbol not in prices:
                continue

            price = prices[vt_symbol]
            if diff > 0:
                order_price = price * (1 + price_add)
                req = OrderRequest(
                    client_order_id=f"{rebalance_id}:{vt_symbol}:buy",
                    vt_symbol=vt_symbol, direction=DIRECTION_LONG, offset=OFFSET_OPEN,
                    volume=diff, price=order_price,
                )
            else:
                order_price = price * (1 - price_add)
                req = OrderRequest(
                    client_order_id=f"{rebalance_id}:{vt_symbol}:sell",
                    vt_symbol=vt_symbol, direction=DIRECTION_SHORT, offset=OFFSET_CLOSE,
                    volume=abs(diff), price=order_price,
                )
            reports.append(self.gateway.send_order(req))

        return reports
