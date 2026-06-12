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
    """把"目标仓位"经 BrokerGateway 落为实际订单的调仓执行器。

    与回测的 execute_trading 同思路（diff = 目标 - 当前 → 买/卖），
    但执行层走 BrokerGateway 接口。幂等键由 (rebalance_id, vt_symbol) 派生，
    重启后以网关 query_positions 为权威持仓来源恢复状态。

    Example:
        >>> trader = LiveTrader(gateway=paper_broker)
        >>> trader.rebalance_to_target("r001", {"000001.SZSE": 1000}, {"000001.SZSE": 12.5})
    """

    def __init__(self, gateway: BrokerGateway, min_volume: int = 100) -> None:
        """初始化 LiveTrader。

        Args:
            gateway:    实现了 BrokerGateway 协议的网关（实盘/模拟盘均可）。
            min_volume: 最小交易手数（股数），低于此值的 diff 忽略不下单，默认 100。
        """
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
        """把当前持仓调整到目标仓位，逐标的下单。

        diff = target - current；diff > 0 买入，diff < 0 卖出；|diff| < min_volume 或
        标的不在 prices 中则跳过。client_order_id 由 (rebalance_id, vt_symbol, 方向) 派生，
        同 rebalance_id 重复调用时幂等（网关层保证不重复成交）。

        Args:
            rebalance_id: 调仓批次幂等键，如 "r001"；用于派生 client_order_id。
            targets:      目标持仓 {vt_symbol: 目标股数}；不在 targets 中的标的目标视为 0。
            prices:       各标的参考价格 {vt_symbol: 价格}；无价格的标的跳过。
            price_add:    限价单价格缓冲率（默认 0）；买入价 = price*(1+price_add)，
                          卖出价 = price*(1-price_add)。

        Returns:
            所有下单的 OrderReport 列表（按合约排序）。
        """
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
