"""
迭代 8 验收测试：执行网关 + 模拟盘 + LiveTrader（幂等下单、状态恢复）。
"""

from __future__ import annotations

from aitrade.live.gateway import (
    DIRECTION_LONG,
    STATUS_FILLED,
    STATUS_REJECTED,
    BrokerGateway,
    OrderRequest,
)
from aitrade.live.live_trader import LiveTrader
from aitrade.live.paper_broker import PaperBroker


def test_paper_broker_is_gateway() -> None:
    assert isinstance(PaperBroker(), BrokerGateway)


def test_paper_broker_buy_updates_cash_and_position() -> None:
    bro = PaperBroker(cash=100000, commission_rate=0.0003, stamp_duty=0.001)
    rep = bro.send_order(OrderRequest("o1", "000001.SZSE", DIRECTION_LONG, "open", 100, 10.0))
    assert rep.status == STATUS_FILLED
    assert bro.positions["000001.SZSE"] == 100
    # 现金 = 100000 - 1000 - 1000*0.0003
    assert abs(bro.cash - (100000 - 1000 - 0.3)) < 1e-6


def test_paper_broker_idempotent_client_order_id() -> None:
    bro = PaperBroker(cash=100000)
    req = OrderRequest("dup", "000001.SZSE", DIRECTION_LONG, "open", 100, 10.0)
    r1 = bro.send_order(req)
    cash_after_first = bro.cash
    r2 = bro.send_order(req)   # 重复提交
    assert r1.client_order_id == r2.client_order_id
    assert bro.cash == cash_after_first        # 未二次扣款
    assert bro.positions["000001.SZSE"] == 100  # 未二次建仓


def test_paper_broker_rejects_insufficient_cash() -> None:
    bro = PaperBroker(cash=500)
    rep = bro.send_order(OrderRequest("o1", "X.SZSE", DIRECTION_LONG, "open", 1000, 10.0))
    assert rep.status == STATUS_REJECTED and "现金不足" in rep.message
    assert "X.SZSE" not in bro.positions


def test_paper_broker_rejects_oversell() -> None:
    bro = PaperBroker(cash=100000)
    bro.send_order(OrderRequest("b", "X.SZSE", DIRECTION_LONG, "open", 100, 10.0))
    rep = bro.send_order(OrderRequest("s", "X.SZSE", "short", "close", 200, 10.0))
    assert rep.status == STATUS_REJECTED and "持仓不足" in rep.message


def test_live_trader_rebalance_to_target() -> None:
    bro = PaperBroker(cash=100000)
    trader = LiveTrader(bro)
    # 目标买入 500 股 @10
    reports = trader.rebalance_to_target("rb1", {"000001.SZSE": 500}, {"000001.SZSE": 10.0})
    assert len(reports) == 1 and reports[0].status == STATUS_FILLED
    assert bro.query_positions()["000001.SZSE"] == 500

    # 再次同 rebalance_id → 幂等，不重复下单
    cash_before = bro.cash
    trader.rebalance_to_target("rb1", {"000001.SZSE": 500}, {"000001.SZSE": 10.0})
    assert bro.cash == cash_before


def test_live_trader_reduces_position() -> None:
    bro = PaperBroker(cash=100000)
    trader = LiveTrader(bro)
    trader.rebalance_to_target("rb1", {"X.SZSE": 500}, {"X.SZSE": 10.0})
    # 目标降到 200 → 卖出 300
    trader.rebalance_to_target("rb2", {"X.SZSE": 200}, {"X.SZSE": 10.0})
    assert bro.query_positions()["X.SZSE"] == 200


def test_live_trader_state_recovery_from_gateway() -> None:
    bro = PaperBroker(cash=100000)
    LiveTrader(bro).rebalance_to_target("rb1", {"X.SZSE": 300}, {"X.SZSE": 10.0})
    # 模拟重启：新建 trader，从网关读回权威持仓
    recovered = LiveTrader(bro)
    assert recovered.current_positions() == {"X.SZSE": 300}
    # 基于恢复状态继续调仓到 0 → 全平
    recovered.rebalance_to_target("rb2", {"X.SZSE": 0}, {"X.SZSE": 10.0})
    assert recovered.current_positions() == {}
