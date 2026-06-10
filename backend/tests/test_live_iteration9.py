"""
迭代 9 验收测试：实盘小资金 + 全风控（审计日志 / 风控闸门网关 / 券商适配桩）。
"""

from __future__ import annotations

import pytest

from aitrade.live.audit import AuditLog
from aitrade.live.gateway import DIRECTION_LONG, STATUS_FILLED, STATUS_REJECTED, OrderRequest
from aitrade.live.guarded_gateway import RiskGuardedGateway
from aitrade.live.live_broker import StubLiveBroker
from aitrade.live.paper_broker import PaperBroker
from aitrade.live.risk import RiskConfig, RiskManager


# ---------------------------------------------------------------------------
# 审计日志
# ---------------------------------------------------------------------------
def test_audit_log_lineage(tmp_path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    sid = "2026-06-09:eod_buy_v1@v3"
    audit.record("signal", {"prob": 0.72}, signal_id=sid, version="v3")
    audit.record("decision", {"action": "buy", "volume": 900}, signal_id=sid, version="v3")
    audit.record("order", {"status": "filled"}, signal_id=sid, version="v3")
    audit.record("order", {"status": "filled"}, signal_id="other")

    assert len(audit.read_all()) == 4
    lineage = audit.for_signal(sid)
    assert [e["event_type"] for e in lineage] == ["signal", "decision", "order"]


# ---------------------------------------------------------------------------
# 风控闸门网关
# ---------------------------------------------------------------------------
def _req(cid="o1"):
    return OrderRequest(cid, "000001.SZSE", DIRECTION_LONG, "open", 100, 10.0)


def test_guarded_gateway_allows_and_audits(tmp_path) -> None:
    audit = AuditLog(tmp_path / "a.jsonl")
    gw = RiskGuardedGateway(PaperBroker(cash=100000), RiskManager(), audit, version="v1")
    rep = gw.send_order(_req())
    assert rep.status == STATUS_FILLED
    events = audit.read_all()
    assert events and events[-1]["event_type"] == "order"


def test_guarded_gateway_kill_switch_blocks(tmp_path) -> None:
    audit = AuditLog(tmp_path / "a.jsonl")
    risk = RiskManager()
    risk.trip_kill_switch()
    gw = RiskGuardedGateway(PaperBroker(cash=100000), risk, audit)
    rep = gw.send_order(_req())
    assert rep.status == STATUS_REJECTED and "kill-switch" in rep.message
    assert audit.read_all()[-1]["event_type"] == "order_rejected"
    # 未实际成交
    assert gw.query_positions() == {}


def test_guarded_gateway_circuit_breaker_blocks() -> None:
    risk = RiskManager(RiskConfig(daily_loss_limit=0.05))
    risk.update_daily_pnl(-6000, 100000)   # 触发熔断
    gw = RiskGuardedGateway(PaperBroker(cash=100000), risk)
    rep = gw.send_order(_req())
    assert rep.status == STATUS_REJECTED and "熔断" in rep.message


def test_guarded_gateway_reconcile_block() -> None:
    gw = RiskGuardedGateway(PaperBroker(cash=100000), RiskManager())
    gw.block_for_reconcile(True)
    rep = gw.send_order(_req())
    assert rep.status == STATUS_REJECTED and "对账" in rep.message


# ---------------------------------------------------------------------------
# 券商适配桩
# ---------------------------------------------------------------------------
def test_stub_live_broker_requires_connector() -> None:
    with pytest.raises(NotImplementedError, match="连接器"):
        StubLiveBroker().send_order(_req())


def test_stub_live_broker_delegates_to_connector() -> None:
    # 注入一个 PaperBroker 充当连接器，验证委托链路形状正确
    connector = PaperBroker(cash=100000)
    broker = StubLiveBroker(connector=connector)
    rep = broker.send_order(_req())
    assert rep.status == STATUS_FILLED
    assert broker.query_positions()["000001.SZSE"] == 100
