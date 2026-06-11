"""
迭代 7 验收测试：监控与对账（监控状态 / 心跳健康 / 对账告警阻断）。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from aitrade.live.monitoring import HeartbeatMonitor, MonitorHub
from aitrade.live.reconciliation import reconcile, reconcile_positions


# ---------------------------------------------------------------------------
# 监控状态
# ---------------------------------------------------------------------------
def test_monitor_hub_snapshot() -> None:
    hub = MonitorHub()
    hub.update_position("000001.SZSE", 900)
    hub.update_account(cash=50000, realized_pnl=1200, unrealized_pnl=-200)
    hub.add_trade({"vt_symbol": "000001.SZSE", "price": 10, "volume": 900})
    snap = hub.snapshot()
    assert snap["positions"] == {"000001.SZSE": 900}
    assert snap["cash"] == 50000
    assert snap["total_pnl"] == 1000
    assert snap["trade_count"] == 1
    # 平仓后持仓清除
    hub.update_position("000001.SZSE", 0)
    assert hub.snapshot()["positions"] == {}


# ---------------------------------------------------------------------------
# 心跳健康
# ---------------------------------------------------------------------------
def test_heartbeat_detects_stale() -> None:
    hb = HeartbeatMonitor(timeout_seconds=60)
    t0 = datetime(2026, 6, 9, 14, 0, 0)
    hb.beat("signal_service", t0)
    hb.beat("data_feed", t0)
    # 30s 后都健康
    assert hb.is_healthy(t0 + timedelta(seconds=30)) is True
    # data_feed 未再心跳，90s 后超时
    hb.beat("signal_service", t0 + timedelta(seconds=80))
    now = t0 + timedelta(seconds=90)
    assert "data_feed" in hb.stale_services(now)
    assert hb.is_healthy(now) is False


# ---------------------------------------------------------------------------
# 对账
# ---------------------------------------------------------------------------
def test_reconcile_positions_diff() -> None:
    diffs = reconcile_positions({"A": 100, "B": 200}, {"A": 100, "B": 300})
    assert len(diffs) == 1
    assert diffs[0].vt_symbol == "B" and diffs[0].diff == 100


def test_reconcile_clean_does_not_block() -> None:
    res = reconcile({"A": 100}, {"A": 100}, theoretical_value=1000, actual_value=1000)
    assert res.ok is True
    assert res.should_block is False


def test_reconcile_breach_blocks_auto_trade() -> None:
    res = reconcile(
        {"A": 100, "B": 200}, {"A": 100, "B": 0},   # B 持仓不一致
        theoretical_value=10000, actual_value=9000, value_tolerance=500,
    )
    assert res.ok is False
    assert res.should_block is True
    assert any("持仓不一致" in a for a in res.alerts)
    assert any("市值" in a for a in res.alerts)


def test_reconcile_within_tolerance() -> None:
    res = reconcile(
        {"A": 100}, {"A": 101},
        theoretical_value=10000, actual_value=10100,
        qty_tolerance=1, value_tolerance=200,
    )
    assert res.should_block is False
