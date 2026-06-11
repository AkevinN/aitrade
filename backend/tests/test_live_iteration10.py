"""
迭代 10 验收测试：高可用（单实例锁 / 降级判定 / 运行时状态恢复 + 调度幂等）。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from aitrade.live.calendar import TradingCalendar
from aitrade.live.degradation import decide_trading, is_data_fresh
from aitrade.live.runtime_state import RuntimeStateStore
from aitrade.live.scheduler import due_slots
from aitrade.live.single_instance import SingleInstanceLock


# ---------------------------------------------------------------------------
# 单实例互斥锁
# ---------------------------------------------------------------------------
def test_single_instance_mutual_exclusion(tmp_path) -> None:
    lock_file = tmp_path / "trader.lock"
    a = SingleInstanceLock(lock_file)
    b = SingleInstanceLock(lock_file)
    assert a.acquire() is True
    assert b.acquire() is False     # 第二实例拿不到锁
    a.release()
    assert b.acquire() is True       # 释放后可被接管（主备切换）
    b.release()


def test_single_instance_context_manager(tmp_path) -> None:
    lock_file = tmp_path / "trader.lock"
    with SingleInstanceLock(lock_file):
        other = SingleInstanceLock(lock_file)
        assert other.acquire() is False
    # 退出上下文后锁释放
    after = SingleInstanceLock(lock_file)
    assert after.acquire() is True
    after.release()


# ---------------------------------------------------------------------------
# 降级判定
# ---------------------------------------------------------------------------
def test_data_freshness() -> None:
    now = datetime(2026, 6, 9, 14, 45)
    assert is_data_fresh(now - timedelta(seconds=30), now, 60) is True
    assert is_data_fresh(now - timedelta(seconds=120), now, 60) is False
    assert is_data_fresh(None, now, 60) is False


def test_decide_trading_pauses_on_anomaly() -> None:
    now = datetime(2026, 6, 9, 14, 45)
    fresh = now - timedelta(seconds=10)

    ok, _ = decide_trading(now=now, last_data_time=fresh, max_staleness_seconds=60)
    assert ok is True

    ok, reason = decide_trading(now=now, last_data_time=fresh, max_staleness_seconds=60, healthy=False)
    assert ok is False and "不健康" in reason

    ok, reason = decide_trading(now=now, last_data_time=fresh, max_staleness_seconds=60,
                                reconcile_blocked=True)
    assert ok is False and "对账" in reason

    ok, reason = decide_trading(now=now, last_data_time=now - timedelta(seconds=300),
                                max_staleness_seconds=60)
    assert ok is False and "过期" in reason

    ok, reason = decide_trading(now=now, last_data_time=None, max_staleness_seconds=60)
    assert ok is False and "无行情" in reason


# ---------------------------------------------------------------------------
# 运行时状态恢复 + 调度幂等
# ---------------------------------------------------------------------------
def test_runtime_state_roundtrip(tmp_path) -> None:
    store = RuntimeStateStore(tmp_path / "state.json")
    assert store.get("last_triggered_date") is None
    store.set("last_triggered_date", "2026-06-09")
    assert store.get("last_triggered_date") == "2026-06-09"
    # 新实例读回（模拟重启）
    assert RuntimeStateStore(tmp_path / "state.json").get("last_triggered_date") == "2026-06-09"


def test_scheduler_restart_no_duplicate_trigger(tmp_path) -> None:
    cal = TradingCalendar(trading_days={date(2026, 6, 9)})
    store = RuntimeStateStore(tmp_path / "state.json")
    now = datetime(2026, 6, 9, 14, 45)
    t = time(14, 45)

    # 首次：当日该唤醒时刻 slot 未触发 → due。
    done = set(store.get("triggered_slots", []))
    assert due_slots(now, [t], cal, done) == ["14:45"]
    # 触发后持久化该 slot。
    store.set("triggered_slots", ["14:45"])

    # 模拟重启：从持久化恢复已触发 slot → 当日不再重复触发。
    done2 = set(store.get("triggered_slots", []))
    assert due_slots(now, [t], cal, done2) == []
