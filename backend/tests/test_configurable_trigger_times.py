"""
多唤醒时刻调度属性测试（决策时刻统一后：trigger_times）。

覆盖 effective_trigger_times 归一化、due_slots 多时刻判定、按 slot 同日去重、
重启恢复、跨日重置/旧状态兼容。外部 I/O 注入、存储 tmp_path、时间注入，确定性。
"""

from __future__ import annotations

import tempfile
from datetime import datetime, time

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aitrade.live.calendar import TradingCalendar
from aitrade.live.plan_scheduler import _LAST_TRIGGERED_KEY, PlanScheduler
from aitrade.live.runtime_state import RuntimeStateStore
from aitrade.live.scheduler import due_slots
from aitrade.live.trading_plan import (
    TradingPlan,
    TradingPlanStore,
    effective_trigger_times,
)

hhmm = st.builds(
    lambda h, m: f"{h:02d}:{m:02d}",
    st.integers(min_value=0, max_value=23),
    st.integers(min_value=0, max_value=59),
)
TRADING_DAY = datetime(2026, 6, 9, 23, 59)  # 周二、任意 HH:MM 均已到达


def _make_plan(plan_id: str = "p1", *, enabled: bool = True, trigger_times=None) -> TradingPlan:
    return TradingPlan(
        plan_id=plan_id,
        name="计划",
        model="m1",
        vt_symbol="000001.SZSE",
        scheme="s1",
        portfolio={"portfolio_value": 1_000_000.0},
        risk={},
        enabled=enabled,
        bar_freq="1d",
        trigger_times=list(trigger_times) if trigger_times is not None else ["15:05"],
        notify_channels=["dingtalk"],
    )


def _scheduler(tmpdir, plans, *, now, healthy=True):
    store = TradingPlanStore(f"{tmpdir}/plans")
    for p in plans:
        store.save(p)
    state = RuntimeStateStore(f"{tmpdir}/state.json")
    calls: list[str] = []
    sched = PlanScheduler(
        store=store,
        state=state,
        trigger_fn=lambda plan: calls.append(plan.plan_id),
        now_fn=lambda: now,
        health_fn=lambda: (healthy, "" if healthy else "unhealthy"),
    )
    return sched, calls, state


# ---------------------------------------------------------------------------
# Property CTT-1: 唤醒时刻归一化稳定（去重升序）
# ---------------------------------------------------------------------------
# Feature: decision-instant-unification, Property CTT-1: trigger_times 归一化
# Validates: Requirements 4.1
@settings(max_examples=100)
@given(times=st.lists(hhmm, min_size=1, max_size=5))
def test_property_ctt_1_effective_trigger_times_normalized(times):
    plan = _make_plan(trigger_times=times)
    eff = effective_trigger_times(plan)
    assert eff == sorted(set(eff))
    assert set(eff) == set(times)


# ---------------------------------------------------------------------------
# Property CTT-3: 每个 slot 同日至多触发一次（多时刻）
# ---------------------------------------------------------------------------
# Feature: decision-instant-unification, Property CTT-3: 每个 slot 同日至多一次
# Validates: Requirements 4.2, 4.3
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    times=st.lists(hhmm, min_size=1, max_size=4, unique=True),
    ticks=st.integers(min_value=1, max_value=4),
)
def test_property_ctt_3_each_slot_at_most_once(times, ticks):
    with tempfile.TemporaryDirectory() as tmpdir:
        plan = _make_plan(trigger_times=times)
        sched, calls, state = _scheduler(tmpdir, [plan], now=TRADING_DAY)
        for _ in range(ticks):
            sched.tick_once()
        distinct = sorted(set(times))
        assert len(calls) == len(distinct)
        rec = state.get(_LAST_TRIGGERED_KEY, {})[plan.plan_id]
        assert rec["date"] == TRADING_DAY.date().isoformat()
        assert sorted(rec["slots"]) == distinct


# ---------------------------------------------------------------------------
# Property CTT-4: 按 slot 重启恢复不重复
# ---------------------------------------------------------------------------
# Feature: decision-instant-unification, Property CTT-4: 按 slot 重启恢复
# Validates: Requirements 4.2
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(times=st.lists(hhmm, min_size=2, max_size=4, unique=True), data=st.data())
def test_property_ctt_4_restart_recovery_per_slot(times, data):
    distinct = sorted(set(times))
    already = data.draw(st.sampled_from(distinct))
    with tempfile.TemporaryDirectory() as tmpdir:
        plan = _make_plan(trigger_times=times)
        store = TradingPlanStore(f"{tmpdir}/plans")
        store.save(plan)
        state = RuntimeStateStore(f"{tmpdir}/state.json")
        state.set(
            _LAST_TRIGGERED_KEY,
            {plan.plan_id: {"date": TRADING_DAY.date().isoformat(), "slots": [already]}},
        )
        calls: list[str] = []
        sched = PlanScheduler(
            store=store,
            state=RuntimeStateStore(f"{tmpdir}/state.json"),
            trigger_fn=lambda p: calls.append(p.plan_id),
            now_fn=lambda: TRADING_DAY,
        )
        sched.tick_once()
        assert len(calls) == len(distinct) - 1
        rec = RuntimeStateStore(f"{tmpdir}/state.json").get(_LAST_TRIGGERED_KEY)[plan.plan_id]
        assert sorted(rec["slots"]) == distinct


# ---------------------------------------------------------------------------
# Property CTT-5: 跨日重置 / 旧字符串状态整日抑制
# ---------------------------------------------------------------------------
# Feature: decision-instant-unification, Property CTT-5: 跨日重置 / 旧状态抑制
# Validates: Requirements 4.2
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(times=st.lists(hhmm, min_size=1, max_size=3, unique=True))
def test_property_ctt_5_crossday_reset_and_legacy_suppress(times):
    distinct = sorted(set(times))
    with tempfile.TemporaryDirectory() as tmpdir:
        plan = _make_plan(trigger_times=times)
        sched, calls, state = _scheduler(tmpdir, [plan], now=TRADING_DAY)
        state.set(
            _LAST_TRIGGERED_KEY,
            {plan.plan_id: {"date": "2026-06-08", "slots": list(distinct)}},
        )
        sched.tick_once()
        assert len(calls) == len(distinct)

    with tempfile.TemporaryDirectory() as tmpdir:
        plan = _make_plan(trigger_times=times)
        sched, calls, state = _scheduler(tmpdir, [plan], now=TRADING_DAY)
        state.set(_LAST_TRIGGERED_KEY, {plan.plan_id: TRADING_DAY.date().isoformat()})  # 旧字符串
        sched.tick_once()
        assert calls == []


# ---------------------------------------------------------------------------
# due_slots 纯函数：仅交易日 + 已到达 + 未触发
# ---------------------------------------------------------------------------
# Feature: decision-instant-unification, Property: due_slots 多时刻判定
# Validates: Requirements 4.2
@settings(max_examples=100)
@given(times=st.lists(hhmm, min_size=1, max_size=4, unique=True))
def test_due_slots_returns_passed_untriggered(times):
    cal = TradingCalendar()
    tt = [time(int(t[:2]), int(t[3:])) for t in times]
    got = due_slots(TRADING_DAY, tt, cal, set())
    assert got == sorted(set(times))  # 23:59 全部已到达且未触发
    # 全部已触发 → 空
    assert due_slots(TRADING_DAY, tt, cal, set(times)) == []
