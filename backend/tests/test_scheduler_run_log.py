"""
调度运行日志测试（task-scheduler-observability Wave 2b）。

覆盖：
- 六类 Skip_Reason 确定性用例（disabled / not_trading_day / schedule_gate /
  already_done / degraded / data_lag）+ trigger 与 error 落盘验证
- TSO-3 属性测试：同日重复 tick N 次，同 (plan_id, reason) 恰一条 skip
- TSO-4 属性测试：注入 append 抛错的故障 store → tick 的触发行为与无 run_log 时一致
- record_error 截断 500 字符
- query 只读倒序

fixture 模式复用 test_trading_plan_properties.py 的 _scheduler 模式。
"""

from __future__ import annotations

import tempfile
from datetime import date, datetime
from typing import Any
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aitrade.live.calendar import TradingCalendar
from aitrade.live.jsonl_store import JsonlDayStore
from aitrade.live.plan_scheduler import PlanScheduler, _LAST_TRIGGERED_KEY
from aitrade.live.runtime_state import RuntimeStateStore
from aitrade.live.scheduler_run_log import SchedulerRunLog
from aitrade.live.trading_plan import TradingPlan, TradingPlanStore


# ---------------------------------------------------------------------------
# 公共 fixture 工具
# ---------------------------------------------------------------------------

def _make_plan(
    plan_id: str = "p1",
    *,
    enabled: bool = True,
    trigger_time: str = "15:05",
    trigger_schedule: str = "daily",
) -> TradingPlan:
    plan = TradingPlan(
        plan_id=plan_id,
        name="测试计划",
        model="m1",
        vt_symbol="000001.SZSE",
        scheme="s1",
        portfolio={"portfolio_value": 1_000_000.0},
        risk={},
        enabled=enabled,
        bar_freq="1d",
        trigger_times=[trigger_time],
        notify_channels=[],
    )
    plan.trigger_schedule = trigger_schedule  # type: ignore[attr-defined]
    return plan


def _make_scheduler(
    tmpdir: str,
    plans: list[TradingPlan],
    *,
    now: datetime,
    healthy: bool = True,
    calendar: TradingCalendar | None = None,
    run_log: SchedulerRunLog | None = None,
):
    """构造注入 run_log 的 PlanScheduler，返回 (scheduler, calls, state)。"""
    store = TradingPlanStore(f"{tmpdir}/plans")
    for p in plans:
        store.save(p)
    state = RuntimeStateStore(f"{tmpdir}/state.json")
    calls: list[str] = []

    def _trigger(plan: TradingPlan) -> dict:
        calls.append(plan.plan_id)
        return {}

    sched = PlanScheduler(
        store=store,
        state=state,
        trigger_fn=_trigger,
        calendar=calendar or TradingCalendar(),
        now_fn=lambda: now,
        health_fn=lambda: (healthy, "" if healthy else "degraded unhealthy"),
        run_log=run_log,
    )
    return sched, calls, state


def _make_run_log(tmpdir: str, now: datetime | None = None) -> tuple[SchedulerRunLog, JsonlDayStore]:
    """创建 SchedulerRunLog，注入 now_fn 使写入日期与测试一致。"""
    if now is not None:
        from datetime import timezone
        # 将 naive datetime 视为本地时间，给 store 注入固定 now_fn
        now_fn = lambda: now.replace(tzinfo=timezone.utc)  # noqa: E731
    else:
        now_fn = None
    store = JsonlDayStore(f"{tmpdir}/scheduler_runs", now_fn=now_fn)
    log = SchedulerRunLog(store)
    return log, store


# ---------------------------------------------------------------------------
# 1. disabled：计划停用
# ---------------------------------------------------------------------------

def test_disabled_reason_logged():
    """disabled 计划在当日首次 tick 后记录一条 disabled skip 事件。"""
    now = datetime(2026, 6, 9, 15, 30)  # 交易日
    with tempfile.TemporaryDirectory() as tmpdir:
        run_log, store = _make_run_log(tmpdir, now=now)
        plan = _make_plan(enabled=False)
        sched, calls, _ = _make_scheduler(tmpdir, [plan], now=now, run_log=run_log)
        sched.tick_once()

        assert calls == []  # 未触发
        records = store.read_day(now.date())
        skip_records = [r for r in records if r.get("event") == "skip" and r.get("reason") == "disabled"]
        assert len(skip_records) == 1
        assert skip_records[0]["plan_id"] == plan.plan_id


# ---------------------------------------------------------------------------
# 2. not_trading_day：非交易日
# ---------------------------------------------------------------------------

def test_not_trading_day_reason_logged():
    """非交易日（节假日）日频计划记录 not_trading_day skip。"""
    # 使用只含特定日期的日历，排除 2026-06-06（周六）
    trading_days: list[date] = []  # 空日历 → 任何日期均非交易日
    cal = TradingCalendar(trading_days=trading_days)
    now = datetime(2026, 6, 6, 15, 30)  # 周六，非交易日
    with tempfile.TemporaryDirectory() as tmpdir:
        run_log, store = _make_run_log(tmpdir, now=now)
        plan = _make_plan(enabled=True)
        sched, calls, _ = _make_scheduler(tmpdir, [plan], now=now, calendar=cal, run_log=run_log)
        sched.tick_once()

        assert calls == []  # 非交易日不触发
        records = store.read_day(now.date())
        skip_records = [r for r in records if r.get("event") == "skip" and r.get("reason") == "not_trading_day"]
        assert len(skip_records) == 1
        assert skip_records[0]["plan_id"] == plan.plan_id


# ---------------------------------------------------------------------------
# 3. schedule_gate：周/月闸门不匹配
# ---------------------------------------------------------------------------

def test_schedule_gate_reason_logged():
    """weekly_first 计划在非本周第一个交易日记录 schedule_gate skip。"""
    # 普通周，周二（2026-06-09）不是本周第一个交易日（周一是）
    now = datetime(2026, 6, 9, 15, 30)  # 周二
    cal = TradingCalendar()  # 默认：工作日即交易日
    with tempfile.TemporaryDirectory() as tmpdir:
        run_log, store = _make_run_log(tmpdir, now=now)
        plan = _make_plan(enabled=True, trigger_schedule="weekly_first")
        sched, calls, _ = _make_scheduler(tmpdir, [plan], now=now, calendar=cal, run_log=run_log)
        sched.tick_once()

        assert calls == []  # 非本周第一个交易日 → 不触发
        records = store.read_day(now.date())
        skip_records = [r for r in records if r.get("event") == "skip" and r.get("reason") == "schedule_gate"]
        assert len(skip_records) == 1
        assert skip_records[0]["plan_id"] == plan.plan_id


# ---------------------------------------------------------------------------
# 4. already_done：当日 slot 已触发
# ---------------------------------------------------------------------------

def test_already_done_reason_logged():
    """当日所有 slot 已触发时记录 already_done skip。"""
    now = datetime(2026, 6, 9, 15, 30)  # 交易日，已过 15:05
    with tempfile.TemporaryDirectory() as tmpdir:
        run_log, store = _make_run_log(tmpdir, now=now)
        plan = _make_plan(enabled=True, trigger_time="15:05")
        plan_store = TradingPlanStore(f"{tmpdir}/plans")
        plan_store.save(plan)
        state = RuntimeStateStore(f"{tmpdir}/state.json")
        # 预先标记当日 15:05 已触发
        state.set(_LAST_TRIGGERED_KEY, {
            plan.plan_id: {"date": now.date().isoformat(), "slots": ["15:05"]}
        })
        calls: list[str] = []

        def _trigger(p: TradingPlan) -> dict:
            calls.append(p.plan_id)
            return {}

        sched = PlanScheduler(
            store=plan_store,
            state=state,
            trigger_fn=_trigger,
            now_fn=lambda: now,
            run_log=run_log,
        )
        sched.tick_once()

        assert calls == []  # 当日已触发 → 不再触发
        records = store.read_day(now.date())
        skip_records = [r for r in records if r.get("event") == "skip" and r.get("reason") == "already_done"]
        assert len(skip_records) == 1
        assert skip_records[0]["plan_id"] == plan.plan_id


# ---------------------------------------------------------------------------
# I1 回归：未到时刻不应记 already_done
# ---------------------------------------------------------------------------

def test_i1_morning_tick_before_slot_no_already_done():
    """I1 回归①：交易日上午 tick（未到 slot 时刻、无历史触发）→ 当日文件无 already_done 事件。"""
    # 09:00 tick，slot=15:05，now.time() < 15:05 → 属正常等待，不该记任何 skip
    now = datetime(2026, 6, 9, 9, 0)  # 交易日（周二）上午 09:00
    with tempfile.TemporaryDirectory() as tmpdir:
        run_log, store = _make_run_log(tmpdir, now=now)
        plan = _make_plan(enabled=True, trigger_time="15:05")
        sched, calls, _ = _make_scheduler(tmpdir, [plan], now=now, run_log=run_log)
        sched.tick_once()

        assert calls == []  # 未到时刻 → 不触发
        records = store.read_day(now.date())
        already_done = [r for r in records if r.get("reason") == "already_done"]
        assert already_done == [], (
            f"交易日未到 slot 时刻时不应记录 already_done，实际记录: {already_done}"
        )


def test_i1_after_trigger_tick_records_already_done():
    """I1 回归②：15:05 后已触发，再次 tick → 恰一条 already_done。"""
    now = datetime(2026, 6, 9, 15, 30)  # 已过 15:05
    with tempfile.TemporaryDirectory() as tmpdir:
        run_log, store = _make_run_log(tmpdir, now=now)
        plan = _make_plan(enabled=True, trigger_time="15:05")
        plan_store = TradingPlanStore(f"{tmpdir}/plans")
        plan_store.save(plan)
        state = RuntimeStateStore(f"{tmpdir}/state.json")
        # 预先标记当日 15:05 已触发
        state.set(_LAST_TRIGGERED_KEY, {
            plan.plan_id: {"date": now.date().isoformat(), "slots": ["15:05"]}
        })

        sched = PlanScheduler(
            store=plan_store,
            state=state,
            trigger_fn=lambda p: {},
            now_fn=lambda: now,
            run_log=run_log,
        )
        sched.tick_once()

        records = store.read_day(now.date())
        already_done = [r for r in records if r.get("reason") == "already_done"]
        assert len(already_done) == 1, (
            f"已触发后再 tick 应恰好一条 already_done，实际: {already_done}"
        )
        assert already_done[0]["plan_id"] == plan.plan_id


# ---------------------------------------------------------------------------
# 5. degraded：降级守卫
# ---------------------------------------------------------------------------

def test_degraded_reason_logged():
    """health_fn 返回不健康时记录 degraded skip。"""
    now = datetime(2026, 6, 9, 15, 30)  # 交易日，已过时点
    with tempfile.TemporaryDirectory() as tmpdir:
        run_log, store = _make_run_log(tmpdir, now=now)
        plan = _make_plan(enabled=True)
        sched, calls, _ = _make_scheduler(tmpdir, [plan], now=now, healthy=False, run_log=run_log)
        sched.tick_once()

        assert calls == []  # 降级 → 不触发
        records = store.read_day(now.date())
        skip_records = [r for r in records if r.get("event") == "skip" and r.get("reason") == "degraded"]
        assert len(skip_records) == 1
        assert skip_records[0]["plan_id"] == plan.plan_id


# ---------------------------------------------------------------------------
# 6. data_lag：行情滞后（监控路径）
# ---------------------------------------------------------------------------

def test_data_lag_reason_logged():
    """监控路径下行情滞后时记录 data_lag skip。"""
    # 使用可完整控制触发结果的监控路径
    # bar_freq 需在 INTRADAY_BAR_FREQS 中
    from aitrade.live.decision_instant import INTRADAY_BAR_FREQS

    bar_freq = next(iter(INTRADAY_BAR_FREQS))  # 取第一个日内频率

    now = datetime(2026, 6, 9, 10, 5)  # 交易日 10:05，10:00 bar 已收盘
    # 构造返回过期 bar 的 trigger_fn
    stale_bar_dt = datetime(2026, 6, 9, 9, 30)  # 比网格时刻 10:00 早的 bar

    with tempfile.TemporaryDirectory() as tmpdir:
        run_log, store = _make_run_log(tmpdir, now=now)

        plan = TradingPlan(
            plan_id="p_intraday",
            name="日内计划",
            model="m1",
            vt_symbol="000001.SZSE",
            scheme="s1",
            portfolio={"portfolio_value": 1_000_000.0},
            risk={},
            enabled=True,
            bar_freq=bar_freq,
            trigger_times=["15:05"],
            notify_channels=[],
        )
        plan_store = TradingPlanStore(f"{tmpdir}/plans")
        plan_store.save(plan)
        state = RuntimeStateStore(f"{tmpdir}/state.json")
        trigger_calls: list[str] = []

        def _trigger(p: TradingPlan) -> dict:
            trigger_calls.append(p.plan_id)
            return {"decision": {"decision_bar_dt": stale_bar_dt.isoformat()}}

        cal = TradingCalendar()  # 周一~周五为交易日
        sched = PlanScheduler(
            store=plan_store,
            state=state,
            trigger_fn=_trigger,
            calendar=cal,
            now_fn=lambda: now,
            run_log=run_log,
        )
        sched.tick_once()

        # trigger_fn 被调用了（监控路径先触发再判断行情是否新鲜）
        # data_lag skip 事件应落盘
        records = store.read_day(now.date())
        skip_records = [r for r in records if r.get("event") == "skip" and r.get("reason") == "data_lag"]
        assert len(skip_records) >= 1  # 至少一条 data_lag
        assert skip_records[0]["plan_id"] == plan.plan_id


# ---------------------------------------------------------------------------
# 7. trigger 事件落盘
# ---------------------------------------------------------------------------

def test_trigger_event_logged():
    """正常触发时 trigger 事件落盘，包含 plan_id 与 slot。"""
    now = datetime(2026, 6, 9, 15, 30)  # 交易日，已过 15:05
    with tempfile.TemporaryDirectory() as tmpdir:
        run_log, store = _make_run_log(tmpdir, now=now)
        plan = _make_plan(enabled=True, trigger_time="15:05")
        sched, calls, _ = _make_scheduler(tmpdir, [plan], now=now, run_log=run_log)
        sched.tick_once()

        assert calls == [plan.plan_id]
        records = store.read_day(now.date())
        trigger_records = [r for r in records if r.get("event") == "trigger"]
        assert len(trigger_records) == 1
        assert trigger_records[0]["plan_id"] == plan.plan_id
        assert trigger_records[0]["slot"] == "15:05"


# ---------------------------------------------------------------------------
# 8. error 事件落盘 + 截断 500 字符
# ---------------------------------------------------------------------------

def test_error_event_logged_and_truncated():
    """trigger_fn 抛出异常时 error 事件落盘，且 error 字段截断至 500 字符。"""
    now = datetime(2026, 6, 9, 15, 30)  # 交易日
    with tempfile.TemporaryDirectory() as tmpdir:
        run_log, store = _make_run_log(tmpdir, now=now)
        plan_store = TradingPlanStore(f"{tmpdir}/plans")
        plan = _make_plan(enabled=True, trigger_time="15:05")
        plan_store.save(plan)
        state = RuntimeStateStore(f"{tmpdir}/state.json")

        long_message = "X" * 600  # 超过 500 字符

        def _boom_trigger(p: TradingPlan) -> dict:
            raise RuntimeError(long_message)

        sched = PlanScheduler(
            store=plan_store,
            state=state,
            trigger_fn=_boom_trigger,
            now_fn=lambda: now,
            run_log=run_log,
        )
        sched.tick_once()

        records = store.read_day(now.date())
        error_records = [r for r in records if r.get("event") == "error"]
        assert len(error_records) == 1
        assert error_records[0]["plan_id"] == plan.plan_id
        assert len(error_records[0]["error"]) <= 500


def test_record_error_truncation_direct():
    """直接测试 SchedulerRunLog.record_error 截断 500 字符。"""
    import datetime as _dt
    today = _dt.date(2026, 6, 9)
    fixed_now = datetime(2026, 6, 9, 12, 0)
    with tempfile.TemporaryDirectory() as tmpdir:
        log, store = _make_run_log(tmpdir, now=fixed_now)
        long_error = "E" * 1000
        log.record_error("plan_x", long_error)

        records = store.read_day(today)
        assert len(records) == 1
        assert len(records[0]["error"]) == 500


# ---------------------------------------------------------------------------
# Property TSO-3: 跳过必留痕且当日同因去重
# ---------------------------------------------------------------------------
# Feature: task-scheduler-observability, Property TSO-3: 跳过必留痕且当日同因去重
# 对任意被跳过的启用计划与任意次数的当日重复 tick（N∈[2,20]），
# SchedulerRunLog 当日文件恰好包含一条该 (plan_id, reason) 的 skip 事件。
# Validates: Requirements 3.1, 3.2
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(n_ticks=st.integers(min_value=2, max_value=20))
def test_tso3_skip_dedup_same_day(n_ticks: int):
    """TSO-3：同日重复 tick N 次，同 (plan_id, reason) 恰一条 skip。"""
    # 使用 disabled 原因，每次 tick 都应被去重为一条
    now = datetime(2026, 6, 9, 15, 30)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_log, store = _make_run_log(tmpdir, now=now)
        plan = _make_plan(enabled=False)
        sched, calls, _ = _make_scheduler(tmpdir, [plan], now=now, run_log=run_log)
        for _ in range(n_ticks):
            sched.tick_once()

        records = store.read_day(now.date())
        skip_records = [
            r for r in records
            if r.get("event") == "skip"
            and r.get("plan_id") == plan.plan_id
            and r.get("reason") == "disabled"
        ]
        # 恰好一条（当日同因去重）
        assert len(skip_records) == 1


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(n_ticks=st.integers(min_value=2, max_value=20))
def test_tso3_skip_dedup_schedule_gate(n_ticks: int):
    """TSO-3：weekly_first 闸门跳过，重复 tick N 次恰一条 schedule_gate skip。"""
    now = datetime(2026, 6, 9, 15, 30)  # 周二，非本周第一个交易日
    cal = TradingCalendar()
    with tempfile.TemporaryDirectory() as tmpdir:
        run_log, store = _make_run_log(tmpdir, now=now)
        plan = _make_plan(enabled=True, trigger_schedule="weekly_first")
        sched, calls, _ = _make_scheduler(tmpdir, [plan], now=now, calendar=cal, run_log=run_log)
        for _ in range(n_ticks):
            sched.tick_once()

        records = store.read_day(now.date())
        skip_records = [
            r for r in records
            if r.get("event") == "skip"
            and r.get("plan_id") == plan.plan_id
            and r.get("reason") == "schedule_gate"
        ]
        assert len(skip_records) == 1


# ---------------------------------------------------------------------------
# Property TSO-4: 记录失败不影响调度与任务
# ---------------------------------------------------------------------------
# Feature: task-scheduler-observability, Property TSO-4: 记录失败不影响调度
# 对任意 JsonlDayStore.append 抛异常的情况（注入故障桩），
# 调度判定结果、触发行为、Last_Triggered_Map 状态与无 run_log 时逐位一致。
# Validates: Requirements 3.5, 3.6, 8.2


class _FaultyStore:
    """始终抛出 OSError 的故障 store 桩。"""

    def append(self, event: dict, *, dedup_key: str | None = None) -> bool:
        raise OSError("故障注入：模拟 IO 错误")

    def read_day(self, day: date) -> list[dict]:
        return []

    def read_range(self, *args: Any, **kwargs: Any) -> list[dict]:
        return []


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    trigger_hour=st.integers(min_value=9, max_value=15),
    trigger_minute=st.integers(min_value=0, max_value=59),
    n_plans=st.integers(min_value=1, max_value=3),
)
def test_tso4_faulty_store_does_not_affect_scheduling(
    trigger_hour: int, trigger_minute: int, n_plans: int
):
    """TSO-4：故障 run_log 不影响 trigger_fn 调用次数与 Last_Triggered_Map 状态。

    M3 修复：seed 参数化 trigger_time（9:00-15:59 各 HH:MM）与计划数（1-3），
    让 Hypothesis 真正变异场景。now 固定 15:30 以涵盖「时点已过」情形。
    """
    trigger_time = f"{trigger_hour:02d}:{trigger_minute:02d}"
    now = datetime(2026, 6, 9, 15, 30)  # 交易日，15:30

    plan_ids = [f"p{i}" for i in range(n_plans)]

    with tempfile.TemporaryDirectory() as tmpdir:
        # 场景 A：注入故障 run_log
        faulty_log = SchedulerRunLog(_FaultyStore())  # type: ignore[arg-type]
        plans_a = [_make_plan(pid, enabled=True, trigger_time=trigger_time) for pid in plan_ids]
        sched_a, calls_a, state_a = _make_scheduler(tmpdir + "/a", plans_a, now=now, run_log=faulty_log)
        sched_a.tick_once()  # 不应抛出

    with tempfile.TemporaryDirectory() as tmpdir:
        # 场景 B：无 run_log（None）
        plans_b = [_make_plan(pid, enabled=True, trigger_time=trigger_time) for pid in plan_ids]
        sched_b, calls_b, state_b = _make_scheduler(tmpdir + "/b", plans_b, now=now, run_log=None)
        sched_b.tick_once()

    # 触发行为逐位一致
    assert calls_a == calls_b
    # Last_Triggered_Map 状态一致（_LAST_TRIGGERED_KEY 已在模块顶部导入）
    map_a = state_a.get(_LAST_TRIGGERED_KEY, {})
    map_b = state_b.get(_LAST_TRIGGERED_KEY, {})
    assert map_a == map_b


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    trigger_hour=st.integers(min_value=9, max_value=15),
    trigger_minute=st.integers(min_value=0, max_value=59),
    n_plans=st.integers(min_value=1, max_value=3),
)
def test_tso4_faulty_store_disabled_plan(
    trigger_hour: int, trigger_minute: int, n_plans: int
):
    """TSO-4：故障 run_log，disabled 计划 tick_once 不抛出，行为与无日志时一致。

    M3 修复：参数化 trigger_time 与计划数，让 Hypothesis 真正变异场景。
    """
    trigger_time = f"{trigger_hour:02d}:{trigger_minute:02d}"
    now = datetime(2026, 6, 9, 15, 30)
    plan_ids = [f"p{i}" for i in range(n_plans)]

    with tempfile.TemporaryDirectory() as tmpdir:
        faulty_log = SchedulerRunLog(_FaultyStore())  # type: ignore[arg-type]
        plans_a = [_make_plan(pid, enabled=False, trigger_time=trigger_time) for pid in plan_ids]
        sched_a, calls_a, _ = _make_scheduler(tmpdir + "/a", plans_a, now=now, run_log=faulty_log)
        sched_a.tick_once()  # 不应抛出

    with tempfile.TemporaryDirectory() as tmpdir:
        plans_b = [_make_plan(pid, enabled=False, trigger_time=trigger_time) for pid in plan_ids]
        sched_b, calls_b, _ = _make_scheduler(tmpdir + "/b", plans_b, now=now, run_log=None)
        sched_b.tick_once()

    assert calls_a == calls_b == []


# ---------------------------------------------------------------------------
# query 只读倒序
# ---------------------------------------------------------------------------

def test_query_reverse_order_and_plan_filter():
    """query 返回当日事件倒序，plan_id 过滤生效。"""
    now = datetime(2026, 6, 9, 15, 30)
    today = now.date()
    with tempfile.TemporaryDirectory() as tmpdir:
        log, store = _make_run_log(tmpdir, now=now)
        log.record_skip("p1", "disabled")
        log.record_trigger("p1", "15:05")
        log.record_skip("p2", "schedule_gate")

        # 不过滤
        all_records = log.query(day=today)
        assert len(all_records) == 3
        # 倒序：最后写的在前
        assert all_records[0]["plan_id"] == "p2"

        # plan_id 过滤
        p1_records = log.query(plan_id="p1", day=today)
        assert len(p1_records) == 2
        assert all(r["plan_id"] == "p1" for r in p1_records)

        # limit
        limited = log.query(day=today, limit=1)
        assert len(limited) == 1
