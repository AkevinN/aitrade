"""
交易计划自动化后端属性测试（Hypothesis）。

每条正确性属性用单个属性测试实现，`@settings(max_examples=100)`，外部 I/O
（通知通道 httpx、调度触发 trigger_fn、时间 now_fn、健康 health_fn）全部桩化/注入，
存储用临时目录隔离。属性见 .kiro/specs/trading-plan-automation/design.md。
"""

from __future__ import annotations

import ast as _ast
import inspect as _inspect
import tempfile
from datetime import date, datetime, time

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aitrade.live import notifier_channels as nc
from aitrade.live import plan_scheduler as ps_mod
from aitrade.live import trading_plan as tp_mod
from aitrade.live.calendar import TradingCalendar
from aitrade.live.legacy_migration import migrate_plan
from aitrade.live.notifier import LogNotifier, MultiNotifier
from aitrade.live.plan_scheduler import PlanScheduler, _schedule_matches_today
from aitrade.live.runtime_state import RuntimeStateStore
from aitrade.live.trading_plan import TradingPlan, TradingPlanStore
from aitrade.models.trading_plan import TradingPlanRequest

_finite = dict(allow_nan=False, allow_infinity=False)
_SENTINEL = "https://secret.example.com/SUPERSECRETTOKEN_XYZ"

# 合法通道名子集生成器。
channel_names = st.lists(
    st.sampled_from(list(nc.SUPPORTED_CHANNELS)), min_size=0, max_size=4, unique=True
)


def _make_plan(plan_id: str = "p1", *, enabled: bool = True, trigger_time: str = "15:05") -> TradingPlan:
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
        trigger_times=[trigger_time],
        notify_channels=["dingtalk"],
    )


# ---------------------------------------------------------------------------
# Property 1: 通道装配脱敏且无凭证回流
# ---------------------------------------------------------------------------
# Feature: trading-plan-automation, Property 1: 通道装配脱敏且无凭证回流
# 对任意 notify_channels 列表与任意环境变量配置，build_notifier 返回的 Notifier
# 对象及其 repr 文本不含任何 webhook/secret/token；且无任一通道配置凭证时返回 LogNotifier。
# Validates: Requirements 1.5, 1.6, 9.4
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(channels=channel_names, configure=st.booleans())
def test_property_1_build_notifier_no_credential_leak(monkeypatch, channels, configure):
    # 清空所有通道凭证环境变量。
    for _, (env_key, _) in nc._CHANNEL_REGISTRY.items():
        monkeypatch.delenv(env_key, raising=False)

    if configure:
        # 为声明的通道注入哨兵凭证。
        for name in channels:
            env_key = nc._CHANNEL_REGISTRY[name][0]
            monkeypatch.setenv(env_key, _SENTINEL)

    notifier = nc.build_notifier(channels)

    if not configure or not channels:
        # 无可用真实通道 -> 退回 LogNotifier（Req 1.5）。
        assert isinstance(notifier, LogNotifier)
    else:
        assert isinstance(notifier, MultiNotifier)

    # repr 全文（含子通道）不含哨兵凭证（Req 1.6 / 9.4）。
    assert _SENTINEL not in repr(notifier)
    if isinstance(notifier, MultiNotifier):
        assert _SENTINEL not in repr(notifier.channels)


# ---------------------------------------------------------------------------
# Property 2: 多通道失败隔离
# ---------------------------------------------------------------------------
class _StubChannel:
    def __init__(self, ok: bool, *, raises: bool = False) -> None:
        self._ok = ok
        self._raises = raises

    def send(self, title: str, message: str) -> bool:
        if self._raises:
            raise RuntimeError("boom")
        return self._ok


# Feature: trading-plan-automation, Property 2: 多通道失败隔离
# 对任意多通道 Notifier，当任意子集通道 send 抛异常或返回 False 时，只要存在一个
# 成功通道，整体 send 返回 True；任一通道异常不向外传播。
# Validates: Requirements 1.2, 1.7
@settings(max_examples=100)
@given(
    flags=st.lists(
        st.sampled_from(["ok", "fail", "raise"]), min_size=1, max_size=6
    )
)
def test_property_2_multi_notifier_failure_isolation(flags):
    channels = [
        _StubChannel(ok=(f == "ok"), raises=(f == "raise")) for f in flags
    ]
    multi = MultiNotifier(channels)
    result = multi.send("t", "m")  # 不应抛出（隔离）
    assert result == ("ok" in flags)


# ---------------------------------------------------------------------------
# Property 3: 计划持久化往返一致
# ---------------------------------------------------------------------------
@st.composite
def trading_plans(draw) -> TradingPlan:
    return TradingPlan(
        plan_id=draw(st.text(alphabet="abcdef0123456789", min_size=1, max_size=12)),
        name=draw(st.text(min_size=1, max_size=20)),
        model=draw(st.text(alphabet="abc", min_size=1, max_size=6)),
        vt_symbol="000001.SZSE",
        scheme=draw(st.text(alphabet="abc", min_size=1, max_size=6)),
        buy_threshold=draw(st.floats(min_value=0.0, max_value=1.0, **_finite)),
        position_ratio=draw(st.floats(min_value=0.01, max_value=1.0, **_finite)),
        min_volume=draw(st.integers(min_value=1, max_value=1000)),
        data_source=draw(st.sampled_from(["upload", "pull"])),
        enabled=draw(st.booleans()),
        bar_freq="1d",
        trigger_times=draw(st.lists(st.sampled_from(["15:05", "09:30", "14:55"]), min_size=1, max_size=3, unique=True)),
        notify_channels=draw(channel_names),
        portfolio={"portfolio_value": draw(st.floats(min_value=1.0, max_value=1e9, **_finite))},
        risk={"max_total_position_ratio": draw(st.floats(min_value=0.01, max_value=1.0, **_finite))},
    )


# Feature: trading-plan-automation, Property 3: 计划持久化往返一致
# 对任意有效 TradingPlan，经 save 后以同一 plan_id get（含模拟重启重读磁盘）所有字段
# 值与保存时相等；delete 后 get 返回 None。
# Validates: Requirements 2.1, 2.3, 2.4, 2.5
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(plan=trading_plans())
def test_property_3_plan_store_roundtrip(plan):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = TradingPlanStore(tmpdir)
        store.save(plan)
        # 模拟重启：新建 store 实例从磁盘重读。
        reloaded = TradingPlanStore(tmpdir).get(plan.plan_id)
        assert reloaded == plan
        assert TradingPlanStore(tmpdir).delete(plan.plan_id) is True
        assert TradingPlanStore(tmpdir).get(plan.plan_id) is None


# ---------------------------------------------------------------------------
# Property 4: 计划不含通知凭证
# ---------------------------------------------------------------------------
# Feature: trading-plan-automation, Property 4: 计划不含通知凭证
# 对任意经请求映射并持久化的 TradingPlan，其 JSON 序列化文本不含任何 webhook/secret/
# token；notify_channels 仅含合法通道名集合。
# Validates: Requirements 2.9, 9.4
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(channels=channel_names)
def test_property_4_plan_excludes_credentials(monkeypatch, channels):
    # 即便环境注入哨兵凭证，计划仍不应记录。
    for name in nc.SUPPORTED_CHANNELS:
        monkeypatch.setenv(nc._CHANNEL_REGISTRY[name][0], _SENTINEL)

    req = TradingPlanRequest(
        name="计划",
        model="m1",
        vt_symbol="000001.SZSE",
        scheme="s1",
        portfolio={"portfolio_value": 1_000_000.0},
        notify_channels=channels,
    )
    plan = TradingPlan(
        plan_id="p1",
        name=req.name,
        model=req.model,
        vt_symbol=req.vt_symbol,
        scheme=req.scheme,
        portfolio=req.portfolio.model_dump(),
        risk=req.risk.model_dump(),
        notify_channels=list(req.notify_channels),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        store = TradingPlanStore(tmpdir)
        path = store.save(plan)
        raw = path.read_text(encoding="utf-8")

    assert _SENTINEL not in raw
    assert set(plan.notify_channels) <= set(nc.SUPPORTED_CHANNELS)


# ---------------------------------------------------------------------------
# 调度器属性公共工具
# ---------------------------------------------------------------------------
def _scheduler(tmpdir, plans, *, now, healthy=True, calendar=None):
    """构造注入收集型 trigger_fn 的 PlanScheduler，返回 (scheduler, calls, state)。"""
    # 计划目录与运行时状态文件须分离（生产环境亦然），避免 state.json 被 *.json 误读。
    store = TradingPlanStore(f"{tmpdir}/plans")
    for p in plans:
        store.save(p)
    state = RuntimeStateStore(f"{tmpdir}/state.json")
    calls: list[str] = []
    sched = PlanScheduler(
        store=store,
        state=state,
        trigger_fn=lambda plan: calls.append(plan.plan_id),
        calendar=calendar or TradingCalendar(),
        now_fn=lambda: now,
        health_fn=lambda: (healthy, "" if healthy else "unhealthy"),
    )
    return sched, calls, state


weekdaydatetimes = st.datetimes(
    min_value=datetime(2026, 1, 1), max_value=datetime(2026, 12, 31)
)
trigger_time_strs = st.builds(
    lambda h, m: f"{h:02d}:{m:02d}",
    st.integers(min_value=0, max_value=23),
    st.integers(min_value=0, max_value=59),
)


# ---------------------------------------------------------------------------
# Property 5: 单时刻调度判定（交易日 + 已到达 + 当日未触发）
# ---------------------------------------------------------------------------
# Feature: trading-plan-automation, Property 5: 单时刻调度判定
# 对任意时刻/唤醒时刻/日历/最近触发日，PlanScheduler 对单时刻启用计划是否触发，
# 等价于「交易日 且 now>=t 且 当日未触发」（旧 should_trigger 语义，按 slot 化）。
# Validates: Requirements 4.2, 4.3
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    now=weekdaydatetimes,
    dt_str=trigger_time_strs,
    last_trig=st.one_of(st.none(), weekdaydatetimes.map(lambda d: d.date())),
)
def test_property_5_single_time_trigger_decision(now, dt_str, last_trig):
    with tempfile.TemporaryDirectory() as tmpdir:
        plan = _make_plan(enabled=True, trigger_time=dt_str)
        sched, calls, state = _scheduler(tmpdir, [plan], now=now, healthy=True)
        if last_trig is not None:
            # 旧字符串值：== 今日 → 整日抑制；否则不影响（兼容读取）。
            state.set(ps_mod._LAST_TRIGGERED_KEY, {plan.plan_id: last_trig.isoformat()})

        cal = TradingCalendar()
        hh, mm = (int(x) for x in dt_str.split(":"))
        expected = (
            cal.is_trading_day(now.date())
            and now.time() >= time(hh, mm)
            and last_trig != now.date()
        )

        sched.tick_once()
        assert (calls == [plan.plan_id]) == expected


# ---------------------------------------------------------------------------
# Property 6: 调度同日幂等
# ---------------------------------------------------------------------------
# Feature: trading-plan-automation, Property 6: 调度同日幂等（按 slot 细化）
# 对任意仅含单一生效时点的启用计划，在同一交易日内的多次 tick_once，至多触发一次该计划；
# 触发后 Last_Triggered_Map 记为 {date, slots:[该时点]}（去重粒度由「按日」细化为「按 slot」，
# 见 configurable-trigger-times Property 6 细化）。
# Validates: Requirements 5.1, 5.2
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ticks=st.integers(min_value=2, max_value=5))
def test_property_6_same_day_idempotent(ticks):
    # 固定一个交易日（周二）且已过决策时点，保证首个 tick 必触发。
    now = datetime(2026, 6, 9, 15, 30)  # 周二
    with tempfile.TemporaryDirectory() as tmpdir:
        plan = _make_plan(enabled=True, trigger_time="15:05")
        sched, calls, state = _scheduler(tmpdir, [plan], now=now, healthy=True)
        for _ in range(ticks):
            sched.tick_once()
        assert len(calls) == 1  # 单时点同日至多一次
        rec = state.get(ps_mod._LAST_TRIGGERED_KEY, {})[plan.plan_id]
        assert rec == {"date": now.date().isoformat(), "slots": ["15:05"]}


# ---------------------------------------------------------------------------
# Property 7: 重启恢复不重复触发
# ---------------------------------------------------------------------------
# Feature: trading-plan-automation, Property 7: 重启恢复不重复触发
# 对任意当日已触发并写入 Last_Triggered_Map 的计划，用新建 PlanScheduler（从同一
# RuntimeStateStore 恢复）在当日继续 tick_once 时不再触发。
# Validates: Requirements 5.3
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(_seed=st.integers())
def test_property_7_restart_recovery_no_duplicate(_seed):
    now = datetime(2026, 6, 9, 15, 30)
    with tempfile.TemporaryDirectory() as tmpdir:
        plan = _make_plan(enabled=True, trigger_time="15:05")
        # 预置当日已触发。
        state = RuntimeStateStore(f"{tmpdir}/state.json")
        store = TradingPlanStore(f"{tmpdir}/plans")
        store.save(plan)
        state.set(ps_mod._LAST_TRIGGERED_KEY, {plan.plan_id: now.date().isoformat()})
        # 新建调度器从同一 state 恢复。
        calls: list[str] = []
        sched = PlanScheduler(
            store=store,
            state=RuntimeStateStore(f"{tmpdir}/state.json"),
            trigger_fn=lambda p: calls.append(p.plan_id),
            now_fn=lambda: now,
        )
        sched.tick_once()
        assert calls == []  # 重启后当日不再触发


# ---------------------------------------------------------------------------
# Property 8: 停用计划不触发
# ---------------------------------------------------------------------------
# Feature: trading-plan-automation, Property 8: 停用计划不触发
# 对任意 enabled=false 的计划，无论时刻与日历如何，tick_once 不触发它。
# Validates: Requirements 4.5
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(now=weekdaydatetimes, dt_str=trigger_time_strs)
def test_property_8_disabled_plan_never_triggers(now, dt_str):
    with tempfile.TemporaryDirectory() as tmpdir:
        plan = _make_plan(enabled=False, trigger_time=dt_str)
        sched, calls, _state = _scheduler(tmpdir, [plan], now=now, healthy=True)
        sched.tick_once()
        assert calls == []


# ---------------------------------------------------------------------------
# Property 9: 降级时跳过且不记当日已触发
# ---------------------------------------------------------------------------
# Feature: trading-plan-automation, Property 9: 降级时跳过且不记当日已触发
# 对任意满足 should_trigger 的启用计划，当 decide_trading 判定为暂停（不健康）时，
# 不调用 trigger_fn，且不在 Last_Triggered_Map 记当日（使恢复正常后同日仍可触发）。
# Validates: Requirements 7.1
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(_seed=st.integers())
def test_property_9_degradation_skips_without_marking(_seed):
    now = datetime(2026, 6, 9, 15, 30)  # 周二、已过时点 -> should_trigger True
    with tempfile.TemporaryDirectory() as tmpdir:
        plan = _make_plan(enabled=True, trigger_time="15:05")
        sched, calls, state = _scheduler(tmpdir, [plan], now=now, healthy=False)
        sched.tick_once()
        # 降级：未触发、未记当日。
        assert calls == []
        mapping = state.get(ps_mod._LAST_TRIGGERED_KEY, {}) or {}
        assert plan.plan_id not in mapping


# ---------------------------------------------------------------------------
# Property 10: 单计划异常隔离不影响其它计划与线程
# ---------------------------------------------------------------------------
# Feature: trading-plan-automation, Property 10: 单计划异常隔离不影响其它计划与线程
# 对任意一组启用计划，当其中任一计划 trigger_fn 抛异常时，同一 tick_once 中其它计划
# 仍被正常判定与触发，且 tick_once 不向外抛异常。
# Validates: Requirements 7.3
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(bad_index=st.integers(min_value=0, max_value=2))
def test_property_10_per_plan_exception_isolation(bad_index):
    now = datetime(2026, 6, 9, 15, 30)
    with tempfile.TemporaryDirectory() as tmpdir:
        plans = [_make_plan(f"p{i}", enabled=True, trigger_time="15:05") for i in range(3)]
        store = TradingPlanStore(f"{tmpdir}/plans")
        for p in plans:
            store.save(p)
        state = RuntimeStateStore(f"{tmpdir}/state.json")
        triggered: list[str] = []

        def _trigger(plan):
            if plan.plan_id == f"p{bad_index}":
                raise RuntimeError("plan trigger boom")
            triggered.append(plan.plan_id)

        sched = PlanScheduler(
            store=store, state=state, trigger_fn=_trigger, now_fn=lambda: now
        )
        sched.tick_once()  # 不应抛出
        # 其余两个计划仍被触发。
        expected = {f"p{i}" for i in range(3)} - {f"p{bad_index}"}
        assert set(triggered) == expected


# 注：原 Property 11（closed_t 时点校验）已随 data_basis 删除而移除——无前视改由 as_of
# 截断在 decision_instant.select_decision_bar 结构性保证（见 test_decision_instant DI-2/DI-3）。


# ---------------------------------------------------------------------------
# Property 12: 无券商下单路径
# ---------------------------------------------------------------------------
_FORBIDDEN_ORDER_TOKENS = ["broker", "submit_order", "place_order", "send_order", "gateway", "order"]
_NO_BROKER_MODULES = [ps_mod, nc, tp_mod]


def _imported_module_names(module) -> list[str]:
    tree = _ast.parse(_inspect.getsource(module))
    names: list[str] = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, _ast.ImportFrom):
            base = node.module or ""
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names)
    return names


# Feature: trading-plan-automation, Property 12: 无券商下单路径
# 对任意计划 CRUD / 按计划触发 / 自动调度执行路径，系统不调用任何券商网关/下单接口；
# 调度器/通道/计划模块的 import 不含下单相关 token。
# Validates: Requirements 9.1, 9.2
@settings(max_examples=1)
@given(_seed=st.just(0))
def test_property_12_no_broker_order_path(_seed):
    for module in _NO_BROKER_MODULES:
        for mod_name in _imported_module_names(module):
            lowered = mod_name.lower()
            for token in _FORBIDDEN_ORDER_TOKENS:
                assert token not in lowered, (
                    f"{module.__name__} 不应 import 下单相关模块: {mod_name}"
                )


# ---------------------------------------------------------------------------
# Task 3.1 / 3.6 新增属性测试
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Property 13: 旧版 JSON（无 v2 字段）读取 → 默认值正确，磁盘回写收敛
# ---------------------------------------------------------------------------
# Phase 3 M2 Task 3.1：旧计划无 strategy_type/signal_source 等 → migrate 后
# 这些字段均取新默认值，再次 migrate 幂等（值不再变化）。
def test_property_13_legacy_json_v2_fields_defaults():
    """旧计划 JSON 读取后新字段均为合法默认值，且 migrate 幂等。"""
    # 最小旧版 JSON（无 v2 字段，只有 v1 必需字段）
    old_raw: dict = {
        "plan_id": "p_legacy",
        "name": "旧计划",
        "model": "m1",
        "vt_symbol": "000001.SZSE",
        "scheme": "s1",
        "bar_freq": "1d",
        "trigger_times": ["15:05"],
    }
    migrated = migrate_plan(old_raw)
    # 新字段默认值
    assert migrated["strategy_type"] == "cnn"
    assert migrated["signal_source"] == ""
    assert migrated["signal_params"] == {}
    assert migrated["trigger_schedule"] == "daily"
    assert migrated["portfolio_id"] == ""
    # migrate 幂等：对已迁移结果再次 migrate 结果不变
    again = migrate_plan(migrated)
    assert again == migrated


# ---------------------------------------------------------------------------
# Property 14: migrate_plan 幂等（对新结构任意调用次数，结果稳定）
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    strat=st.sampled_from(["cnn", "rule"]),
    src=st.text(alphabet="abcdef_", min_size=0, max_size=12),
    sched=st.sampled_from(["daily", "weekly_first", "monthly_first"]),
)
def test_property_14_migrate_plan_idempotent(strat, src, sched):
    """migrate_plan 对任意含 v2 字段的 plan 幂等。"""
    raw = {
        "plan_id": "px",
        "name": "n",
        "model": "m",
        "vt_symbol": "000001.SZSE",
        "scheme": "s",
        "bar_freq": "1d",
        "trigger_times": ["15:05"],
        "strategy_type": strat,
        "signal_source": src,
        "signal_params": {},
        "trigger_schedule": sched,
        "portfolio_id": "",
    }
    once = migrate_plan(raw)
    twice = migrate_plan(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Property 15: _schedule_matches_today 闸门行为（确定性用例）
# ---------------------------------------------------------------------------
def _cal_from_trading_days(days: list[date]) -> TradingCalendar:
    return TradingCalendar(trading_days=days)


def test_property_15a_daily_always_passes():
    """strategy_type='cnn'，trigger_schedule='daily' → 闸门恒真。"""
    plan = _make_plan(trigger_time="15:05")
    # 给 plan 加 trigger_schedule
    plan.trigger_schedule = "daily"  # type: ignore[attr-defined]
    cal = TradingCalendar()
    for d in [date(2026, 6, 9), date(2026, 6, 10), date(2026, 6, 11)]:
        assert _schedule_matches_today(plan, d, cal) is True


def test_property_15b_weekly_first_mon_only_trading():
    """周一是本周唯一交易日 → weekly_first 在周一通过，周二不通过（非交易日忽略）。

    构造：本周仅周一（2026-06-08）为交易日。
    """
    trading_days = [date(2026, 6, 8)]  # 仅周一
    cal = _cal_from_trading_days(trading_days)
    plan = _make_plan(trigger_time="15:05")
    plan.trigger_schedule = "weekly_first"  # type: ignore[attr-defined]
    # 周一是本周第一个（也是唯一）交易日 → True
    assert _schedule_matches_today(plan, date(2026, 6, 8), cal) is True
    # 周三不是交易日 → False（today 本身不是交易日）
    assert _schedule_matches_today(plan, date(2026, 6, 10), cal) is False


def test_property_15c_weekly_first_holiday_mon_tue_is_first():
    """周一是节假日，周二是本周第一个交易日 → weekly_first 在周二通过。

    构造：2026-06-08（周一）是节假日；2026-06-09（周二）是交易日。
    """
    trading_days = [date(2026, 6, 9), date(2026, 6, 10), date(2026, 6, 11),
                    date(2026, 6, 12)]  # 周二~周五
    cal = _cal_from_trading_days(trading_days)
    plan = _make_plan(trigger_time="15:05")
    plan.trigger_schedule = "weekly_first"  # type: ignore[attr-defined]
    # 周一不是交易日 → False（today 自身不是交易日）
    assert _schedule_matches_today(plan, date(2026, 6, 8), cal) is False
    # 周二是本周第一个交易日 → True
    assert _schedule_matches_today(plan, date(2026, 6, 9), cal) is True
    # 周三不是本周第一个交易日（周二已是）→ False
    assert _schedule_matches_today(plan, date(2026, 6, 10), cal) is False


def test_property_15d_monthly_first():
    """月初第一个交易日。

    构造：2026-07-01（周三）节假日，2026-07-02（周四）是7月第一个交易日。
    """
    trading_days = [date(2026, 7, 2), date(2026, 7, 3), date(2026, 7, 6)]
    cal = _cal_from_trading_days(trading_days)
    plan = _make_plan(trigger_time="15:05")
    plan.trigger_schedule = "monthly_first"  # type: ignore[attr-defined]
    # 7月1日不是交易日 → False
    assert _schedule_matches_today(plan, date(2026, 7, 1), cal) is False
    # 7月2日是本月第一个交易日 → True
    assert _schedule_matches_today(plan, date(2026, 7, 2), cal) is True
    # 7月3日不是第一个 → False
    assert _schedule_matches_today(plan, date(2026, 7, 3), cal) is False


def test_property_15e_weekly_first_normal_week():
    """普通周：周一是交易日 → weekly_first 在周一通过，周二不通过。"""
    # 全周均为交易日（默认日历，不过节假日）
    cal = TradingCalendar()  # 默认：工作日即交易日
    plan = _make_plan(trigger_time="15:05")
    plan.trigger_schedule = "weekly_first"  # type: ignore[attr-defined]
    mon = date(2026, 6, 8)   # 周一
    tue = date(2026, 6, 9)   # 周二
    assert _schedule_matches_today(plan, mon, cal) is True
    assert _schedule_matches_today(plan, tue, cal) is False


# ---------------------------------------------------------------------------
# Property 16: rule 计划缺 signal_source → 请求体校验 422
# ---------------------------------------------------------------------------
def test_property_16_rule_plan_missing_signal_source_rejected():
    """strategy_type='rule' 且 signal_source 为空 → pydantic 校验 ValueError。"""
    import pytest as _pytest
    from pydantic import ValidationError
    with _pytest.raises(ValidationError):
        TradingPlanRequest(
            name="规则计划",
            model="m1",
            vt_symbol="000001.SZSE",
            scheme="s1",
            portfolio={"portfolio_value": 1_000_000.0},
            strategy_type="rule",
            signal_source="",  # 空 → 应 422
        )


def test_property_16b_rule_plan_with_signal_source_ok():
    """strategy_type='rule' 且 signal_source 非空 → 校验通过。"""
    req = TradingPlanRequest(
        name="规则计划",
        model="m1",
        vt_symbol="000001.SZSE",
        scheme="s1",
        portfolio={"portfolio_value": 1_000_000.0},
        strategy_type="rule",
        signal_source="etf_momentum",
    )
    assert req.strategy_type == "rule"
    assert req.signal_source == "etf_momentum"


def test_property_16c_invalid_trigger_schedule_rejected():
    """trigger_schedule 非法值 → pydantic 校验 ValueError。"""
    import pytest as _pytest
    from pydantic import ValidationError
    with _pytest.raises(ValidationError):
        TradingPlanRequest(
            name="计划",
            model="m1",
            vt_symbol="000001.SZSE",
            scheme="s1",
            portfolio={"portfolio_value": 1_000_000.0},
            trigger_schedule="weekly",  # 非法值（应为 weekly_first）
        )


# ---------------------------------------------------------------------------
# Property 17: 周/月闸门与调度器集成 — 非触发日计划不执行
# ---------------------------------------------------------------------------
def test_property_17_weekly_gate_non_first_day_skipped():
    """weekly_first 计划在非本周第一个交易日不触发（调度器层集成）。"""
    # 周二（2026-06-09）→ 普通周，周一（06-08）才是第一个交易日
    now = datetime(2026, 6, 9, 15, 30)  # 周二
    cal = TradingCalendar()  # 默认：工作日即交易日
    with tempfile.TemporaryDirectory() as tmpdir:
        plan = _make_plan(enabled=True, trigger_time="15:05")
        plan.trigger_schedule = "weekly_first"  # type: ignore[attr-defined]
        sched, calls, _state = _scheduler(tmpdir, [plan], now=now, calendar=cal)
        sched.tick_once()
        assert calls == []  # 周二不是本周第一个交易日 → 不触发


def test_property_17b_weekly_gate_first_day_triggers():
    """weekly_first 计划在本周第一个交易日触发。"""
    now = datetime(2026, 6, 8, 15, 30)  # 周一（本周第一个交易日）
    cal = TradingCalendar()
    with tempfile.TemporaryDirectory() as tmpdir:
        plan = _make_plan(enabled=True, trigger_time="15:05")
        plan.trigger_schedule = "weekly_first"  # type: ignore[attr-defined]
        sched, calls, _state = _scheduler(tmpdir, [plan], now=now, calendar=cal)
        sched.tick_once()
        assert calls == [plan.plan_id]  # 周一是本周第一个交易日 → 触发
