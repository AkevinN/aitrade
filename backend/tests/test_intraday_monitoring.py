"""
盘中监控决策（Intraday Monitoring Decision）— 调度监控模式与间隔锁定测试（Wave 3）。

- P4：监控推进（due_bar_slot 判定 + 数据滞后重试语义）。
- P5：1d 等价（日频计划路径行为不变）。
- P6：间隔锁定（API 400/404）。
- 示例：监控模式一日回放（假时钟）。

外部 I/O 全部桩化：trigger_fn 注入收集型 stub（返回可控 decision_bar_dt），
存储用 tmp_path，时间用固定 now_fn 注入。
"""

from __future__ import annotations

import tempfile
from datetime import date, datetime, time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.live.calendar import TradingCalendar
from aitrade.live.decision_instant import INTRADAY_BAR_FREQS, bar_close_grid
from aitrade.live.plan_scheduler import PlanScheduler
from aitrade.live.runtime_state import RuntimeStateStore
from aitrade.live.scheduler import due_bar_slot
from aitrade.live.trading_plan import TradingPlan, TradingPlanStore

VT = "000001.SZSE"
DAY = date(2026, 6, 10)  # 周三（交易日）
CAL = TradingCalendar(trading_days=[DAY])


def _plan(bar_freq: str = "30m", **over) -> TradingPlan:
    kwargs = dict(
        plan_id="p1",
        name="监控计划",
        model="m30",
        vt_symbol=VT,
        scheme="intra_v1",
        enabled=True,
        bar_freq=bar_freq,
        trigger_times=[] if bar_freq != "1d" else ["15:05"],
    )
    kwargs.update(over)
    return TradingPlan(**kwargs)


def _scheduler(tmp_path, trigger_fn, now: datetime, calendar=CAL) -> PlanScheduler:
    store = TradingPlanStore(tmp_path / "plans")
    store.save(_plan())
    state = RuntimeStateStore(tmp_path / "state.json")
    box = {"now": now}
    sched = PlanScheduler(
        store, state, trigger_fn, calendar=calendar, now_fn=lambda: box["now"]
    )
    sched._now_box = box  # 测试用：可推进的假时钟
    return sched


def _result(bar_dt: datetime) -> dict:
    return {"decision": {"decision_bar_dt": bar_dt.isoformat()}, "risk_detail": [], "idempotent_hit": False}


# ---------------------------------------------------------------------------
# Property P4: due_bar_slot 判定
# ---------------------------------------------------------------------------
# Feature: intraday-monitoring-decision, Property 4: 监控推进
# 非交易日/开盘前 → None；否则返回网格上 <= now 的最近收盘时刻；
# 已在 triggered_slots → None（同一 slot 至多一次）。
# Validates: Requirements 3.1, 3.2, 3.4
@settings(max_examples=100)
@given(
    freq=st.sampled_from(INTRADAY_BAR_FREQS),
    hh=st.integers(min_value=0, max_value=23),
    mm=st.integers(min_value=0, max_value=59),
    done_latest=st.booleans(),
)
def test_property_p4_due_bar_slot(freq, hh, mm, done_latest):
    now = datetime.combine(DAY, time(hh, mm))
    grid = bar_close_grid(freq)
    reachable = [t for t in grid if now.time() >= t]

    # 非交易日恒为 None
    assert due_bar_slot(now, freq, TradingCalendar(trading_days=[]), set()) is None

    if not reachable:  # 开盘前（当日网格上无 <= now 的收盘时刻）
        assert due_bar_slot(now, freq, CAL, set()) is None
        return

    latest = reachable[-1].strftime("%H:%M")
    assert due_bar_slot(now, freq, CAL, set()) == latest
    # 已完成 → None（同一 slot 至多一次）
    if done_latest:
        assert due_bar_slot(now, freq, CAL, {latest}) is None


# Feature: intraday-monitoring-decision, Property 4: 监控推进（数据滞后不标记 → 重试）
# trigger 结果的 decision_bar_dt < 网格时刻 → 不标记 slot，下一 tick 重复触发；
# decision_bar_dt >= 网格时刻 → 标记，同一 bar 不再触发。
# Validates: Requirements 3.2, 3.3
@settings(max_examples=50)
@given(lag_ticks=st.integers(min_value=1, max_value=4))
def test_property_p4_lagging_data_retries_until_caught_up(lag_ticks):
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path

        tmp_path = Path(tmpdir)
        calls: list[datetime] = []
        data_box = {"bar_dt": datetime.combine(DAY, time(9, 30))}  # 本地数据落后

        def trigger(plan):
            calls.append(data_box["bar_dt"])
            return _result(data_box["bar_dt"])

        now = datetime.combine(DAY, time(10, 0, 30))  # 10:00 bar（30m）已收盘
        sched = _scheduler(tmp_path, trigger, now)

        for _ in range(lag_ticks):
            sched.tick_once()  # 数据未跟上：每 tick 重试
        assert len(calls) == lag_ticks  # 滞后期间每 tick 都重试（不标记）

        data_box["bar_dt"] = datetime.combine(DAY, time(10, 0))  # 数据跟上
        sched.tick_once()
        assert len(calls) == lag_ticks + 1

        sched.tick_once()  # 已标记：同一 bar 不再触发
        assert len(calls) == lag_ticks + 1


# ---------------------------------------------------------------------------
# Property P5: 1d 等价（日频路径行为不变）
# ---------------------------------------------------------------------------
# Feature: intraday-monitoring-decision, Property 5: 1d 等价
# 日频计划仍走 trigger_times + due_slots 路径：触发即标记（不读 trigger 结果），
# 即使桩返回滞后的 decision_bar_dt 也不影响标记（与历史行为逐位一致）。
# Validates: Requirements 3.5, 6.5
def test_property_p5_daily_path_unchanged(tmp_path):
    calls: list[str] = []

    def trigger(plan):
        calls.append(plan.plan_id)
        return _result(datetime.combine(DAY, time(0, 0)))  # 滞后的 bar：1d 路径不关心

    store = TradingPlanStore(tmp_path / "plans")
    store.save(_plan(bar_freq="1d", trigger_times=["15:05"]))
    state = RuntimeStateStore(tmp_path / "state.json")
    sched = PlanScheduler(
        store, state, trigger,
        calendar=CAL, now_fn=lambda: datetime.combine(DAY, time(15, 6)),
    )
    sched.tick_once()
    assert calls == ["p1"]
    sched.tick_once()  # slot 已标记 → 不重复触发（即使桩返回滞后 bar）
    assert calls == ["p1"]
    assert sched.last_triggered_map() == {"p1": DAY.isoformat()}


# ---------------------------------------------------------------------------
# 示例：监控模式一日回放（30m 计划，假时钟步进）
# ---------------------------------------------------------------------------
def test_monitor_mode_one_day_replay(tmp_path):
    calls: list[tuple[str, str]] = []  # (now, decision_bar)
    data_box = {"bar_dt": datetime.combine(DAY, time(10, 0))}

    def trigger(plan):
        calls.append((sched._now_box["now"].strftime("%H:%M"), data_box["bar_dt"].strftime("%H:%M")))
        return _result(data_box["bar_dt"])

    sched = _scheduler(tmp_path, trigger, datetime.combine(DAY, time(9, 0)))

    sched.tick_once()  # 09:00 开盘前 → 不触发
    assert calls == []

    sched._now_box["now"] = datetime.combine(DAY, time(10, 0, 30))
    sched.tick_once()  # 10:00 bar 收盘，数据就绪 → 触发并标记
    assert calls == [("10:00", "10:00")]

    sched._now_box["now"] = datetime.combine(DAY, time(10, 5))
    sched.tick_once()  # 同一 bar → 不再触发
    assert len(calls) == 1

    sched._now_box["now"] = datetime.combine(DAY, time(10, 30, 30))
    sched.tick_once()  # 10:30 bar 收盘但数据滞后（仍是 10:00）→ 触发但不标记
    assert calls[-1] == ("10:30", "10:00")

    sched.tick_once()  # 数据仍滞后 → 重试
    assert len(calls) == 3

    data_box["bar_dt"] = datetime.combine(DAY, time(10, 30))
    sched.tick_once()  # 数据跟上 → 触发并标记
    assert calls[-1] == ("10:30", "10:30") and len(calls) == 4

    data_box["bar_dt"] = datetime.combine(DAY, time(11, 30))
    sched._now_box["now"] = datetime.combine(DAY, time(12, 0))
    sched.tick_once()  # 午休：最新已收盘 bar = 11:30 → 触发一次
    assert calls[-1] == ("12:00", "11:30")

    sched._now_box["now"] = datetime.combine(DAY, time(12, 30))
    sched.tick_once()  # 午休内同一 bar → 不再触发
    assert len(calls) == 5

    data_box["bar_dt"] = datetime.combine(DAY, time(15, 0))
    sched._now_box["now"] = datetime.combine(DAY, time(15, 0, 30))
    sched.tick_once()  # 收盘末 bar
    assert calls[-1] == ("15:00", "15:00")

    sched._now_box["now"] = datetime.combine(DAY, time(16, 0))
    sched.tick_once()  # 收盘后同一 bar → 不再触发
    assert len(calls) == 6


# ---------------------------------------------------------------------------
# Property P6: 间隔锁定（API 400/404）
# ---------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    """隔离 TestClient + 真实 torch checkpoint 桩（仅 train_config，无权重）。"""
    import torch
    from fastapi.testclient import TestClient

    from aitrade.api import live as live_api
    from aitrade.cnn import storage as cnn_storage
    from aitrade.live.decision import DecisionStore
    from aitrade.live.decision_trace import DecisionTraceStore
    from aitrade.main import create_app

    monkeypatch.setenv("AITRADE_SCHEDULER_ENABLED", "false")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    torch.save({"train_config": {"input_interval": "30m"}}, str(models_dir / "m30.pt"))
    torch.save({"train_config": {"input_interval": "d"}}, str(models_dir / "md.pt"))
    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", models_dir)
    monkeypatch.setattr(live_api, "CNN_MODEL_PATH", models_dir)
    monkeypatch.setattr(live_api, "_plan_store", TradingPlanStore(tmp_path / "plans"))
    monkeypatch.setattr(live_api, "_store", DecisionStore(tmp_path / "decisions"))
    monkeypatch.setattr(live_api, "_trace_store", DecisionTraceStore(tmp_path / "decisions"))
    monkeypatch.setattr(live_api, "_runtime_state", RuntimeStateStore(tmp_path / "state.json"))

    app = create_app()
    with TestClient(app) as c:
        yield c


def _plan_body(**over) -> dict:
    body = {
        "name": "计划",
        "model": "m30",
        "vt_symbol": VT,
        "scheme": "intra_v1",
        "portfolio": {"portfolio_value": 1_000_000.0},
        "bar_freq": "30m",
    }
    body.update(over)
    return body


# Feature: intraday-monitoring-decision, Property 6: 间隔锁定
# 模型 interval 与 bar_freq 不匹配 → 400；日内 + 模型缺失 → 404；匹配 → 通过。
# Validates: Requirements 2.1, 2.2
def test_property_p6_interval_lock_on_plan(client):
    # 匹配：30m 模型 + bar_freq=30m → 通过，且 trigger_times 归一化为空（监控模式）
    resp = client.post("/api/live/plans", json=_plan_body())
    assert resp.status_code == 200, resp.text
    assert resp.json()["bar_freq"] == "30m"
    assert resp.json()["trigger_times"] == []

    # 不匹配：日频模型 + bar_freq=30m → 400
    resp = client.post("/api/live/plans", json=_plan_body(model="md"))
    assert resp.status_code == 400
    assert "训练间隔" in resp.json()["detail"]

    # 不匹配：30m 模型 + bar_freq=1d → 400
    resp = client.post("/api/live/plans", json=_plan_body(bar_freq="1d", trigger_times=["15:05"]))
    assert resp.status_code == 400

    # 日内 + 模型缺失 → 404
    resp = client.post("/api/live/plans", json=_plan_body(model="nope"))
    assert resp.status_code == 404

    # 1d + 模型缺失 → 放过（既有宽松行为）
    resp = client.post(
        "/api/live/plans", json=_plan_body(model="nope", bar_freq="1d", trigger_times=["15:05"])
    )
    assert resp.status_code == 200


def test_property_p6_interval_lock_on_manual_decision(client):
    body = {
        "model": "md",
        "vt_symbol": VT,
        "scheme": "s1",
        "portfolio": {"portfolio_value": 1_000_000.0},
        "bar_freq": "30m",
    }
    resp = client.post("/api/live/decision", json=body)
    assert resp.status_code == 400
    assert "训练间隔" in resp.json()["detail"]

    # 不受支持的 bar_freq → 422（pydantic 校验）
    resp = client.post("/api/live/decision", json={**body, "bar_freq": "2h"})
    assert resp.status_code == 422


def test_p6_update_plan_also_locked(client):
    resp = client.post("/api/live/plans", json=_plan_body())
    pid = resp.json()["plan_id"]
    resp = client.put(f"/api/live/plans/{pid}", json=_plan_body(model="md"))
    assert resp.status_code == 400
