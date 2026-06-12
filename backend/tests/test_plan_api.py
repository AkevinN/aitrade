"""
交易计划自动化 API 路由与生命周期集成测试（任务 5 / 7 / 8）。
"""

from __future__ import annotations

import time
from datetime import datetime

import polars as pl
import pytest
from fastapi.testclient import TestClient

from aitrade.api import live as live_api
from aitrade.live import orchestrator
from aitrade.live.decision import DecisionStore
from aitrade.live.decision_trace import DecisionTraceStore
from aitrade.live.runtime_state import RuntimeStateStore
from aitrade.live.single_instance import SingleInstanceLock
from aitrade.live.trading_plan import TradingPlanStore
from aitrade.main import create_app

VT = "000001.SZSE"


def _plan_body(**over) -> dict:
    body = {
        "name": "测试计划",
        "model": "m1",
        "vt_symbol": VT,
        "scheme": "eod_buy_v1",
        "portfolio": {"portfolio_value": 1_000_000.0},
        "bar_freq": "1d",
        "trigger_times": ["15:05"],
        "notify_channels": [],
    }
    body.update(over)
    return body


@pytest.fixture
def client(tmp_path, monkeypatch):
    """隔离的 TestClient：tmp_path 计划/决策/状态存储 + 桩化外部 I/O。"""
    monkeypatch.setenv("AITRADE_SCHEDULER_ENABLED", "false")  # 测试不启后台线程
    monkeypatch.setattr(live_api, "_plan_store", TradingPlanStore(tmp_path / "plans"))
    monkeypatch.setattr(live_api, "_store", DecisionStore(tmp_path / "decisions"))
    monkeypatch.setattr(live_api, "_trace_store", DecisionTraceStore(tmp_path / "decisions"))
    monkeypatch.setattr(live_api, "_runtime_state", RuntimeStateStore(tmp_path / "state.json"))

    def _stub_predict(**kwargs):
        start = kwargs.get("start")
        return pl.DataFrame({
            "datetime": [datetime.combine(start, datetime.min.time())],
            "vt_symbol": [VT],
            "signal": [0.72],
        })

    monkeypatch.setattr(orchestrator, "predict_cnn_signals", _stub_predict)
    monkeypatch.setattr(
        orchestrator, "_load_close_price",
        lambda vt_symbol, instant: (10.0, "d"),
    )

    app = create_app()
    with TestClient(app) as c:
        yield c


def _poll(c: TestClient, task_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = c.get(f"/api/alpha/tasks/{task_id}").json()
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.01)
    raise AssertionError("任务未完成")


def test_plan_crud_roundtrip(client):
    # 创建
    resp = client.post("/api/live/plans", json=_plan_body())
    assert resp.status_code == 200, resp.text
    plan = resp.json()
    pid = plan["plan_id"]
    assert plan["name"] == "测试计划"
    # 列表
    plans = client.get("/api/live/plans").json()
    assert any(p["plan_id"] == pid for p in plans)
    # 详情
    got = client.get(f"/api/live/plans/{pid}").json()
    assert got["plan_id"] == pid
    # 更新
    upd = client.put(f"/api/live/plans/{pid}", json=_plan_body(name="改名")).json()
    assert upd["name"] == "改名" and upd["plan_id"] == pid
    # 启停
    tog = client.patch(f"/api/live/plans/{pid}/enabled", json={"enabled": True}).json()
    assert tog["enabled"] is True
    # 删除
    assert client.delete(f"/api/live/plans/{pid}").json()["deleted"] is True
    assert client.get(f"/api/live/plans/{pid}").status_code == 404


def test_plan_not_found_404(client):
    assert client.get("/api/live/plans/nope").status_code == 404
    assert client.put("/api/live/plans/nope", json=_plan_body()).status_code == 404
    assert client.delete("/api/live/plans/nope").status_code == 404
    assert client.patch("/api/live/plans/nope/enabled", json={"enabled": True}).status_code == 404
    assert client.post("/api/live/plans/nope/run").status_code == 404


def test_invalid_trigger_time_rejected(client):
    resp = client.post("/api/live/plans", json=_plan_body(trigger_times=["25:99"]))
    assert resp.status_code == 422


def test_unsupported_bar_freq_rejected(client):
    # 仅支持 SUPPORTED_BAR_FREQS（1d + 分钟频）；其它值 422。
    resp = client.post("/api/live/plans", json=_plan_body(bar_freq="2h"))
    assert resp.status_code == 422


def test_intraday_plan_requires_existing_model(client):
    # 日内计划必须能锁定模型训练间隔：模型 checkpoint 不存在 → 404（Req 2.2）。
    resp = client.post("/api/live/plans", json=_plan_body(bar_freq="30m"))
    assert resp.status_code == 404


def test_trigger_times_roundtrip(client):
    # 创建含多唤醒时刻的计划：详情原样返回；摘要含去重升序的生效时刻。
    body = _plan_body(trigger_times=["15:30", "15:05"])
    resp = client.post("/api/live/plans", json=body)
    assert resp.status_code == 200, resp.text
    plan = resp.json()
    pid = plan["plan_id"]
    assert plan["trigger_times"] == ["15:30", "15:05"]  # 持久化原样
    assert plan["bar_freq"] == "1d"
    summary = next(p for p in client.get("/api/live/plans").json() if p["plan_id"] == pid)
    assert summary["trigger_times"] == ["15:05", "15:30"]  # 摘要为生效（去重升序）


def test_run_plan_produces_decision(client):
    pid = client.post("/api/live/plans", json=_plan_body()).json()["plan_id"]
    resp = client.post(f"/api/live/plans/{pid}/run")
    assert resp.status_code == 200
    task = _poll(client, resp.json()["task_id"])
    assert task["status"] == "completed", task.get("message")
    decision = task["result"]["decision"]
    assert decision["action"] in ("buy", "sell", "hold")


def test_scheduler_status(client):
    pid = client.post("/api/live/plans", json=_plan_body()).json()["plan_id"]
    client.patch(f"/api/live/plans/{pid}/enabled", json={"enabled": True})
    status = client.get("/api/live/scheduler/status").json()
    assert "running" in status and "tick_seconds" in status
    assert status["enabled_plan_count"] >= 1


# ---------------------------------------------------------------------------
# 生命周期/调度器冒烟
# ---------------------------------------------------------------------------
def test_scheduler_start_stop(tmp_path):
    from aitrade.live.plan_scheduler import PlanScheduler

    store = TradingPlanStore(tmp_path / "plans")
    state = RuntimeStateStore(tmp_path / "state.json")
    sched = PlanScheduler(store=store, state=state, trigger_fn=lambda p: None, tick_seconds=1.0)
    assert sched.start() is True
    assert sched.is_running() is True
    sched.stop()
    assert sched.is_running() is False


def test_scheduler_lock_occupied_does_not_start(tmp_path):
    from aitrade.live.plan_scheduler import PlanScheduler

    lock_path = tmp_path / "scheduler.lock"
    holder = SingleInstanceLock(lock_path)
    assert holder.acquire() is True  # 先占用锁

    store = TradingPlanStore(tmp_path / "plans")
    state = RuntimeStateStore(tmp_path / "state.json")
    sched = PlanScheduler(store=store, state=state, trigger_fn=lambda p: None, tick_seconds=1.0)
    second_lock = SingleInstanceLock(lock_path)
    assert sched.start(lock=second_lock) is False  # 锁被占用 -> 不启动
    assert sched.is_running() is False
    holder.release()


# ---------------------------------------------------------------------------
# 前后端契约测试：rule 计划真实载荷形态
# ---------------------------------------------------------------------------

def _rule_plan_body(**over) -> dict:
    """前端 rule 模式提交的真实载荷形态（model/vt_symbol/scheme 为空串）。"""
    body = {
        "name": "ETF 动量轮动计划",
        "model": "",
        "vt_symbol": "",
        "scheme": "",
        "bar_freq": "1d",
        "trigger_times": ["15:05"],
        "notify_channels": [],
        "data_source": "pull",
        "enabled": False,
        "buy_threshold": 0,
        "position_ratio": 0,
        "min_volume": 0,
        "model_version": "",
        "portfolio": {"portfolio_value": 0},
        "strategy_type": "rule",
        "signal_source": "etf_momentum",
        "signal_params": {"universe": ["510300.SSE", "510500.SSE"]},
        "trigger_schedule": "daily",
        "portfolio_id": "portfolio-001",
    }
    body.update(over)
    return body


def test_rule_plan_frontend_payload_creates_201(client):
    """前端 rule 模式真实载荷（model/vt_symbol/scheme 为空串）→ 成功落库（200）。"""
    resp = client.post("/api/live/plans", json=_rule_plan_body())
    assert resp.status_code == 200, resp.text
    plan = resp.json()
    assert plan["strategy_type"] == "rule"
    assert plan["signal_source"] == "etf_momentum"
    assert plan["trigger_times"] == ["15:05"]
    assert plan["model"] == ""
    assert plan["vt_symbol"] == ""
    assert plan["scheme"] == ""


def test_rule_plan_missing_trigger_times_rejected(client):
    """rule 计划缺 trigger_times（空列表）→ 422（调度器依赖 trigger_times 确定当日触发时刻）。"""
    resp = client.post("/api/live/plans", json=_rule_plan_body(trigger_times=[]))
    assert resp.status_code == 422, resp.text


def test_rule_plan_missing_signal_source_rejected(client):
    """rule 计划缺 signal_source（空串）→ 422。"""
    resp = client.post("/api/live/plans", json=_rule_plan_body(signal_source=""))
    assert resp.status_code == 422, resp.text


def test_cnn_plan_missing_model_rejected(client):
    """cnn 计划缺 model（空串）→ 422（既有语义零回归）。"""
    resp = client.post("/api/live/plans", json=_plan_body(model=""))
    assert resp.status_code == 422, resp.text


def test_cnn_plan_missing_vt_symbol_rejected(client):
    """cnn 计划缺 vt_symbol（空串）→ 422（既有语义零回归）。"""
    resp = client.post("/api/live/plans", json=_plan_body(vt_symbol=""))
    assert resp.status_code == 422, resp.text


def test_cnn_plan_missing_scheme_rejected(client):
    """cnn 计划缺 scheme（空串）→ 422（既有语义零回归）。"""
    resp = client.post("/api/live/plans", json=_plan_body(scheme=""))
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# 契约测试：GET /plans 摘要含 strategy_type / portfolio_id / signal_source
# ---------------------------------------------------------------------------

def test_cnn_plan_summary_contains_strategy_type(client):
    """cnn 计划的 GET /plans 摘要必须含 strategy_type=='cnn'，portfolio_id/signal_source 为空串。"""
    pid = client.post("/api/live/plans", json=_plan_body()).json()["plan_id"]
    summaries = client.get("/api/live/plans").json()
    summary = next(p for p in summaries if p["plan_id"] == pid)
    assert summary["strategy_type"] == "cnn", f"摘要 strategy_type 应为 cnn，实际: {summary.get('strategy_type')!r}"
    assert summary["portfolio_id"] == "", f"cnn 计划摘要 portfolio_id 应为空串，实际: {summary.get('portfolio_id')!r}"
    assert summary["signal_source"] == "", f"cnn 计划摘要 signal_source 应为空串，实际: {summary.get('signal_source')!r}"


def test_rule_plan_summary_contains_strategy_type_and_portfolio_id(client):
    """rule 计划的 GET /plans 摘要必须含 strategy_type=='rule' 且 portfolio_id 非空。"""
    body = _rule_plan_body()  # portfolio_id="portfolio-001", signal_source="etf_momentum"
    pid = client.post("/api/live/plans", json=body).json()["plan_id"]
    summaries = client.get("/api/live/plans").json()
    summary = next(p for p in summaries if p["plan_id"] == pid)
    assert summary["strategy_type"] == "rule", f"摘要 strategy_type 应为 rule，实际: {summary.get('strategy_type')!r}"
    assert summary["portfolio_id"] == "portfolio-001", f"摘要 portfolio_id 应为 'portfolio-001'，实际: {summary.get('portfolio_id')!r}"
    assert summary["signal_source"] == "etf_momentum", f"摘要 signal_source 应为 'etf_momentum'，实际: {summary.get('signal_source')!r}"
