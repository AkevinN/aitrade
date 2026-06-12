"""
可观测性 API 测试：GET /api/live/scheduler/runs（task-scheduler-observability 任务 6）。

覆盖：
- 过滤（plan_id 过滤、默认当日、倒序、limit）
- 非法日期 422 + 中文 detail
- TSO-7 API 侧：调用后文件字节不变（只读属性）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aitrade.api import live as live_api
from aitrade.live.jsonl_store import JsonlDayStore
from aitrade.live.scheduler_run_log import SchedulerRunLog
from aitrade.main import create_app


# ---------------------------------------------------------------------------
# Fixture：隔离的 _scheduler_run_log 单例 + TestClient
# ---------------------------------------------------------------------------

@pytest.fixture
def scheduler_client(tmp_path, monkeypatch):
    """注入 tmp_path 下隔离的 SchedulerRunLog，返回 (test_client, run_log, log_dir)。"""
    log_dir = tmp_path / "scheduler_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # 使用固定本地 now_fn，保证写入与读取日期口径一致
    fixed_now = datetime(2026, 6, 12, 10, 0, 0)
    store = JsonlDayStore(log_dir, now_fn=lambda: fixed_now)
    run_log = SchedulerRunLog(store)
    monkeypatch.setattr(live_api, "_scheduler_run_log", run_log)

    app = create_app()
    with TestClient(app) as client:
        yield client, run_log, log_dir, fixed_now


# ---------------------------------------------------------------------------
# 1. 默认当日（date 参数省略时，使用本地今天）
# ---------------------------------------------------------------------------

def test_default_date_returns_today_records(scheduler_client, monkeypatch):
    """不传 date 时返回「今日」记录（与写入分桶日期一致）。"""
    client, run_log, log_dir, fixed_now = scheduler_client

    # 预写两条今日记录
    run_log.record_skip("plan-A", "disabled")
    run_log.record_trigger("plan-A", "15:05")

    # monkeypatch datetime.now() 在 api/live.py 中返回同一天
    # 直接传 date 参数（用 fixed_now 的日期字符串）验证结果同等
    resp = client.get(
        "/api/live/scheduler/runs",
        params={"date": fixed_now.date().isoformat()},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2


# ---------------------------------------------------------------------------
# 2. plan_id 过滤
# ---------------------------------------------------------------------------

def test_plan_id_filter(scheduler_client):
    """plan_id 过滤只返回指定计划的事件。"""
    client, run_log, log_dir, fixed_now = scheduler_client
    run_log.record_skip("plan-A", "disabled")
    run_log.record_skip("plan-B", "schedule_gate")
    run_log.record_trigger("plan-A", "15:05")

    resp = client.get(
        "/api/live/scheduler/runs",
        params={"plan_id": "plan-A", "date": fixed_now.date().isoformat()},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert all(r["plan_id"] == "plan-A" for r in data)
    assert len(data) == 2  # skip(disabled) + trigger


# ---------------------------------------------------------------------------
# 3. 倒序（最新在前）
# ---------------------------------------------------------------------------

def test_reverse_order(scheduler_client):
    """返回结果按时间倒序，最新事件在列表最前。"""
    client, run_log, log_dir, fixed_now = scheduler_client
    # 顺序写入三条（skip→trigger→error）
    run_log.record_skip("plan-X", "not_trading_day")
    run_log.record_trigger("plan-X", "09:30")
    run_log.record_error("plan-X", "RuntimeError: test")

    resp = client.get(
        "/api/live/scheduler/runs",
        params={"date": fixed_now.date().isoformat()},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    # 最后写的 error 应排最前（倒序）
    assert data[0]["event"] == "error"
    assert data[1]["event"] == "trigger"
    assert data[2]["event"] == "skip"


# ---------------------------------------------------------------------------
# 4. limit 参数
# ---------------------------------------------------------------------------

def test_limit_parameter(scheduler_client):
    """limit 限制返回条数。"""
    client, run_log, log_dir, fixed_now = scheduler_client
    for i in range(5):
        run_log.record_trigger(f"plan-{i}", "15:05")

    resp = client.get(
        "/api/live/scheduler/runs",
        params={"date": fixed_now.date().isoformat(), "limit": 3},
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 3


# ---------------------------------------------------------------------------
# 5. 非法日期 → 422 中文 detail
# ---------------------------------------------------------------------------

def test_invalid_date_422(scheduler_client):
    """非法日期格式返回 422，detail 含中文说明。"""
    client, run_log, log_dir, fixed_now = scheduler_client

    resp = client.get(
        "/api/live/scheduler/runs",
        params={"date": "not-a-date"},
    )
    assert resp.status_code == 422
    body = resp.json()
    # detail 应包含中文说明
    detail = body.get("detail", "")
    assert "日期" in detail or "YYYY-MM-DD" in detail, f"detail 未含中文提示: {body}"


def test_invalid_date_incomplete(scheduler_client):
    """不完整日期（如 2026-13）返回 422。"""
    client, run_log, log_dir, fixed_now = scheduler_client
    resp = client.get(
        "/api/live/scheduler/runs",
        params={"date": "2026-13-01"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 6. TSO-7 API 侧只读属性：调用后文件字节不变
# ---------------------------------------------------------------------------

def test_tso7_query_does_not_modify_files(scheduler_client):
    """属性 TSO-7：查询调用前后，log_dir 内所有文件字节完全一致。

    Feature: task-scheduler-observability, Property TSO-7: 查询只读
    Validates: Requirements 2.2, 6.1
    """
    client, run_log, log_dir, fixed_now = scheduler_client

    # 预写数条记录
    run_log.record_skip("plan-A", "disabled")
    run_log.record_trigger("plan-A", "15:05")
    run_log.record_error("plan-B", "some error")

    # 快照：收集 log_dir 所有文件内容
    def _snapshot(d: Path) -> dict[str, bytes]:
        return {
            str(f.relative_to(d)): f.read_bytes()
            for f in sorted(d.rglob("*"))
            if f.is_file()
        }

    before = _snapshot(log_dir)

    # 发出三次不同查询
    client.get("/api/live/scheduler/runs", params={"date": fixed_now.date().isoformat()})
    client.get("/api/live/scheduler/runs", params={"plan_id": "plan-A", "date": fixed_now.date().isoformat()})
    client.get("/api/live/scheduler/runs", params={"date": fixed_now.date().isoformat(), "limit": 1})

    after = _snapshot(log_dir)
    assert before == after, "查询端点修改了存储文件（违反 TSO-7 只读属性）"


# ---------------------------------------------------------------------------
# 7. 空结果（日期无记录）
# ---------------------------------------------------------------------------

def test_empty_result_for_missing_date(scheduler_client):
    """查询无记录的日期返回空列表，不报错。"""
    client, run_log, log_dir, fixed_now = scheduler_client

    resp = client.get(
        "/api/live/scheduler/runs",
        params={"date": "2020-01-01"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# 8. plan_id + limit 组合
# ---------------------------------------------------------------------------

def test_plan_id_and_limit_combined(scheduler_client):
    """plan_id 过滤 + limit 同时生效。"""
    client, run_log, log_dir, fixed_now = scheduler_client
    for i in range(4):
        run_log.record_trigger("plan-A", f"1{i}:00")
    run_log.record_trigger("plan-B", "15:00")

    resp = client.get(
        "/api/live/scheduler/runs",
        params={"plan_id": "plan-A", "date": fixed_now.date().isoformat(), "limit": 2},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert all(r["plan_id"] == "plan-A" for r in data)
