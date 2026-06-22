"""
GET /api/alpha/tasks 的 task_type 多值过滤测试（运行历史用）。

覆盖 backtest-screening-run-history 特性的 Property 1/2：
- 无类型 → 不过滤；单类型 → 与改造前一致（向后兼容）；逗号多值 → 返回 type 属于集合者。
- 多值过滤同时作用于历史记录；dedup/倒序/limit 不变。

测试经 TestClient 打 API；用独立 TaskManager 把不同类型任务归档进临时 history_store，
再用 create_app(history_store=同目录) 模拟"重启后查历史"，断言只对**本测试创建的** task_id
做存在/缺失判定（全局 task_manager 内存可能有其他任务，故核心断言用"返回项 type 必 ∈ 集合"）。

Feature: backtest-screening-run-history
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from aitrade.config import MAX_WORKERS
from aitrade.models.alpha import TaskType
from aitrade.task.history import TaskHistoryStore
from aitrade.task.manager import TaskManager


def _fresh_manager(tmp_path: Path) -> TaskManager:
    """独立 TaskManager（绕过单例），注入 tmp_path 的 history_store。"""
    mgr = object.__new__(TaskManager)
    mgr._tasks = {}
    mgr._task_lock = Lock()
    mgr._executor = ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS), thread_name_prefix="test-mt")
    mgr._history_store = TaskHistoryStore(tmp_path / "task_history")
    return mgr


def _wait_archived(store: TaskHistoryStore, task_id: str, timeout: float = 2.0) -> bool:
    """等待 task_id 出现在当日归档中。"""
    today = date.today()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(r.get("task_id") == task_id for r in store.query(start=today, end=today)):
            return True
        time.sleep(0.01)
    return False


def _archive_run(mgr: TaskManager, task_type: TaskType) -> str:
    """跑一个该类型的完成任务并等待归档，返回 task_id。"""
    tid = mgr.create_task(task_type, title=f"run-{task_type.value}")
    mgr.run_async(tid, lambda: {"ok": True})
    assert _wait_archived(mgr._history_store, tid), f"{task_type} 未归档"
    return tid


@pytest.fixture()
def client_with_runs(tmp_path: Path):
    """归档三类运行（回测/选股/无关），返回 (client, ids)。"""
    from aitrade.main import create_app

    mgr = _fresh_manager(tmp_path)
    ids = {
        "backtest": _archive_run(mgr, TaskType.CNN_BACKTEST),
        "screening": _archive_run(mgr, TaskType.CNN_SCREENING),
        "download": _archive_run(mgr, TaskType.DATA_DOWNLOAD),
    }
    app = create_app(history_store=TaskHistoryStore(tmp_path / "task_history"))
    return TestClient(app), ids


def _get(client: TestClient, task_type: str | None) -> list[dict]:
    """带 include_history 的 GET，可选 task_type。"""
    url = "/api/alpha/tasks?include_history=true&history_days=2"
    if task_type is not None:
        url += f"&task_type={task_type}"
    resp = client.get(url)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_multi_type_returns_only_requested_set(client_with_runs) -> None:
    # Feature: backtest-screening-run-history, Property 1: 多值过滤返回 type 属于集合者
    """逗号多值：返回项 type 必 ∈ 集合；回测+选股 id 在、无关 id 不在。"""
    client, ids = client_with_runs
    data = _get(client, "cnn_backtest,cnn_screening")

    returned = {t["task_id"]: t["type"] for t in data}
    # 核心属性：所有返回项的 type 都在请求集合内（内存 + 历史一致过滤）
    assert all(t in {"cnn_backtest", "cnn_screening"} for t in returned.values())
    assert ids["backtest"] in returned and ids["screening"] in returned
    assert ids["download"] not in returned  # 无关类型被排除


def test_single_type_backward_compatible(client_with_runs) -> None:
    # Feature: backtest-screening-run-history, Property 1: 单类型向后兼容
    """单类型：仅该类型，回测 id 在、选股/无关不在（与改造前一致）。"""
    client, ids = client_with_runs
    data = _get(client, "cnn_backtest")

    returned = {t["task_id"]: t["type"] for t in data}
    assert all(t == "cnn_backtest" for t in returned.values())
    assert ids["backtest"] in returned
    assert ids["screening"] not in returned
    assert ids["download"] not in returned


def test_no_type_does_not_filter(client_with_runs) -> None:
    # Feature: backtest-screening-run-history, Property 1: 无类型不过滤
    """不传 task_type：三类 id 都可见（历史不被类型过滤）。"""
    client, ids = client_with_runs
    data = _get(client, None)
    seen = {t["task_id"] for t in data}
    assert ids["backtest"] in seen
    assert ids["screening"] in seen
    assert ids["download"] in seen


def test_multi_type_preserves_dedup_and_order(client_with_runs) -> None:
    # Feature: backtest-screening-run-history, Property 2: dedup + 倒序不被多值过滤破坏
    """多值过滤下：无重复 task_id，且按 updated_at 倒序。"""
    client, _ = client_with_runs
    data = _get(client, "cnn_backtest,cnn_screening,backtest_run")

    task_ids = [t["task_id"] for t in data]
    assert len(task_ids) == len(set(task_ids)), "不应有重复 task_id"
    updated = [t.get("updated_at") or "" for t in data]
    assert updated == sorted(updated, reverse=True), "应按 updated_at 倒序"


def test_unknown_type_yields_empty_not_error(client_with_runs) -> None:
    # Feature: backtest-screening-run-history, Property 1: 未知类型命中 0 条不报错
    """请求一个不存在的类型：返回 200 且无本测试任务（不报错）。"""
    client, ids = client_with_runs
    data = _get(client, "no_such_type_xyz")
    seen = {t["task_id"] for t in data}
    assert ids["backtest"] not in seen and ids["screening"] not in seen
