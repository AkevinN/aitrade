"""
运行历史删除（管理）测试：DELETE /api/alpha/tasks/{task_id} + 底座删除方法。

覆盖 backtest-screening-run-history「管理删除」：
- JsonlDayStore.delete_where：重写当日文件去掉命中记录，保留其余，更新去重集合。
- TaskHistoryStore.delete_by_task_id：从归档窗口删指定 task。
- TaskManager.delete_task：从内存任务表删除。
- API：归档后 DELETE → 200 且历史不再可见；运行中 → 409；不存在 → 404。

Feature: backtest-screening-run-history
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from aitrade.config import MAX_WORKERS
from aitrade.live.jsonl_store import JsonlDayStore
from aitrade.models.alpha import TaskType
from aitrade.task.history import TaskHistoryStore
from aitrade.task.manager import TaskManager, task_manager


def _fresh_manager(tmp_path: Path) -> TaskManager:
    mgr = object.__new__(TaskManager)
    mgr._tasks = {}
    mgr._task_lock = Lock()
    mgr._executor = ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS), thread_name_prefix="test-del")
    mgr._history_store = TaskHistoryStore(tmp_path / "task_history")
    return mgr


def _wait_archived(store: TaskHistoryStore, task_id: str, timeout: float = 2.0) -> bool:
    today = date.today()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(r.get("task_id") == task_id for r in store.query(start=today, end=today)):
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# 底座：JsonlDayStore.delete_where
# ---------------------------------------------------------------------------


def test_jsonl_delete_where_removes_match_keeps_rest(tmp_path: Path) -> None:
    # Feature: backtest-screening-run-history, Property: 删除只移除命中、保留其余
    store = JsonlDayStore(tmp_path / "s")
    today = date.today()
    store.append({"task_id": "a", "v": 1}, dedup_key="task:a")
    store.append({"task_id": "b", "v": 2}, dedup_key="task:b")

    removed = store.delete_where(today, lambda r: r.get("task_id") == "a")
    assert removed == 1
    remaining = store.read_day(today)
    ids = {r["task_id"] for r in remaining}
    assert ids == {"b"}

    # 去重集合已移除 task:a → 可再次写入同键
    assert store.append({"task_id": "a", "v": 3}, dedup_key="task:a") is True


def test_jsonl_delete_where_no_match_returns_zero(tmp_path: Path) -> None:
    store = JsonlDayStore(tmp_path / "s")
    today = date.today()
    store.append({"task_id": "a"}, dedup_key="task:a")
    assert store.delete_where(today, lambda r: r.get("task_id") == "zzz") == 0
    assert len(store.read_day(today)) == 1


# ---------------------------------------------------------------------------
# TaskHistoryStore.delete_by_task_id + TaskManager.delete_task
# ---------------------------------------------------------------------------


def test_history_delete_by_task_id(tmp_path: Path) -> None:
    mgr = _fresh_manager(tmp_path)
    tid = mgr.create_task(TaskType.CNN_BACKTEST, title="bt")
    mgr.run_async(tid, lambda: {"ok": True})
    assert _wait_archived(mgr._history_store, tid)

    assert mgr._history_store.delete_by_task_id(tid) is True
    today = date.today()
    assert not any(r.get("task_id") == tid for r in mgr._history_store.query(start=today, end=today))
    # 再删一次（已不存在）→ False
    assert mgr._history_store.delete_by_task_id(tid) is False


def test_manager_delete_task(tmp_path: Path) -> None:
    mgr = _fresh_manager(tmp_path)
    tid = mgr.create_task(TaskType.CNN_SCREENING)
    assert mgr.get_task(tid) is not None
    assert mgr.delete_task(tid) is True
    assert mgr.get_task(tid) is None
    assert mgr.delete_task(tid) is False


# ---------------------------------------------------------------------------
# API：DELETE /api/alpha/tasks/{task_id}
# ---------------------------------------------------------------------------


def test_api_delete_archived_run(tmp_path: Path) -> None:
    # Feature: backtest-screening-run-history, Property: 删除后历史不再可见
    from aitrade.main import create_app

    mgr = _fresh_manager(tmp_path)
    tid = mgr.create_task(TaskType.CNN_SCREENING, title="选股运行")
    mgr.run_async(tid, lambda: {"run_id": "scr_x"})
    assert _wait_archived(mgr._history_store, tid)

    app = create_app(history_store=TaskHistoryStore(tmp_path / "task_history"))
    client = TestClient(app)

    # 删前可见
    before = client.get("/api/alpha/tasks?include_history=true&history_days=2").json()
    assert any(t["task_id"] == tid for t in before)

    resp = client.delete(f"/api/alpha/tasks/{tid}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True

    # 删后不可见
    after = client.get("/api/alpha/tasks?include_history=true&history_days=2").json()
    assert not any(t["task_id"] == tid for t in after)


def test_api_delete_unknown_returns_404() -> None:
    from aitrade.main import create_app

    client = TestClient(create_app())
    resp = client.delete("/api/alpha/tasks/no_such_task_xyz")
    assert resp.status_code == 404


def test_api_delete_running_returns_409() -> None:
    # Feature: backtest-screening-run-history, Property: 运行中不可删
    from aitrade.main import create_app

    client = TestClient(create_app())
    # 在全局 task_manager 建一个未跑的任务（pending 态），断言 409，再清理。
    tid = task_manager.create_task(TaskType.CNN_BACKTEST, title="pending-del")
    try:
        resp = client.delete(f"/api/alpha/tasks/{tid}")
        assert resp.status_code == 409, resp.text
    finally:
        task_manager.delete_task(tid)


# ---------------------------------------------------------------------------
# 级联清理：选股运行删除连带清掉 WF 报告 + 临时模型 + 落盘 ScreeningResult
# ---------------------------------------------------------------------------


def _make_screening_artifacts() -> tuple[str, str, list[str]]:
    """造一份 WF 报告 + 两个临时候选模型（.pt + _history.json），返回 (run_id, report_id, models)。"""
    from aitrade.cnn.storage import CNN_MODEL_DIR
    from aitrade.screening.store import build_screening_governance_store

    models = [
        "screening_x_wf1_s0_aaaa_20240101-20250101",
        "screening_x_wf2_s0_bbbb_20240101-20250101",
    ]
    CNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for m in models:
        (CNN_MODEL_DIR / f"{m}.pt").write_text("weights", encoding="utf-8")
        (CNN_MODEL_DIR / f"{m}_history.json").write_text("{}", encoding="utf-8")

    gov = build_screening_governance_store()
    report_id = "wf_purge_test_1"
    gov.save_report(
        {
            "report_id": report_id,
            "folds": [
                {"candidate_models": [models[0]], "candidate_model": models[0]},
                {"candidate_models": [models[1]], "candidate_model": models[1]},
            ],
        }
    )
    return "scr_purge_test_1", report_id, models


def test_purge_screening_run_cascade() -> None:
    # Feature: backtest-screening-run-history, Property: 删除级联清理隔离产物
    from aitrade.cnn.storage import CNN_MODEL_DIR
    from aitrade.screening.cleanup import purge_screening_run
    from aitrade.screening.store import build_screening_governance_store

    run_id, report_id, models = _make_screening_artifacts()
    gov = build_screening_governance_store()
    assert gov.get_report(report_id) is not None

    result = {
        "run_id": run_id,
        "leaderboard": [{"tier2": {"report_id": report_id}}, {"tier2": None}],
    }
    counts = purge_screening_run(result)

    assert counts["reports"] == 1
    assert counts["models"] == 2
    # 报告与临时模型（.pt + _history.json）都已删除
    assert gov.get_report(report_id) is None
    for m in models:
        assert not (CNN_MODEL_DIR / f"{m}.pt").exists()
        assert not (CNN_MODEL_DIR / f"{m}_history.json").exists()


def test_api_delete_screening_run_purges_artifacts(tmp_path: Path) -> None:
    # Feature: backtest-screening-run-history, Property: API 删除选股运行级联清产物
    from aitrade.cnn.storage import CNN_MODEL_DIR
    from aitrade.main import create_app
    from aitrade.screening.store import build_screening_governance_store

    run_id, report_id, models = _make_screening_artifacts()

    # 归档一个 CNN_SCREENING 任务，其 result 指向上述产物
    mgr = _fresh_manager(tmp_path)
    tid = mgr.create_task(TaskType.CNN_SCREENING, title="选股运行")
    mgr.run_async(
        tid,
        lambda: {"run_id": run_id, "leaderboard": [{"tier2": {"report_id": report_id}}]},
    )
    assert _wait_archived(mgr._history_store, tid)

    app = create_app(history_store=TaskHistoryStore(tmp_path / "task_history"))
    client = TestClient(app)
    resp = client.delete(f"/api/alpha/tasks/{tid}")
    assert resp.status_code == 200, resp.text
    purged = resp.json()["purged"]
    assert purged["reports"] == 1 and purged["models"] == 2

    gov = build_screening_governance_store()
    assert gov.get_report(report_id) is None
    for m in models:
        assert not (CNN_MODEL_DIR / f"{m}.pt").exists()
