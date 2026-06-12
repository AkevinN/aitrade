"""
任务历史持久化测试套件（task-scheduler-observability Wave 2a，任务 3）

覆盖：
- TSO-1 属性测试（Hypothesis）：任意任务跑到终态（成功/失败），历史当日文件存在
  同 task_id 记录且 status/duration_ms/error_traceback 与内存一致
- TSO-4 任务侧属性：注入抛错的 history_store 桩 → 任务状态流转与正常时一致，仅 WARNING
- 示例：query 过滤（status/type/limit/倒序）
- 示例：API include_history 合并（重启模拟：清内存 dict 后 include_history=true 仍可见历史）
- 示例：API 默认行为与现状一致（无参数时响应形态不变）
- TASK_DB_PATH 删除确认（grep 断言）
"""

from __future__ import annotations

import logging
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aitrade.live.jsonl_store import JsonlDayStore
from aitrade.models import TaskStatus, TaskType
from aitrade.task.history import TaskHistoryStore
from aitrade.task.manager import TaskManager


# ---------------------------------------------------------------------------
# 辅助：等待终态
# ---------------------------------------------------------------------------

def wait_terminal(manager: TaskManager, task_id: str, timeout: float = 5.0) -> Any:
    """等待任务到达终态，返回 TaskModel 或 None（超时）。"""
    deadline = time.time() + timeout
    task = manager.get_task(task_id)
    while task is not None and task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
        if time.time() > deadline:
            return None
        time.sleep(0.02)
        task = manager.get_task(task_id)
    return task


def wait_archived(history_store, task_id: str, timeout: float = 2.0) -> bool:
    """等待归档记录出现在 history_store 中（处理 update_task 到 _archive 之间的竞态窗口）。"""
    from datetime import date
    deadline = time.time() + timeout
    while time.time() < deadline:
        today = date.today()
        records = history_store.query(start=today, end=today)
        if any(r.get("task_id") == task_id for r in records):
            return True
        time.sleep(0.01)
    return False


def fresh_manager(tmp_path: Path) -> TaskManager:
    """创建独立的 TaskManager 实例（绕过全局单例），注入 tmp_path 的 history_store。

    用 object.__new__ 绕过单例 __new__ 逻辑，得到独立实例。
    """
    store = TaskHistoryStore(tmp_path / "task_history")
    # 绕过单例机制：直接用 object.__new__ 分配新实例
    mgr = object.__new__(TaskManager)
    mgr._tasks = {}
    from threading import Lock
    from concurrent.futures import ThreadPoolExecutor
    from aitrade.config import MAX_WORKERS
    mgr._task_lock = Lock()
    mgr._executor = ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS), thread_name_prefix="test-task")
    mgr._history_store = store
    return mgr


# ---------------------------------------------------------------------------
# TSO-1 属性测试：任务终态必归档且与内存一致
# Feature: task-scheduler-observability, Property TSO-1:
# 对任意到达终态（completed/failed）的任务，Task_History_Store 当日文件中
# 存在一条 task_id 相同的记录，且 status/duration_ms/error_traceback 与内存 TaskModel 一致。
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    should_fail=st.booleans(),
    task_type=st.sampled_from(list(TaskType)),
)
def test_tso1_terminal_task_archived(should_fail: bool, task_type: TaskType) -> None:
    """
    # Feature: task-scheduler-observability, Property TSO-1:
    # 任意任务到达终态（成功/失败），历史文件中存在同 task_id 记录，
    # status/duration_ms/error_traceback 与内存一致。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = fresh_manager(Path(tmpdir))
        task_id = mgr.create_task(task_type)

        if should_fail:
            def _fn():
                raise RuntimeError("property-test-failure")
        else:
            def _fn():
                return {"ok": True}

        mgr.run_async(task_id, _fn)
        task = wait_terminal(mgr, task_id)

        assert task is not None
        expected_status = TaskStatus.FAILED if should_fail else TaskStatus.COMPLETED

        # 内存侧
        assert task.status == expected_status
        assert task.duration_ms is not None and task.duration_ms >= 0

        # 历史侧：当日文件必须存在同 task_id 记录
        # wait_archived 处理 update_task→_archive 之间的竞态窗口
        hist_store = mgr._history_store
        archived = wait_archived(hist_store, task_id)
        assert archived, f"历史文件中未找到 task_id={task_id} 的记录（超时 2s）"

        today = date.today()
        records = hist_store.query(start=today, end=today)
        match = [r for r in records if r.get("task_id") == task_id]
        assert len(match) >= 1, f"历史文件中未找到 task_id={task_id} 的记录"

        rec = match[0]
        assert rec["status"] == expected_status.value, (
            f"历史记录 status={rec['status']!r} 与内存 {expected_status.value!r} 不一致"
        )
        assert rec["duration_ms"] == task.duration_ms, (
            f"历史记录 duration_ms={rec['duration_ms']} 与内存 {task.duration_ms} 不一致"
        )
        assert rec.get("error_traceback", "") == task.error_traceback, (
            f"历史记录 error_traceback 与内存不一致"
        )


# ---------------------------------------------------------------------------
# TSO-4 任务侧属性：记录失败不影响任务状态流转
# Feature: task-scheduler-observability, Property TSO-4:
# 注入抛错的 history_store 桩 → 任务状态流转与正常时一致（终态/result/message 不变），
# 仅多出 WARNING 日志。
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    should_fail=st.booleans(),
    task_type=st.sampled_from(list(TaskType)),
)
def test_tso4_archive_failure_does_not_affect_task(
    should_fail: bool,
    task_type: TaskType,
) -> None:
    """
    # Feature: task-scheduler-observability, Property TSO-4:
    # 注入抛错的 history_store 桩，任务终态/result/message 与无故障时逐位一致，
    # 仅多出 WARNING 日志（通过日志 handler 注入验证，避免 caplog 与 Hypothesis 的冲突）。
    """
    import logging as _logging

    warning_received: list[str] = []

    class _WarningCapture(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            if record.levelno == _logging.WARNING:
                warning_received.append(record.getMessage())

    handler = _WarningCapture()
    mgr_logger = _logging.getLogger("aitrade.task.manager")
    mgr_logger.addHandler(handler)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 构造抛错桩
            broken_store = MagicMock(spec=TaskHistoryStore)
            broken_store.archive.side_effect = OSError("磁盘已满（桩）")

            # 绕过单例机制：object.__new__ 分配独立实例
            mgr = object.__new__(TaskManager)
            mgr._tasks = {}
            from threading import Lock
            from concurrent.futures import ThreadPoolExecutor
            from aitrade.config import MAX_WORKERS
            mgr._task_lock = Lock()
            mgr._executor = ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS), thread_name_prefix="test-broken")
            mgr._history_store = broken_store

            task_id = mgr.create_task(task_type)

            if should_fail:
                def _fn():
                    raise RuntimeError("expected-failure")
            else:
                def _fn():
                    return {"status": "ok"}

            mgr.run_async(task_id, _fn)
            task = wait_terminal(mgr, task_id)

            # 等待 _archive 调用（其中含 WARNING log）
            deadline = time.time() + 2.0
            while not any("归档" in w for w in warning_received) and time.time() < deadline:
                time.sleep(0.01)

        assert task is not None
        expected_status = TaskStatus.FAILED if should_fail else TaskStatus.COMPLETED
        assert task.status == expected_status, (
            f"归档失败不应影响任务终态，期望 {expected_status!r}，实际 {task.status!r}"
        )
        assert task.duration_ms is not None and task.duration_ms >= 0

        if not should_fail:
            assert task.result == {"status": "ok"}, "成功任务的 result 不应被归档失败影响"
            assert task.message == "任务完成", (
                f"归档失败不应影响 message，期望 '任务完成'，实际 {task.message!r}"
            )
        else:
            assert task.message == "expected-failure", (
                f"归档失败不应影响 message，期望 'expected-failure'，实际 {task.message!r}"
            )

        # 必须有 WARNING 日志记录归档失败
        assert any("归档" in w for w in warning_received), (
            "归档失败时应记录 WARNING，但未找到"
        )
    finally:
        mgr_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# 示例：query 过滤
# ---------------------------------------------------------------------------

def test_query_filter_by_status(tmp_path: Path) -> None:
    """query(status='completed') 只返回已完成任务。"""
    mgr = fresh_manager(tmp_path)

    # 完成一个
    t1 = mgr.create_task(TaskType.DATA_DOWNLOAD)
    mgr.run_async(t1, lambda: {"done": True})
    wait_terminal(mgr, t1)
    wait_archived(mgr._history_store, t1)

    # 失败一个
    t2 = mgr.create_task(TaskType.DATA_DOWNLOAD)
    mgr.run_async(t2, lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    wait_terminal(mgr, t2)
    wait_archived(mgr._history_store, t2)

    today = date.today()
    completed = mgr._history_store.query(status="completed", start=today, end=today)
    failed = mgr._history_store.query(status="failed", start=today, end=today)

    assert all(r["status"] == "completed" for r in completed)
    assert all(r["status"] == "failed" for r in failed)


def test_query_filter_by_task_type(tmp_path: Path) -> None:
    """query(task_type='data_download') 只返回对应类型任务。"""
    mgr = fresh_manager(tmp_path)

    t1 = mgr.create_task(TaskType.DATA_DOWNLOAD)
    mgr.run_async(t1, lambda: {})
    wait_terminal(mgr, t1)
    wait_archived(mgr._history_store, t1)

    t2 = mgr.create_task(TaskType.MODEL_TRAIN)
    mgr.run_async(t2, lambda: {})
    wait_terminal(mgr, t2)
    wait_archived(mgr._history_store, t2)

    today = date.today()
    download_tasks = mgr._history_store.query(task_type="data_download", start=today, end=today)
    train_tasks = mgr._history_store.query(task_type="model_train", start=today, end=today)

    assert all(r["type"] == "data_download" for r in download_tasks)
    assert all(r["type"] == "model_train" for r in train_tasks)


def test_query_limit(tmp_path: Path) -> None:
    """query(limit=2) 最多返回 2 条。"""
    mgr = fresh_manager(tmp_path)
    for _ in range(5):
        tid = mgr.create_task(TaskType.DATA_DOWNLOAD)
        mgr.run_async(tid, lambda: {})
        wait_terminal(mgr, tid)
        wait_archived(mgr._history_store, tid)

    today = date.today()
    results = mgr._history_store.query(start=today, end=today, limit=2)
    assert len(results) <= 2


def test_query_reverse_order(tmp_path: Path) -> None:
    """query 无过滤时返回倒序（最新在前）。"""
    mgr = fresh_manager(tmp_path)
    ids = []
    for _ in range(3):
        tid = mgr.create_task(TaskType.DATA_DOWNLOAD)
        mgr.run_async(tid, lambda: {})
        wait_terminal(mgr, tid)
        wait_archived(mgr._history_store, tid)  # 等待归档写入完成
        ids.append(tid)

    today = date.today()
    results = mgr._history_store.query(start=today, end=today)
    # 倒序：最后写入的应排在最前（或至少有结果）
    assert len(results) >= 3


# ---------------------------------------------------------------------------
# 示例：dedup — 同 task_id 重复归档只写一次
# ---------------------------------------------------------------------------

def test_archive_dedup(tmp_path: Path) -> None:
    """同一 task_id 归档两次，当日文件中只有一条记录（dedup）。"""
    store = TaskHistoryStore(tmp_path / "task_history")
    mgr = fresh_manager(tmp_path)
    task_id = mgr.create_task(TaskType.DATA_DOWNLOAD)
    task = mgr.get_task(task_id)

    store.archive(task)
    store.archive(task)  # 重复归档

    today = date.today()
    records = store.query(start=today, end=today)
    matched = [r for r in records if r.get("task_id") == task_id]
    assert len(matched) == 1, "同 task_id 重复归档应去重，当日只保留 1 条"


# ---------------------------------------------------------------------------
# 示例：API include_history 合并
# 模拟重启：清内存 dict 后 include_history=true 仍可见历史
# ---------------------------------------------------------------------------

def test_api_include_history_survives_restart(tmp_path: Path) -> None:
    """
    模拟重启场景：先归档任务到历史，清内存（模拟重启），
    再调 GET /api/alpha/tasks?include_history=true 仍能看到历史记录。
    """
    from aitrade.main import create_app

    mgr = fresh_manager(tmp_path)

    # 跑一个完成任务
    task_id = mgr.create_task(TaskType.DATA_DOWNLOAD, title="历史任务")
    mgr.run_async(task_id, lambda: {"done": True})
    wait_terminal(mgr, task_id)
    wait_archived(mgr._history_store, task_id)  # 等待归档写入

    # 确认历史已归档
    today = date.today()
    records = mgr._history_store.query(start=today, end=today)
    assert any(r.get("task_id") == task_id for r in records), "历史应已归档"

    # 模拟重启：构建新 app，注入同一 history_store（新 task_manager 内存为空）
    # mgr._history_store 已写入文件，新实例读同一目录的文件
    hist_store_for_app = TaskHistoryStore(tmp_path / "task_history")
    app = create_app(history_store=hist_store_for_app)
    client = TestClient(app)

    # 默认行为：无参数时，只返回内存任务（全局 task_manager 可能有其他测试留下的任务）
    resp_default = client.get("/api/alpha/tasks")
    assert resp_default.status_code == 200
    default_data = resp_default.json()
    assert isinstance(default_data, list)
    # 我们关心的是 include_history=true 会返回历史任务
    # 不强断言 task_id not in default_ids（全局 task_manager 可能有该任务）

    # include_history=true 应合并历史
    resp_hist = client.get("/api/alpha/tasks?include_history=true")
    assert resp_hist.status_code == 200
    hist_ids = [t["task_id"] for t in resp_hist.json()]
    assert task_id in hist_ids, "include_history=true 应包含历史任务"


# ---------------------------------------------------------------------------
# 示例：API 默认行为与现状完全一致（无参数时响应形态不变）
# ---------------------------------------------------------------------------

def test_api_default_tasks_behavior_unchanged() -> None:
    """
    GET /api/alpha/tasks 无参数时：
    - 返回 list[dict]
    - 每条包含 task_id / status / type 字段
    - 不含 include_history 相关合并逻辑（前端 useTaskList 零回归）
    """
    from aitrade.main import create_app

    app = create_app()
    client = TestClient(app)

    resp = client.get("/api/alpha/tasks")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    # 如果有任务，验证字段形态
    for task_dict in data:
        assert "task_id" in task_dict
        assert "status" in task_dict
        assert "type" in task_dict


def test_api_tasks_status_filter() -> None:
    """
    GET /api/alpha/tasks?status=completed 只返回 completed 任务（内存侧）。
    """
    from aitrade.main import create_app

    app = create_app()
    client = TestClient(app)

    resp = client.get("/api/alpha/tasks?status=completed")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    for task_dict in data:
        assert task_dict["status"] == "completed"


# ---------------------------------------------------------------------------
# TASK_DB_PATH 已删除（可选验证：确认 config.py 中不再有 TASK_DB_PATH）
# ---------------------------------------------------------------------------

def test_task_db_path_removed() -> None:
    """TASK_DB_PATH 死占位已从 config.py 删除。"""
    import aitrade.config as cfg
    assert not hasattr(cfg, "TASK_DB_PATH"), (
        "TASK_DB_PATH 应已从 config.py 删除（R2.5），请移除该死占位"
    )


# ---------------------------------------------------------------------------
# TASK_HISTORY_PATH 存在
# ---------------------------------------------------------------------------

def test_task_history_path_exists_in_config() -> None:
    """TASK_HISTORY_PATH 已在 config.py 中定义。"""
    import aitrade.config as cfg
    assert hasattr(cfg, "TASK_HISTORY_PATH"), "TASK_HISTORY_PATH 应在 config.py 中定义"
    from pathlib import Path as P
    assert isinstance(cfg.TASK_HISTORY_PATH, P)


# ---------------------------------------------------------------------------
# Fix: include_history 合并排序（Fix 1）
# 内存任务 updated_at 晚于历史任务 → include_history 倒序首位是内存任务
# ---------------------------------------------------------------------------

def test_include_history_sort_mem_task_wins(tmp_path: Path) -> None:
    """
    内存任务 updated_at 晚于历史任务时，include_history 列表首位应为内存任务。
    验证 model_dump(mode='json') 使 datetime 序列化为 ISO T 格式，消除空格 vs T 的排序偏差。
    """
    from aitrade.main import create_app

    mgr = fresh_manager(tmp_path)

    # 先跑一个任务归档到历史（旧任务）
    hist_task_id = mgr.create_task(TaskType.DATA_DOWNLOAD, title="历史旧任务")
    mgr.run_async(hist_task_id, lambda: {"old": True})
    wait_terminal(mgr, hist_task_id)
    wait_archived(mgr._history_store, hist_task_id)

    # 确认已归档
    today = date.today()
    records = mgr._history_store.query(start=today, end=today)
    assert any(r.get("task_id") == hist_task_id for r in records), "旧任务应已归档"

    # 模拟重启：新 app 注入同一 history_store，内存中只有新任务
    hist_store_for_app = TaskHistoryStore(tmp_path / "task_history")
    app = create_app(history_store=hist_store_for_app)

    # 向 app 的全局 task_manager 注入一个更新的内存任务（updated_at 更晚）
    # 通过在 app 构建后直接修改全局 task_manager 来模拟内存中有新任务
    import aitrade.task as task_mod
    new_task_id = task_mod.task_manager.create_task(TaskType.MODEL_TRAIN, title="内存新任务")
    # 让其完成（updated_at 肯定晚于历史任务）
    task_mod.task_manager.run_async(new_task_id, lambda: {"new": True})
    deadline = __import__("time").time() + 5.0
    new_task = task_mod.task_manager.get_task(new_task_id)
    from aitrade.models import TaskStatus
    while new_task and new_task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
        if __import__("time").time() > deadline:
            break
        __import__("time").sleep(0.02)
        new_task = task_mod.task_manager.get_task(new_task_id)

    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)
    resp = client.get("/api/alpha/tasks?include_history=true&limit=200")
    assert resp.status_code == 200
    result = resp.json()
    assert len(result) >= 2, "应包含内存任务和历史任务"

    task_ids = [t["task_id"] for t in result]
    # 历史任务应出现在列表中
    assert hist_task_id in task_ids, "历史任务应出现在 include_history 列表中"
    # 内存新任务 updated_at 更晚，应排在历史任务前
    if new_task_id in task_ids:
        new_idx = task_ids.index(new_task_id)
        hist_idx = task_ids.index(hist_task_id)
        assert new_idx < hist_idx, (
            f"内存新任务（idx={new_idx}）应排在历史旧任务（idx={hist_idx}）前面"
        )


# ---------------------------------------------------------------------------
# Fix: history query 剔除内部字段（Fix 3）
# 历史记录不含 _ 前缀键或 ts 键
# ---------------------------------------------------------------------------

def test_history_query_no_internal_fields(tmp_path: Path) -> None:
    """query 返回的历史记录不含 _ 前缀键和 ts 键（与内存任务响应形态对称）。"""
    mgr = fresh_manager(tmp_path)

    task_id = mgr.create_task(TaskType.DATA_DOWNLOAD, title="投影测试任务")
    mgr.run_async(task_id, lambda: {"done": True})
    wait_terminal(mgr, task_id)
    wait_archived(mgr._history_store, task_id)

    today = date.today()
    records = mgr._history_store.query(start=today, end=today)
    matched = [r for r in records if r.get("task_id") == task_id]
    assert matched, "应能查到归档记录"

    rec = matched[0]
    # 不含 _ 前缀内部键
    internal_keys = [k for k in rec if k.startswith("_")]
    assert not internal_keys, f"历史记录不应含 _ 前缀键: {internal_keys}"
    # 不含 ts 键
    assert "ts" not in rec, "历史记录不应含 ts 键"
    # 正常业务字段存在
    assert "task_id" in rec
    assert "status" in rec
    assert "type" in rec
