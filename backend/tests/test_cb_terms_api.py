"""
CB 条款刷新 API 测试（Task 4.2）。

覆盖：
1. POST /cb-terms/refresh 全量（monkeypatch akshare，mock 返回 2 只转债）→ 任务 completed
2. POST /cb-terms/refresh 子集（symbols 参数）→ 仅处理指定子集
3. 进度回调不报错
4. akshare 拉快照失败 → 任务 failed，message 含错误信息
"""

from __future__ import annotations

import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aitrade.api import strategy as strategy_api
from aitrade.main import create_app
from aitrade.rules.store import CBTermsStore


# ---------------------------------------------------------------------------
# 构造伪 akshare 返回值
# ---------------------------------------------------------------------------


def _make_fake_snapshot_df() -> pd.DataFrame:
    """构造一个最小化的 bond_zh_cov 返回 DataFrame（pandas）。"""
    return pd.DataFrame(
        {
            "债券代码": ["113050", "128093"],
            "债券简称": ["富投转债", "岱勒转债"],
            "转股价": [10.5, 8.3],
            "债现价": [108.7, 95.2],
            "转股溢价率": [12.34, 8.91],
            "发行规模": [6.0, 4.5],
            "信用评级": ["AA", "AA-"],
            "上市时间": ["2023-01-01", "2023-06-15"],
        }
    )


def _make_fake_premium_df() -> pd.DataFrame:
    """构造一个最小化的 bond_zh_cov_value_analysis 返回 DataFrame（pandas）。"""
    return pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "收盘价": [108.7, 109.2],
            "纯债价值": [95.0, 95.1],
            "转股价值": [97.5, 98.0],
            "纯债溢价率": [14.42, 14.82],
            "转股溢价率": [11.49, 11.42],
        }
    )


# ---------------------------------------------------------------------------
# Fixture：隔离的 FastAPI 测试客户端
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    """隔离的 TestClient，monkeypatch _ak_bond_zh_cov 和 _ak_bond_zh_cov_value_analysis。"""
    # 替换两个模块级 akshare 包装函数（测试桩点）
    monkeypatch.setattr(strategy_api, "_ak_bond_zh_cov", _make_fake_snapshot_df)
    monkeypatch.setattr(strategy_api, "_ak_bond_zh_cov_value_analysis", lambda symbol: _make_fake_premium_df())

    # 使用 tmp_path 隔离存储
    monkeypatch.setattr(
        "aitrade.rules.store.RULES_DATA_PATH",
        tmp_path / "rules",
    )

    app = create_app()
    with TestClient(app) as c:
        yield c


def _poll(c: TestClient, task_id: str, timeout: float = 30.0) -> dict:
    """轮询任务终态，超时抛 AssertionError。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = c.get(f"/api/alpha/tasks/{task_id}").json()
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.05)
    raise AssertionError(f"任务 {task_id} 未在 {timeout}s 内完成")


# ---------------------------------------------------------------------------
# 测试 1：全量刷新任务 completed
# ---------------------------------------------------------------------------


def test_refresh_cb_terms_full_completes(client, tmp_path, monkeypatch) -> None:
    """POST /cb-terms/refresh（无 symbols）→ 任务 completed，快照含 2 只转债。"""
    # 需要 CBTermsStore 也指向 tmp_path
    store = CBTermsStore(base_path=tmp_path / "rules")
    monkeypatch.setattr(strategy_api, "CBTermsStore", lambda: store)

    resp = client.post("/api/strategy/cb-terms/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data

    task = _poll(client, data["task_id"])
    assert task["status"] == "completed", f"任务失败：{task.get('message')}"

    result = task.get("result") or {}
    assert result.get("snapshot_count", 0) >= 2


# ---------------------------------------------------------------------------
# 测试 2：子集刷新（symbols 参数）
# ---------------------------------------------------------------------------


def test_refresh_cb_terms_subset(tmp_path, monkeypatch) -> None:
    """_refresh_cb_terms 带 symbols 子集时仅处理指定子集（直接调用任务体）。"""
    monkeypatch.setattr(strategy_api, "_ak_bond_zh_cov", _make_fake_snapshot_df)
    monkeypatch.setattr(strategy_api, "_ak_bond_zh_cov_value_analysis", lambda symbol: _make_fake_premium_df())
    # 禁用实际等待
    monkeypatch.setattr(strategy_api.time, "sleep", lambda _: None)

    # 只指定 1 只转债
    result = strategy_api._refresh_cb_terms(symbols=["113050.SSE"])

    # 快照应含全量（2 只），但溢价率历史仅处理 1 只
    assert result["snapshot_count"] == 2
    # success 应 <= 1（快照有 2 只，但子集只有 1 只在快照中）
    assert result["success"] <= 1


# ---------------------------------------------------------------------------
# 测试 3：akshare 快照失败 → 任务 failed
# ---------------------------------------------------------------------------


def test_refresh_cb_terms_snapshot_failure_fails_task(client, tmp_path, monkeypatch) -> None:
    """bond_zh_cov() 抛异常时任务应为 failed，message 含错误信息。"""
    def _raise_bond():
        raise RuntimeError("接口超时")

    monkeypatch.setattr(strategy_api, "_ak_bond_zh_cov", _raise_bond)

    resp = client.post("/api/strategy/cb-terms/refresh")
    assert resp.status_code == 200

    task = _poll(client, resp.json()["task_id"])
    assert task["status"] == "failed"
    assert "接口超时" in task.get("message", "")


# ---------------------------------------------------------------------------
# 测试 4：_refresh_cb_terms 任务体直接调用（单元测试，无网络，带进度回调）
# ---------------------------------------------------------------------------


def test_refresh_cb_terms_task_body_with_progress(tmp_path, monkeypatch) -> None:
    """直接调用 _refresh_cb_terms 任务体，进度回调应被正常触发。"""
    monkeypatch.setattr(strategy_api, "_ak_bond_zh_cov", _make_fake_snapshot_df)
    monkeypatch.setattr(strategy_api, "_ak_bond_zh_cov_value_analysis", lambda symbol: _make_fake_premium_df())
    # 禁用实际等待
    monkeypatch.setattr(strategy_api.time, "sleep", lambda _: None)

    progress_calls: list[tuple[float, str]] = []

    def _on_progress(pct: float, msg: str) -> None:
        progress_calls.append((pct, msg))

    result = strategy_api._refresh_cb_terms(
        symbols=[],
        on_progress=_on_progress,
    )

    assert result["snapshot_count"] >= 2
    assert result["success"] >= 0

    # 进度回调应至少被调用过一次
    assert len(progress_calls) >= 1
    # 最后一次进度应接近 100%
    last_pct, _ = progress_calls[-1]
    assert last_pct == pytest.approx(100.0, abs=1.0)
