"""
规则策略回测/扫参/WalkForward API 集成测试（Phase 2，任务 2.3 + 2.4）。

覆盖用例：
1. GET /sources 含 etf_momentum 与 cnn，且每项带 param_spec
2. POST /backtest/run 全链路：任务 completed，result 含 statistics/trades/equity_curve，equity_curve 非空
3. 空信号（universe 指向无数据标的）→ 任务 FAILED，message 含中文提示
4. sweep 3 个网格点 → rows 长度 3，params 字段区分
5. grid 超 50 → 422
6. walkforward 返回窗口数与聚合字段
7. 未注册 signal_source → 任务 FAILED，message 含已注册名

外部 I/O 全桩化：AlphaLab 用 tmp_path 隔离，写确定性合成日线数据。
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

import aitrade.rules  # noqa: F401  确保注册副作用在测试过程中生效

from aitrade.alpha.lab import AlphaLab
from aitrade.api import strategy as strategy_api
from aitrade.main import create_app

# ============================================================================
# 辅助常量
# ============================================================================

VT_A = "510300.SSE"  # 沪深300 ETF
VT_B = "510500.SSE"  # 中证500 ETF

# 合成数据：覆盖 2023-01-01 ~ 2024-06-01（约 17 个月，动量有足够的预热窗口）
DATA_START = date(2023, 1, 1)
DATA_END = date(2024, 6, 1)

# 回测区间（数据内的子集）
BT_START = date(2023, 6, 1)
BT_END = date(2024, 1, 1)


# ============================================================================
# 辅助：构造合成日线数据并写入 AlphaLab
# ============================================================================


def _make_bar_frame(
    prices: list[float],
    base_date: date = DATA_START,
) -> pl.DataFrame:
    """构造确定性合成日线 DataFrame，volume > 0。"""
    rows = []
    for i, p in enumerate(prices):
        dt = base_date + timedelta(days=i)
        rows.append({
            "datetime": datetime(dt.year, dt.month, dt.day, 9, 30),
            "open": p - 0.1,
            "high": p + 0.5,
            "low": p - 0.5,
            "close": p,
            "volume": 1_000_000.0,
            "turnover": p * 1_000_000.0,
            "open_interest": 0.0,
        })
    return pl.DataFrame(rows)


def _fill_lab(lab: AlphaLab) -> None:
    """向 lab 写入 VT_A、VT_B 各约 17 个月的确定性日线数据。"""
    n_days = (DATA_END - DATA_START).days + 1

    # VT_A：单调递增（正动量，始终有信号）
    prices_a = [10.0 + i * 0.01 for i in range(n_days)]
    # VT_B：略带震荡但整体正动量
    prices_b = [8.0 + (i % 30) * 0.02 for i in range(n_days)]

    lab.save_bar_frame(VT_A, "d", _make_bar_frame(prices_a))
    lab.save_bar_frame(VT_B, "d", _make_bar_frame(prices_b))


# ============================================================================
# Fixture
# ============================================================================


@pytest.fixture
def client(tmp_path, monkeypatch):
    """隔离的 TestClient：tmp_path AlphaLab + 合成数据 + monkeypatch _get_lab。"""
    lab = AlphaLab(tmp_path / "alpha_lab")
    _fill_lab(lab)

    # monkeypatch api.strategy._get_lab，使任务体获取同一 tmp_path lab
    monkeypatch.setattr(strategy_api, "_get_lab", lambda: AlphaLab(tmp_path / "alpha_lab"))

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


# ============================================================================
# 用例 1：GET /sources
# ============================================================================


def test_list_sources_contains_etf_momentum_and_cnn(client):
    """GET /sources 必须包含 etf_momentum 与 cnn，且每项有 param_spec 字段。"""
    resp = client.get("/api/strategy/sources")
    assert resp.status_code == 200
    sources = resp.json()
    names = {s["name"] for s in sources}
    assert "etf_momentum" in names, f"未找到 etf_momentum，已注册：{names}"
    assert "cnn" in names, f"未找到 cnn，已注册：{names}"
    for s in sources:
        assert "param_spec" in s, f"信号源 {s['name']} 缺少 param_spec 字段"


# ============================================================================
# 用例 2：POST /backtest/run 全链路
# ============================================================================


def test_backtest_run_full_pipeline(client):
    """全链路回测：任务 completed，result 含 statistics/trades/equity_curve，equity_curve 非空。"""
    body = {
        "signal_source": "etf_momentum",
        "signal_params": {
            "universe": [VT_A, VT_B],
            "lookback": 10,
            "min_momentum": 0.0,
        },
        "strategy_name": "rebalancing_topk",
        "strategy_params": {"top_k": 1},
        "start": BT_START.isoformat(),
        "end": BT_END.isoformat(),
        "capital": 500_000,
    }
    resp = client.post("/api/strategy/backtest/run", json=body)
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll(client, task_id)
    assert task["status"] == "completed", f"任务失败，message: {task.get('message')}"

    result = task["result"]
    assert "statistics" in result, "result 缺少 statistics"
    assert "trades" in result, "result 缺少 trades"
    assert "equity_curve" in result, "result 缺少 equity_curve"
    assert isinstance(result["equity_curve"], list), "equity_curve 应为 list"
    assert len(result["equity_curve"]) > 0, "equity_curve 不应为空"


# ============================================================================
# 用例 3：空信号 → FAILED + 中文提示
# ============================================================================


def test_backtest_run_empty_signal_fails_with_chinese_message(client):
    """universe 指向无本地数据的标的 → 任务 FAILED，message 含中文提示。"""
    body = {
        "signal_source": "etf_momentum",
        "signal_params": {
            "universe": ["999999.SSE"],  # 无本地数据
            "lookback": 10,
            "min_momentum": 0.0,
        },
        "strategy_name": "rebalancing_topk",
        "start": BT_START.isoformat(),
        "end": BT_END.isoformat(),
    }
    resp = client.post("/api/strategy/backtest/run", json=body)
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll(client, task_id)
    assert task["status"] == "failed", f"预期失败，实际状态: {task['status']}"
    assert "信号" in task["message"] or "数据" in task["message"], (
        f"message 应含中文提示，实际：{task['message']}"
    )


# ============================================================================
# 用例 4：sweep 3 个网格点
# ============================================================================


def test_sweep_three_grid_points(client):
    """sweep 3 组网格点 → rows 长度 3，各行 params 字段不同。"""
    body = {
        "signal_source": "etf_momentum",
        "signal_params": {
            "universe": [VT_A, VT_B],
            "lookback": 10,
            "min_momentum": 0.0,
        },
        "strategy_name": "rebalancing_topk",
        "strategy_params": {"top_k": 1},
        "start": BT_START.isoformat(),
        "end": BT_END.isoformat(),
        "capital": 500_000,
        "grid": [
            {"strategy_params": {"top_k": 1}},
            {"strategy_params": {"top_k": 2}},
            {"signal_params": {"lookback": 20}},
        ],
    }
    resp = client.post("/api/strategy/sweep/run", json=body)
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll(client, task_id, timeout=60.0)
    assert task["status"] == "completed", f"sweep 任务失败：{task.get('message')}"

    result = task["result"]
    assert "rows" in result, "result 缺少 rows"
    rows = result["rows"]
    assert len(rows) == 3, f"期望 3 行，实际 {len(rows)}"

    # 各行 params 字段不同
    params_list = [str(r["params"]) for r in rows]
    assert len(set(params_list)) == 3, "三行 params 应各不相同"


# ============================================================================
# 用例 5：grid 超 50 → 422
# ============================================================================


def test_sweep_grid_too_large_returns_422(client):
    """grid 超过 50 项 → Pydantic 校验失败，HTTP 422。"""
    body = {
        "signal_source": "etf_momentum",
        "signal_params": {"universe": [VT_A]},
        "strategy_name": "rebalancing_topk",
        "start": BT_START.isoformat(),
        "end": BT_END.isoformat(),
        "grid": [{"strategy_params": {"top_k": i + 1}} for i in range(51)],
    }
    resp = client.post("/api/strategy/sweep/run", json=body)
    assert resp.status_code == 422, f"期望 422，实际 {resp.status_code}"


# ============================================================================
# 用例 6：walkforward 窗口数与聚合字段
# ============================================================================


def test_walkforward_returns_windows_and_aggregate(client):
    """walkforward 应返回非零窗口数，以及 avg_return/avg_sharpe/positive_window_ratio/total_windows。"""
    body = {
        "signal_source": "etf_momentum",
        "signal_params": {
            "universe": [VT_A, VT_B],
            "lookback": 10,
            "min_momentum": 0.0,
        },
        "strategy_name": "rebalancing_topk",
        "strategy_params": {"top_k": 1},
        "start": BT_START.isoformat(),
        "end": BT_END.isoformat(),
        "capital": 300_000,
        "train_days": 60,
        "test_days": 30,
    }
    resp = client.post("/api/strategy/walkforward/run", json=body)
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll(client, task_id, timeout=120.0)
    assert task["status"] == "completed", f"walkforward 失败：{task.get('message')}"

    result = task["result"]
    assert "windows" in result, "result 缺少 windows"
    assert "aggregate" in result, "result 缺少 aggregate"

    windows = result["windows"]
    assert len(windows) > 0, "windows 不应为空"

    agg = result["aggregate"]
    assert "total_windows" in agg
    assert agg["total_windows"] == len(windows)
    assert "avg_return" in agg
    assert "avg_sharpe" in agg
    assert "positive_window_ratio" in agg

    # 每个窗口含必要字段
    for w in windows:
        assert "train_start" in w
        assert "test_start" in w
        assert "statistics" in w


# ============================================================================
# 用例 7：未注册 signal_source → 任务 FAILED，message 含已注册名
# ============================================================================


def test_unregistered_signal_source_fails_with_registry_hint(client):
    """未注册的 signal_source → 任务 FAILED，message 应包含已注册源名称提示。"""
    body = {
        "signal_source": "nonexistent_source_xyz",
        "signal_params": {},
        "strategy_name": "rebalancing_topk",
        "start": BT_START.isoformat(),
        "end": BT_END.isoformat(),
    }
    resp = client.post("/api/strategy/backtest/run", json=body)
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll(client, task_id)
    assert task["status"] == "failed", f"预期失败，实际状态: {task['status']}"
    # KeyError message 格式：「未注册的信号源：xxx（已注册：[...]）」
    assert "nonexistent_source_xyz" in task["message"] or "已注册" in task["message"], (
        f"message 应含信号源名或已注册列表，实际：{task['message']}"
    )
