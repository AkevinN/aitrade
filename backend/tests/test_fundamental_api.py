"""
基本面 API 集成测试（Phase 5 Task 5.1 / 5.2）。

覆盖：
1. 正常流程：mock provider 返回数据 → 任务 completed，success 数正确
2. 部分失败：某标的返回空列表 → 任务 completed，failed 数正确，success 数正确
3. 全部失败：所有标的返回空 → 任务 FAILED，message 含中文
4. 回测结果含 universe_coverage 字段（依托 test_strategy_api 的 client fixture 复用）
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

from aitrade.alpha.lab import AlphaLab
from aitrade.api import strategy as strategy_api
from aitrade.datasource.types import FundamentalRecord
from aitrade.main import create_app


# ============================================================================
# 辅助
# ============================================================================

VT_A = "600519.SSE"
VT_B = "000001.SZSE"
VT_C = "999999.SSE"  # 模拟无数据标的

START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 3, 31)


def _make_records(vt_symbol: str, n: int = 3) -> list[FundamentalRecord]:
    """构造 n 条 FundamentalRecord（YYYYMMDD 格式 trade_date）。"""
    symbol, exchange = vt_symbol.rsplit(".", 1)
    records = []
    base = date(2024, 1, 2)
    for i in range(n):
        d = base + timedelta(days=i)
        records.append(FundamentalRecord(
            symbol=symbol,
            exchange=exchange,
            trade_date=d.strftime("%Y%m%d"),
            pe=20.0 + i,
            pe_ttm=19.5 + i,
            pb=2.0,
            total_mv=100_000.0 + i * 1000,  # 万元
            circ_mv=80_000.0,
            turnover_rate=1.5,
        ))
    return records


def _make_bar_frame(prices: list[float], base_date: date = date(2023, 1, 1)) -> pl.DataFrame:
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


def _poll(c: TestClient, task_id: str, timeout: float = 30.0) -> dict:
    """轮询任务终态。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = c.get(f"/api/alpha/tasks/{task_id}").json()
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.05)
    raise AssertionError(f"任务 {task_id} 未在 {timeout}s 内完成")


# ============================================================================
# Fixture
# ============================================================================


@pytest.fixture
def client_with_mock_fundamental(tmp_path, monkeypatch):
    """注入 mock fundamental provider + tmp_path AlphaLab。"""
    # mock _fetch_fundamental：VT_A 返回 3 条记录，VT_B 返回 0（无数据）
    def _mock_fetch(symbol: str, exchange: str, start_str: str, end_str: str) -> list:
        vt = f"{symbol}.{exchange}"
        if vt == VT_A:
            return _make_records(VT_A, n=3)
        return []  # VT_B 无数据

    monkeypatch.setattr(strategy_api, "_fetch_fundamental", _mock_fetch)

    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_all_fail(tmp_path, monkeypatch):
    """所有标的均返回空数据的 client。"""
    monkeypatch.setattr(strategy_api, "_fetch_fundamental", lambda *_: [])

    app = create_app()
    with TestClient(app) as c:
        yield c


# ============================================================================
# 测试 1：正常流程（单标的有数据）
# ============================================================================


def test_fundamental_refresh_success(client_with_mock_fundamental, tmp_path, monkeypatch):
    """mock provider 返回数据 → 任务 completed，success = 1。"""
    client = client_with_mock_fundamental

    body = {
        "vt_symbols": [VT_A],
        "start": START_DATE.isoformat(),
        "end": END_DATE.isoformat(),
    }
    resp = client.post("/api/strategy/fundamental/refresh", json=body)
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll(client, task_id)
    assert task["status"] == "completed", f"期望 completed，实际：{task.get('message')}"

    result = task["result"]
    assert result["success"] == 1
    assert result["failed"] == 0


# ============================================================================
# 测试 2：部分失败（VT_A 成功，VT_B 返回空）
# ============================================================================


def test_fundamental_refresh_partial_failure(client_with_mock_fundamental):
    """部分标的无数据 → 任务 completed，success 和 failed 数正确。"""
    client = client_with_mock_fundamental

    body = {
        "vt_symbols": [VT_A, VT_B],
        "start": START_DATE.isoformat(),
        "end": END_DATE.isoformat(),
    }
    resp = client.post("/api/strategy/fundamental/refresh", json=body)
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll(client, task_id)
    assert task["status"] == "completed", f"部分失败时任务不应为 failed：{task.get('message')}"

    result = task["result"]
    assert result["success"] == 1, f"期望 success=1，实际 {result['success']}"
    assert result["failed"] == 1, f"期望 failed=1，实际 {result['failed']}"
    assert len(result["failed_symbols"]) == 1


# ============================================================================
# 测试 3：全部失败 → FAILED + 中文 message
# ============================================================================


def test_fundamental_refresh_all_fail(client_all_fail):
    """所有标的无数据 → 任务 FAILED，message 含中文提示。"""
    client = client_all_fail

    body = {
        "vt_symbols": [VT_B, VT_C],
        "start": START_DATE.isoformat(),
        "end": END_DATE.isoformat(),
    }
    resp = client.post("/api/strategy/fundamental/refresh", json=body)
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll(client, task_id)
    assert task["status"] == "failed", f"全部失败时任务应为 failed，实际：{task['status']}"
    msg = task.get("message", "")
    # 中文提示应含"失败"或"基本面"字样
    assert "失败" in msg or "基本面" in msg, f"message 应含中文提示，实际：{msg!r}"


# ============================================================================
# 测试 4：回测结果含 universe_coverage 字段
# ============================================================================


def test_backtest_result_has_universe_coverage(tmp_path, monkeypatch):
    """全链路回测结果应含 universe_coverage 字段，且含必要子字段。"""
    import aitrade.rules  # noqa: F401

    DATA_START = date(2023, 1, 1)
    DATA_END = date(2024, 6, 1)
    BT_START = date(2023, 6, 1)
    BT_END = date(2024, 1, 1)
    VT_ETF_A = "510300.SSE"
    VT_ETF_B = "510500.SSE"

    lab = AlphaLab(tmp_path / "alpha_lab")
    n_days = (DATA_END - DATA_START).days + 1

    prices_a = [10.0 + i * 0.01 for i in range(n_days)]
    prices_b = [8.0 + (i % 30) * 0.02 for i in range(n_days)]

    lab.save_bar_frame(VT_ETF_A, "d", _make_bar_frame(prices_a, DATA_START))
    lab.save_bar_frame(VT_ETF_B, "d", _make_bar_frame(prices_b, DATA_START))

    monkeypatch.setattr(strategy_api, "_get_lab", lambda: AlphaLab(tmp_path / "alpha_lab"))

    app = create_app()
    with TestClient(app) as client:
        body = {
            "signal_source": "etf_momentum",
            "signal_params": {
                "universe": [VT_ETF_A, VT_ETF_B],
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
        assert task["status"] == "completed", f"任务失败：{task.get('message')}"

        result = task["result"]
        assert "universe_coverage" in result, "result 应含 universe_coverage 字段"

        cov = result["universe_coverage"]
        assert "requested" in cov
        assert "with_bars" in cov
        assert "with_fundamental" in cov
        assert "excluded_not_listed" in cov
        assert "coverage_ratio" in cov
        assert "warnings" in cov

        # 请求 2 只，信号有数据，覆盖率应 >= 0
        assert cov["requested"] == 2
        assert cov["with_bars"] >= 1
        assert isinstance(cov["coverage_ratio"], float)
