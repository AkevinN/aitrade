"""做 T 回测 API 的多策略 / 条件规则 / 信号接线测试。

Feature: t0-conditional-tick-frontend
- 多档位策略解析与 label 唯一（Property 5）；声明判别联合解析、未知 kind 拒绝（Property 3/6）；
- 信号规则触发信号源加载并下传 runner；不传 tick_policies 回退单 FixedTick（Property 7）；GET /signals。
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from pydantic import ValidationError

import aitrade.api.t0 as t0
from aitrade.api.t0 import T0BacktestRequest, _compile_tick_policies
from aitrade.main import create_app


# ---- 纯单元：编译 + label 唯一 + 联合解析 ----

def test_compile_tick_policies_multi_and_names() -> None:
    req = T0BacktestRequest(symbol="A.SZSE", start="2025-01-01", end="2025-06-30", tick_policies=[
        {"kind": "fixed", "label": "固定2分", "sell_tick": 0.02, "buy_tick": 0.02},
        {"kind": "conditional", "label": "信号", "default_sell_tick": 0.03, "default_buy_tick": 0.03,
         "rules": [{"lhs": "signal", "op": "gt", "threshold": 0.6, "signal_name": "mdl", "sell_tick": 0.08, "buy_tick": 0.01}]},
    ])
    policies, names = _compile_tick_policies(req.tick_policies)
    assert [lbl for lbl, _ in policies] == ["固定2分", "信号"]
    assert names == ["mdl"]


def test_compile_tick_policies_duplicate_label_raises_400() -> None:
    req = T0BacktestRequest(symbol="A.SZSE", start="2025-01-01", end="2025-06-30", tick_policies=[
        {"kind": "fixed", "label": "dup", "sell_tick": 0.02, "buy_tick": 0.02},
        {"kind": "fixed", "label": "dup", "sell_tick": 0.03, "buy_tick": 0.03},
    ])
    with pytest.raises(HTTPException) as ei:
        _compile_tick_policies(req.tick_policies)
    assert ei.value.status_code == 400


def test_request_union_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        T0BacktestRequest(symbol="A.SZSE", start="2025-01-01", end="2025-06-30",
                          tick_policies=[{"kind": "evil", "label": "x"}])


# ---- HTTP：用假 runner 捕获实参，绕开真实数据/引擎 ----

class _CaptureRunner:
    captured: dict = {}

    def run(self, symbol, start, end, daily, tick_policies=None, fill_grid=None,
            signal_provider=None, **kw):
        type(self).captured = {"tick_policies": tick_policies, "signal_provider": signal_provider}

        class _Rep:
            def to_dict(self_inner):
                return {"symbol": symbol, "eval_window": [str(start), str(end)],
                        "fill_sensitivity": [], "results": []}
        return _Rep()


def _patch(monkeypatch, tmp_path) -> None:
    """打桩：存在性检查的 parquet、日线加载、runner——使端点不触真实数据/引擎。"""
    p = tmp_path / "bars" / "1m"
    p.mkdir(parents=True, exist_ok=True)
    (p / "AAA.SSE.parquet").write_bytes(b"x")
    daily = pl.DataFrame({"d": [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4)],
                          "open": [10.0, 10.1, 10.2], "high": [10.1, 10.2, 10.3],
                          "low": [9.9, 10.0, 10.1], "close": [10.05, 10.15, 10.25]})
    monkeypatch.setattr(t0, "ALPHA_LAB_PATH", tmp_path)
    monkeypatch.setattr(t0, "load_daily_from_1m", lambda *a, **k: daily)
    monkeypatch.setattr(t0, "T0BacktestRunner", _CaptureRunner)
    _CaptureRunner.captured = {}


def test_backtest_fallback_single_fixed(monkeypatch, tmp_path) -> None:
    """不传 tick_policies → 回退单 FixedTick、无信号源（向后兼容）。"""
    _patch(monkeypatch, tmp_path)
    with TestClient(create_app()) as c:
        r = c.post("/api/t0/backtest", json={"symbol": "AAA.SSE", "start": "2025-01-01", "end": "2025-01-31"})
    assert r.status_code == 200
    assert len(_CaptureRunner.captured["tick_policies"]) == 1
    assert _CaptureRunner.captured["signal_provider"] is None


def test_backtest_multi_policy_passed_to_runner(monkeypatch, tmp_path) -> None:
    """多策略 + 信号规则 → runner 收到 N 个策略，且加载到信号源。"""
    _patch(monkeypatch, tmp_path)
    monkeypatch.setattr(t0, "_load_signal_provider", lambda symbol, names: ("SP" if names else None))
    body = {"symbol": "AAA.SSE", "start": "2025-01-01", "end": "2025-01-31", "tick_policies": [
        {"kind": "fixed", "label": "固定2分", "sell_tick": 0.02, "buy_tick": 0.02},
        {"kind": "conditional", "label": "信号择时", "default_sell_tick": 0.03, "default_buy_tick": 0.03,
         "rules": [{"lhs": "signal", "op": "gt", "threshold": 0.6, "signal_name": "mdl", "sell_tick": 0.08, "buy_tick": 0.01}]},
    ]}
    with TestClient(create_app()) as c:
        r = c.post("/api/t0/backtest", json=body)
    assert r.status_code == 200
    assert len(_CaptureRunner.captured["tick_policies"]) == 2
    assert _CaptureRunner.captured["signal_provider"] == "SP"


def test_backtest_duplicate_label_returns_400(monkeypatch, tmp_path) -> None:
    _patch(monkeypatch, tmp_path)
    body = {"symbol": "AAA.SSE", "start": "2025-01-01", "end": "2025-01-31", "tick_policies": [
        {"kind": "fixed", "label": "dup", "sell_tick": 0.02, "buy_tick": 0.02},
        {"kind": "fixed", "label": "dup", "sell_tick": 0.03, "buy_tick": 0.03},
    ]}
    with TestClient(create_app()) as c:
        r = c.post("/api/t0/backtest", json=body)
    assert r.status_code == 400


def test_list_t0_signals_returns_list(monkeypatch, tmp_path) -> None:
    with TestClient(create_app()) as c:
        r = c.get("/api/t0/signals")
    assert r.status_code == 200
    assert isinstance(r.json()["names"], list)
