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


# ---- 安全加固：signal_name 路径穿越 + 网格规模封顶 ----

def test_load_signal_provider_rejects_traversal_names(monkeypatch) -> None:
    """伪造 signal_name（穿越/绝对路径/未知）绝不触达 load_signal；只加载白名单内信号。"""
    import aitrade.alpha as alpha_mod

    calls: list[str] = []

    class _FakeLab:
        def __init__(self, _p): pass
        def list_all_signals(self): return ['realsig']
        def load_signal(self, name): calls.append(name); return None

    monkeypatch.setattr(alpha_mod, 'AlphaLab', _FakeLab)
    sp = t0._load_signal_provider('000415.SZSE',
                                  ['../../../etc/passwd', '/abs/secret', 'realsig', 'unknown'])
    assert calls == ['realsig']     # 仅白名单内被加载；伪造名从不触达文件系统
    assert sp is None               # realsig 返回空帧 → 无 frames → None（优雅降级）


def test_oversized_tick_policies_rejected_422() -> None:
    """tick_policies 超 max_length → 422（请求校验层拦截，防组合爆炸）。"""
    policies = [{"kind": "fixed", "label": f"p{i}", "sell_tick": 0.02, "buy_tick": 0.02} for i in range(21)]
    with TestClient(create_app()) as c:
        r = c.post("/api/t0/backtest", json={"symbol": "AAA.SSE", "start": "2025-01-01",
                                             "end": "2025-01-31", "tick_policies": policies})
    assert r.status_code == 422


def test_grid_product_capped_400(monkeypatch, tmp_path) -> None:
    """策略 × 成交假设 组合数超上限 → 400（每格都是一次完整回测）。"""
    _patch(monkeypatch, tmp_path)
    policies = [{"kind": "fixed", "label": f"p{i}", "sell_tick": 0.02, "buy_tick": 0.02} for i in range(12)]
    fills = [{"penetration": 0.0, "ratio": round(0.1 + i * 0.01, 2)} for i in range(11)]  # 11 个 → 12×11=132>120
    with TestClient(create_app()) as c:
        r = c.post("/api/t0/backtest", json={"symbol": "AAA.SSE", "start": "2025-01-01",
                                             "end": "2025-01-31", "tick_policies": policies, "fill_grid": fills})
    assert r.status_code == 400


# ---- 分场景画像端点 /profile_segmented ----

def _patch_profile(monkeypatch, tmp_path, daily) -> None:
    """打桩：parquet 存在 + load_daily_from_1m 返回指定日线（画像端点用真实 profiler 计算）。"""
    p = tmp_path / "bars" / "1m"
    p.mkdir(parents=True, exist_ok=True)
    (p / "AAA.SSE.parquet").write_bytes(b"x")
    monkeypatch.setattr(t0, "ALPHA_LAB_PATH", tmp_path)
    monkeypatch.setattr(t0, "load_daily_from_1m", lambda *a, **k: daily)


def _gap_daily():
    """7 日日线：高2/低2/平2（首日占位）。"""
    rows = []
    prev = None
    for i, g in enumerate([0.0, 0.01, -0.01, 0.0, 0.006, -0.006, 0.001]):
        o = 10.0 if prev is None else round(prev * (1 + g), 4)
        c = round(o + 0.03, 4)
        rows.append({"d": date(2025, 1, 1) + (date(2025, 1, 1 + i) - date(2025, 1, 1)),
                     "open": o, "high": max(o, c) + 0.05, "low": min(o, c) - 0.05, "close": c})
        prev = c
    return pl.DataFrame(rows)


def test_profile_segmented_returns_three_regimes(monkeypatch, tmp_path) -> None:
    """/profile_segmented 返回 高/低/平开 三段，各带 n_days；和=总日数−1。"""
    _patch_profile(monkeypatch, tmp_path, _gap_daily())
    with TestClient(create_app()) as c:
        r = c.post("/api/t0/profile_segmented", json={"symbol": "AAA.SSE", "start": "2025-01-01",
                                                      "end": "2025-12-31", "gap_thresh": 0.003})
    assert r.status_code == 200
    body = r.json()
    assert [s["regime"] for s in body["segments"]] == ["high", "low", "flat"]
    assert sum(s["n_days"] for s in body["segments"]) == 7 - 1
    assert all("profile" in s and "suggested_sell_tick" in s["profile"] for s in body["segments"])


def test_profile_segmented_insufficient_days_400(monkeypatch, tmp_path) -> None:
    """有效日不足（<6）→ 400。"""
    small = _gap_daily().head(3)
    _patch_profile(monkeypatch, tmp_path, small)
    with TestClient(create_app()) as c:
        r = c.post("/api/t0/profile_segmented", json={"symbol": "AAA.SSE", "start": "2025-01-01", "end": "2025-12-31"})
    assert r.status_code == 400


# ---- 画像端点返回逐日 OHLC bars（供前端画 K 线 + 标买卖腿） ----

def test_profile_returns_window_bars(monkeypatch, tmp_path) -> None:
    """/profile 额外返回标定窗逐日 OHLC bars。"""
    _patch_profile(monkeypatch, tmp_path, _gap_daily())
    with TestClient(create_app()) as c:
        r = c.post("/api/t0/profile", json={"symbol": "AAA.SSE", "start": "2025-01-01",
                                            "end": "2025-12-31", "x_max_fen": 15})
    assert r.status_code == 200
    bars = r.json()["bars"]
    assert len(bars) == 7
    assert set(bars[0]) == {"d", "open", "high", "low", "close"}
    assert bars[0]["d"] <= bars[-1]["d"]   # 升序


def test_profile_segmented_returns_window_bars(monkeypatch, tmp_path) -> None:
    """/profile_segmented 顶层额外返回整窗逐日 OHLC bars。"""
    _patch_profile(monkeypatch, tmp_path, _gap_daily())
    with TestClient(create_app()) as c:
        r = c.post("/api/t0/profile_segmented", json={"symbol": "AAA.SSE", "start": "2025-01-01", "end": "2025-12-31"})
    assert r.status_code == 200
    assert len(r.json()["bars"]) == 7
    assert "segments" in r.json()
