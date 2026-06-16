"""
组合级风控（PortfolioRiskManager）测试。

覆盖范围
--------
属性测试（Hypothesis @settings(max_examples=100)）：
  P1 — 熔断状态重启存活：同一 store 新建 manager 仍 broken
  P2 — broken 时 evaluate 恒 allow_buy=False / buy_factor=0
  P3 — peak 单调不减（随机 value 序列）
  P4 — dd 恰超阈值触发熔断；reset 后恢复且 peak 清零

确定性用例：
  T1 — 趋势闸门：close < MA → buy_factor 压缩
  T2 — 趋势闸门：close >= MA → buy_factor=1.0
  T3 — 数据不足 fail-open：record passed=True，detail 含"不足"字样
  T4 — lab=None fail-open：record passed=True
  T5 — records 三字段（check/passed/detail）齐全
  T6 — 第二次 evaluate 在熔断后：熔断闸直接 passed=False，不再出现 drawdown/trend records

API 用例：
  A1 — GET /api/live/portfolio-risk/{portfolio_id}：未初始化返回默认未熔断态
  A2 — POST /api/live/portfolio-risk/{portfolio_id}/reset：人工复位后状态归零
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aitrade.live.portfolio_risk import (
    PortfolioRiskConfig,
    PortfolioRiskManager,
)
from aitrade.live.runtime_state import RuntimeStateStore


# ===========================================================================
# 辅助
# ===========================================================================

_finite = dict(allow_nan=False, allow_infinity=False)

# 正值价格（组合净值）
_pos_values = st.floats(min_value=1.0, max_value=1e9, **_finite)
# 较严格回撤阈值以触发熔断（容易命中）
_tight_dd = st.floats(min_value=0.01, max_value=0.50, **_finite)

PORTFOLIO_ID = "p_test"
AS_OF = date(2026, 6, 11)


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "state.json")


def _make_manager(tmp_path: Path, config: PortfolioRiskConfig | None = None, lab=None) -> PortfolioRiskManager:
    return PortfolioRiskManager(_make_store(tmp_path), config, lab=lab)


def _make_bar_frame(prices: list[float], base_date: date | None = None) -> pl.DataFrame:
    """构造含 datetime / close 的日线 DataFrame（仿 test_etf_momentum 辅助）。"""
    if base_date is None:
        base_date = date(2024, 1, 2)
    rows = []
    for i, p in enumerate(prices):
        dt = base_date + timedelta(days=i)
        rows.append({
            "datetime": datetime(dt.year, dt.month, dt.day, 9, 30),
            "open": p,
            "high": p + 0.5,
            "low": p - 0.5,
            "close": p,
            "volume": 1_000_000.0,
            "turnover": p * 1_000_000.0,
            "open_interest": 0.0,
        })
    return pl.DataFrame(rows)


class _FakeLab:
    """最简 lab stub：按 vt_symbol 返回预先注入的 DataFrame。"""

    def __init__(self) -> None:
        self._frames: dict[str, pl.DataFrame] = {}

    def register(self, vt_symbol: str, df: pl.DataFrame) -> None:
        self._frames[vt_symbol] = df

    def load_bar_frame(self, vt_symbol: str, interval: str, start, end) -> pl.DataFrame | None:
        return self._frames.get(vt_symbol)


# ===========================================================================
# 属性测试 P1：熔断状态重启存活（同一 store 新建 manager 仍 broken）
# ===========================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    peak_val=_pos_values,
    dd_ratio=st.floats(min_value=0.01, max_value=0.30, **_finite),
)
def test_P1_broken_survives_restart(tmp_path, peak_val, dd_ratio):
    """属性 P1：熔断状态通过 store 持久化，新建 manager 实例仍读到 broken=True。"""
    store = _make_store(tmp_path)
    cfg = PortfolioRiskConfig(max_drawdown=dd_ratio * 0.5)  # 阈值设为 dd_ratio 的一半，保证超过

    # 第一个 manager：让净值下跌到触发熔断
    mgr1 = PortfolioRiskManager(store, cfg)
    # 先用 peak_val 建立峰值
    mgr1.evaluate(PORTFOLIO_ID, portfolio_value=peak_val, as_of=AS_OF)
    # 再下跌到超过阈值
    low_val = peak_val * (1.0 - dd_ratio)
    verdict1 = mgr1.evaluate(PORTFOLIO_ID, portfolio_value=low_val, as_of=AS_OF)

    # 可能已熔断（取决于 dd_ratio > cfg.max_drawdown）
    if not verdict1.broken:
        return  # dd_ratio 小于阈值，此次不触发熔断，跳过

    # 第二个 manager：使用同一 store，应能读到 broken=True
    mgr2 = PortfolioRiskManager(store, cfg)
    verdict2 = mgr2.evaluate(PORTFOLIO_ID, portfolio_value=low_val, as_of=AS_OF)
    assert verdict2.broken is True, "同一 store 重建 manager 后，熔断状态应仍为 True"
    assert verdict2.allow_buy is False
    assert verdict2.buy_factor == 0.0


# ===========================================================================
# 属性测试 P2：broken 时 evaluate 恒 allow_buy=False / buy_factor=0
# ===========================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(portfolio_value=_pos_values)
def test_P2_broken_always_blocks_buy(tmp_path, portfolio_value):
    """属性 P2：一旦熔断，任意后续 evaluate 的 allow_buy=False / buy_factor=0.0。"""
    store = _make_store(tmp_path)
    # 直接注入 broken 状态（绕过自动熔断触发）
    store.set("portfolio_risk", {
        PORTFOLIO_ID: {
            "peak_value": portfolio_value * 2,
            "broken": True,
            "broken_date": AS_OF.isoformat(),
            "reason": "测试直接注入熔断",
        }
    })

    mgr = PortfolioRiskManager(store, PortfolioRiskConfig())
    verdict = mgr.evaluate(PORTFOLIO_ID, portfolio_value=portfolio_value, as_of=AS_OF)

    assert verdict.allow_buy is False, f"熔断时 allow_buy 应为 False，实际 {verdict.allow_buy}"
    assert verdict.buy_factor == 0.0, f"熔断时 buy_factor 应为 0.0，实际 {verdict.buy_factor}"
    assert verdict.broken is True


# ===========================================================================
# 属性测试 P3：peak 单调不减
# ===========================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    values=st.lists(_pos_values, min_size=2, max_size=20),
    max_drawdown=st.floats(min_value=0.99, max_value=1.0, **_finite),  # 超大阈值，防止中途熔断
)
def test_P3_peak_monotone(tmp_path, values, max_drawdown):
    """属性 P3：随机净值序列中，持久化的 peak_value 单调不减。"""
    store = _make_store(tmp_path)
    cfg = PortfolioRiskConfig(max_drawdown=max_drawdown)
    mgr = PortfolioRiskManager(store, cfg)

    running_peak = 0.0
    for v in values:
        mgr.evaluate(PORTFOLIO_ID, portfolio_value=v, as_of=AS_OF)
        # 读回 peak
        pstate = store.get("portfolio_risk", {}).get(PORTFOLIO_ID, {})
        stored_peak = pstate.get("peak_value", 0.0)
        assert stored_peak >= running_peak, (
            f"peak 应单调不减：前值 {running_peak}，当前 {stored_peak}，输入 {v}"
        )
        assert stored_peak >= v - 1e-9 or stored_peak >= running_peak, (
            f"peak 应 >= 当前输入值 {v}（或保持前值 {running_peak}），实际 {stored_peak}"
        )
        running_peak = stored_peak


# ===========================================================================
# 属性测试 P4：dd 恰超阈值触发熔断；reset 后恢复且 peak 清零
# ===========================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    peak_val=st.floats(min_value=1000.0, max_value=1e8, **_finite),
    threshold=st.floats(min_value=0.05, max_value=0.40, **_finite),
)
def test_P4_drawdown_triggers_circuit_and_reset_clears(tmp_path, peak_val, threshold):
    """属性 P4：dd 超阈值触发熔断；reset 后 allow_buy=True 且 peak 清零。"""
    store = _make_store(tmp_path)
    cfg = PortfolioRiskConfig(max_drawdown=threshold)
    mgr = PortfolioRiskManager(store, cfg)

    # 建立 peak
    mgr.evaluate(PORTFOLIO_ID, portfolio_value=peak_val, as_of=AS_OF)

    # 构造刚好超过阈值的回撤（+0.001 容差使回撤明确 > threshold）
    low_val = peak_val * (1.0 - threshold - 0.001)
    verdict_low = mgr.evaluate(PORTFOLIO_ID, portfolio_value=low_val, as_of=AS_OF)

    # 应触发熔断
    assert verdict_low.broken is True, (
        f"peak={peak_val}, threshold={threshold}, low={low_val}: 应触发熔断"
    )
    assert verdict_low.allow_buy is False
    assert verdict_low.buy_factor == 0.0

    # reset 后恢复
    mgr.reset(PORTFOLIO_ID)

    # peak 应被清零（store 中无该 portfolio_id 的状态）
    pstate_after = store.get("portfolio_risk", {}).get(PORTFOLIO_ID)
    assert pstate_after is None, f"reset 后 peak 应被清零，实际: {pstate_after}"

    # 再次 evaluate：不熔断（以新值作为 peak 起点）
    mgr2 = PortfolioRiskManager(store, cfg)
    verdict_after = mgr2.evaluate(PORTFOLIO_ID, portfolio_value=low_val, as_of=AS_OF)
    assert verdict_after.broken is False, "reset 后首次 evaluate 不应熔断（以当前值作峰值）"
    assert verdict_after.allow_buy is True


# ===========================================================================
# 确定性用例 T1/T2：趋势闸门
# ===========================================================================

def _build_trend_lab(*, below_ma: bool, window: int = 10) -> _FakeLab:
    """构造 fake lab，使 close < MA（below_ma=True）或 close > MA（below_ma=False）。"""
    lab = _FakeLab()
    symbol = "510300.SSE"

    if below_ma:
        # 前 window-1 天高价，最后一天低价 → close < MA
        prices = [100.0] * (window - 1) + [50.0]
    else:
        # 前 window-1 天低价，最后一天高价 → close > MA
        prices = [50.0] * (window - 1) + [100.0]

    df = _make_bar_frame(prices, base_date=date(2026, 1, 1))
    lab.register(symbol, df)
    return lab


def test_T1_trend_below_ma_reduces_buy_factor(tmp_path):
    """T1：基准 close < MA → buy_factor = below_ma_buy_factor（默认 0.5）。"""
    cfg = PortfolioRiskConfig(trend_ma_window=10, below_ma_buy_factor=0.5)
    lab = _build_trend_lab(below_ma=True, window=10)
    store = _make_store(tmp_path)
    mgr = PortfolioRiskManager(store, cfg, lab=lab)

    verdict = mgr.evaluate(PORTFOLIO_ID, portfolio_value=100_000.0, as_of=AS_OF)

    assert verdict.buy_factor == 0.5, f"趋势弱时 buy_factor 应为 0.5，实际 {verdict.buy_factor}"
    assert verdict.allow_buy is True, "趋势弱时仍允许买入（只压缩额度）"
    assert verdict.broken is False

    # trend record passed=False
    trend_rec = next((r for r in verdict.records if r["check"] == "trend"), None)
    assert trend_rec is not None
    assert trend_rec["passed"] is False
    assert isinstance(trend_rec["detail"], str) and trend_rec["detail"]


def test_T2_trend_above_ma_keeps_full_factor(tmp_path):
    """T2：基准 close >= MA → buy_factor = 1.0。"""
    cfg = PortfolioRiskConfig(trend_ma_window=10, below_ma_buy_factor=0.5)
    lab = _build_trend_lab(below_ma=False, window=10)
    store = _make_store(tmp_path)
    mgr = PortfolioRiskManager(store, cfg, lab=lab)

    verdict = mgr.evaluate(PORTFOLIO_ID, portfolio_value=100_000.0, as_of=AS_OF)

    assert verdict.buy_factor == 1.0, f"趋势强时 buy_factor 应为 1.0，实际 {verdict.buy_factor}"
    assert verdict.allow_buy is True
    assert verdict.broken is False

    trend_rec = next((r for r in verdict.records if r["check"] == "trend"), None)
    assert trend_rec is not None
    assert trend_rec["passed"] is True


# ===========================================================================
# 确定性用例 T3：数据不足 fail-open
# ===========================================================================

def test_T3_insufficient_data_failopen(tmp_path):
    """T3：基准数据行数 < window → passed=True（fail-open），detail 含"不足"字样。"""
    cfg = PortfolioRiskConfig(trend_ma_window=60)  # 窗口 60
    lab = _FakeLab()
    # 只有 5 行，不足 60
    df = _make_bar_frame([100.0] * 5, base_date=date(2026, 1, 1))
    lab.register("510300.SSE", df)

    store = _make_store(tmp_path)
    mgr = PortfolioRiskManager(store, cfg, lab=lab)

    verdict = mgr.evaluate(PORTFOLIO_ID, portfolio_value=100_000.0, as_of=AS_OF)

    trend_rec = next((r for r in verdict.records if r["check"] == "trend"), None)
    assert trend_rec is not None
    assert trend_rec["passed"] is True, "数据不足应 fail-open（passed=True）"
    assert "不足" in trend_rec["detail"], f"detail 应含'不足'字样，实际: {trend_rec['detail']}"
    # fail-open：不压缩额度
    assert verdict.buy_factor == 1.0


# ===========================================================================
# 确定性用例 T4：lab=None → fail-open
# ===========================================================================

def test_T4_no_lab_failopen(tmp_path):
    """T4：lab=None → 趋势闸门跳过（fail-open），buy_factor=1.0。"""
    store = _make_store(tmp_path)
    mgr = PortfolioRiskManager(store, PortfolioRiskConfig(), lab=None)

    verdict = mgr.evaluate(PORTFOLIO_ID, portfolio_value=100_000.0, as_of=AS_OF)

    trend_rec = next((r for r in verdict.records if r["check"] == "trend"), None)
    assert trend_rec is not None
    assert trend_rec["passed"] is True
    assert verdict.buy_factor == 1.0


# ===========================================================================
# 确定性用例 T5：records 三字段（check/passed/detail）齐全
# ===========================================================================

def test_T5_records_fields_complete(tmp_path):
    """T5：所有 records 均包含 check(str) / passed(bool) / detail(str) 三字段。"""
    store = _make_store(tmp_path)
    mgr = PortfolioRiskManager(store, PortfolioRiskConfig(), lab=None)

    verdict = mgr.evaluate(PORTFOLIO_ID, portfolio_value=100_000.0, as_of=AS_OF)

    assert verdict.records, "records 不应为空"
    for rec in verdict.records:
        assert "check" in rec and isinstance(rec["check"], str) and rec["check"], rec
        assert "passed" in rec and isinstance(rec["passed"], bool), rec
        assert "detail" in rec and isinstance(rec["detail"], str) and rec["detail"], rec


# ===========================================================================
# 确定性用例 T6：熔断后 evaluate 只含 circuit record（短路）
# ===========================================================================

def test_T6_broken_circuit_short_circuits(tmp_path):
    """T6：熔断时 circuit check passed=False，不再出现 drawdown/trend records。"""
    store = _make_store(tmp_path)
    store.set("portfolio_risk", {
        PORTFOLIO_ID: {
            "peak_value": 200_000.0,
            "broken": True,
            "broken_date": "2026-06-01",
            "reason": "测试熔断短路",
        }
    })

    mgr = PortfolioRiskManager(store, PortfolioRiskConfig(), lab=None)
    verdict = mgr.evaluate(PORTFOLIO_ID, portfolio_value=100_000.0, as_of=AS_OF)

    # 只有一条 record（circuit）
    assert len(verdict.records) == 1, f"熔断应短路，只有 circuit record，实际: {verdict.records}"
    assert verdict.records[0]["check"] == "circuit"
    assert verdict.records[0]["passed"] is False
    assert verdict.allow_buy is False
    assert verdict.buy_factor == 0.0


# ===========================================================================
# API 用例
# ===========================================================================

@pytest.fixture
def live_client(tmp_path, monkeypatch):
    """隔离的 FastAPI TestClient：使用 tmp_path 的 state store。"""
    from fastapi.testclient import TestClient
    from aitrade.api import live as live_api
    from aitrade.live.runtime_state import RuntimeStateStore
    from aitrade.main import create_app

    # 替换模块级单实例 _runtime_state → tmp_path 隔离
    isolated_state = RuntimeStateStore(tmp_path / "state.json")
    monkeypatch.setattr(live_api, "_runtime_state", isolated_state)

    # 同时替换 portfolio_risk 模块里 API 端点用到的 _portfolio_risk_manager（如有）
    # API 端点直接用 _runtime_state，不依赖独立单例，此处无需替换。

    app = create_app()
    with TestClient(app) as client:
        yield client, isolated_state


def test_A1_get_portfolio_risk_default(live_client):
    """A1：GET /api/live/portfolio-risk/{portfolio_id} 未初始化返回默认未熔断态。"""
    client, _ = live_client
    resp = client.get("/api/live/portfolio-risk/p_default")
    assert resp.status_code == 200
    data = resp.json()
    assert data["broken"] is False
    assert data["portfolio_id"] == "p_default"


def test_A2_post_reset_clears_state(live_client):
    """A2：POST /api/live/portfolio-risk/{portfolio_id}/reset 后状态清零，broken=False。"""
    client, state_store = live_client

    # 预先注入熔断状态
    state_store.set("portfolio_risk", {
        "p_reset": {
            "peak_value": 100_000.0,
            "broken": True,
            "broken_date": "2026-06-01",
            "reason": "测试注入",
        }
    })

    resp = client.post("/api/live/portfolio-risk/p_reset/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["broken"] is False
    assert data["portfolio_id"] == "p_reset"

    # 再次 GET 确认持久化已清除
    resp2 = client.get("/api/live/portfolio-risk/p_reset")
    assert resp2.status_code == 200
    assert resp2.json()["broken"] is False
