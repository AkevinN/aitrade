"""
CNN 选股 Tier-2 可配置 + 数据充足性预检测试。

覆盖本次"薄数据下晦涩报错"修复引入的三块新能力：

- **窗口解析器 `_resolve_tier2_window`**：请求级覆盖（train_days/fold_test_days/
  eval_window_days/n_seeds）优先，None 回退 ScreeningRules 默认；窗口 ``start``
  被夹进本地数据范围 ``max(desired_start, local_start)``，``end <= as_of`` 仍成立。
- **数据充足性预检**：本地缺数据或可用天数 < train_days + fold_test_days 时，
  Tier-2 直接产出 ``Tier2Verdict(evaluable=False, note="数据不足…")``，
  **不**调用 ``run_walk_forward_evaluate``（杜绝 governance 抛出的晦涩 load error）。
- **配置不变量校验**：解析后 ``eval_window_days < train_days + fold_test_days``
  时 ``run()`` 直接抛清晰 ``ValueError`` 快速失败。
- **配置回显**：``ScreeningResult.input["tier2_config"]`` 携带解析后超参，便于复现。

全部桩化 ``run_walk_forward_evaluate`` / ``Profiler.profile``（不跑真实训练），
沿用 test_runner.py 的 FakeLab / _make_profile 范式。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import polars as pl
import pytest

from aitrade.models.governance import CNNWalkForwardRequest
from aitrade.models.screening import CNNScreeningRequest
from aitrade.profiling.types import (
    MetricBlock,
    MetricValue,
    ProfileInput,
    SymbolProfile,
)
from aitrade.screening import runner as runner_mod
from aitrade.screening.rules import DEFAULT_SCREENING_RULES, ScreeningRules
from aitrade.screening.runner import ScreeningRunner

_AS_OF = datetime(2026, 1, 6)


# ---------------------------------------------------------------------------
# 测试辅助：FakeLab（universe 发现 + 可配置本地范围探测）
# ---------------------------------------------------------------------------


class FakeLab:
    """最小伪 AlphaLab：暴露 list_data_resources 与 load_bar_frame_any_range。

    与 test_runner.py 的 FakeLab 相比，本测试需要控制本地范围的**两端**
    （local_start + local_end），用于验证窗口夹取与数据充足性预检。

    Args:
        symbols: 进入 raw_bars 的标的列表。
        local_start: load_bar_frame_any_range 返回的本地最早 datetime；
            与 local_end 配合界定本地可用范围。None 且 local_end 也 None 时返回空 frame。
        local_end: 本地最晚 datetime。
    """

    def __init__(
        self,
        symbols: list[str],
        local_start: datetime | None = None,
        local_end: datetime | None = None,
    ) -> None:
        self._symbols = symbols
        self._local_start = local_start
        self._local_end = local_end

    def list_data_resources(self) -> dict[str, Any]:
        """返回合成数据资源摘要（仅 raw_bars 有内容）。"""
        return {
            "raw_bars": [
                {
                    "vt_symbol": s,
                    "interval": "30m",
                    "row_count": 500,
                    "start": "2025-01-06",
                    "end": "2026-01-06",
                    "kind": "raw_bar",
                }
                for s in self._symbols
            ],
            "raw_ticks": [],
            "raw_bar_batches": [],
            "raw_tick_batches": [],
            "derived_bars": [],
            "raw_bar_intervals": [],
            "derived_intervals": [],
        }

    def load_bar_frame_any_range(
        self, vt_symbol: str, interval: str, include_derived: bool = True
    ) -> pl.DataFrame | None:
        """供 load_local_range 探测本地范围；返回含 start/end 两行 datetime 的最小 frame。"""
        if self._local_start is None and self._local_end is None:
            return None
        dts = [d for d in (self._local_start, self._local_end) if d is not None]
        return pl.DataFrame({"datetime": dts, "close": [10.0] * len(dts)})


# ---------------------------------------------------------------------------
# 测试辅助：合成 SymbolProfile / 窗口 frame / 请求 / 桩
# ---------------------------------------------------------------------------


def _make_profile(vt_symbol: str, *, available: bool = True) -> SymbolProfile:
    """构造四块齐全、available 可控的合成 SymbolProfile（high 置信、可入围）。"""
    inp = ProfileInput(
        vt_symbol=vt_symbol,
        interval="30m",
        as_of=_AS_OF,
        lookback_days=365,
        effective_bar_count=300 if available else 0,
        effective_right_bound=_AS_OF if available else None,
        rules_id="builtin-v1",
    )
    blocks: list[MetricBlock] = []
    if available:
        blocks = [
            MetricBlock(
                block="data_quality",
                metrics=[
                    MetricValue(key="count_valid_bars", value=300.0, effective_sample=300, confidence="high")
                ],
            ),
            MetricBlock(
                block="liquidity",
                metrics=[MetricValue(key="avg_turnover", value=1e7, effective_sample=300, confidence="high")],
                level="high",
            ),
            MetricBlock(
                block="volatility",
                metrics=[MetricValue(key="realized_volatility", value=0.02, effective_sample=300, confidence="high")],
                level="high",
            ),
            MetricBlock(
                block="predictability",
                metrics=[MetricValue(key="hurst_exponent", value=0.6, effective_sample=300, confidence="high")],
                level="trending",
            ),
        ]
    return SymbolProfile(
        input=inp,
        available=available,
        unavailable_reason=None if available else "本地无数据",
        blocks=blocks,
        overall_confidence="high" if available else "insufficient",
    )


def _synthetic_window() -> pl.DataFrame:
    """构造确定性窗口 frame（trend + 波动），供 Tier-1 代理指标计算。"""
    import numpy as np

    n = 300
    rng = np.random.default_rng(42)
    rets = 0.001 + 0.01 * rng.standard_normal(n)
    close = 10.0 * np.exp(np.cumsum(rets))
    dts = [datetime(2025, 6, 1) for _ in range(n)]
    return pl.DataFrame({"datetime": dts, "close": close.tolist()})


def _make_request(**overrides: Any) -> CNNScreeningRequest:
    """构造默认 CNNScreeningRequest（30m / run_tier2=True），可覆盖任意字段。"""
    params: dict[str, Any] = {
        "name": "tier2-config-test",
        "interval": "30m",
        "as_of": _AS_OF,
        "lookback_days": 365,
        "min_bar_count": 100,
        "top_k": 3,
        "run_tier2": True,
    }
    params.update(overrides)
    return CNNScreeningRequest(**params)


def _patch_profiler(monkeypatch, syms: list[str], *, available: bool = True) -> None:
    """monkeypatch runner 内的 Profiler，使其按 vt_symbol 返回 available 画像。"""

    profile_map = {s: _make_profile(s, available=available) for s in syms}

    class _StubProfiler:
        def __init__(self, lab: Any) -> None:
            self.lab = lab

        def profile(self, *, vt_symbol: str, **_kw: Any) -> SymbolProfile:
            return profile_map[vt_symbol]

    monkeypatch.setattr(runner_mod, "Profiler", _StubProfiler)
    monkeypatch.setattr(runner_mod, "_load_window_frame", lambda *a, **k: _synthetic_window())


def _patch_wf(monkeypatch, captured: list[CNNWalkForwardRequest], score: float = 1.0):
    """monkeypatch run_walk_forward_evaluate：捕获请求并返回合成正分 WF 报告。"""

    def _fake(req: CNNWalkForwardRequest, on_progress=None, store=None):  # noqa: ANN001
        captured.append(req)
        if on_progress:
            on_progress(100, "stub done")
        return {
            "report_id": f"wf_{req.target_symbol}",
            "request": {"target_symbol": req.target_symbol},
            "folds": [
                {"candidate_score": score, "cross_seed": {"std": 0.1}},
                {"candidate_score": score, "cross_seed": {"std": 0.1}},
            ],
            "summary": {"avg_cross_seed_std": 0.1},
        }

    monkeypatch.setattr(runner_mod, "run_walk_forward_evaluate", _fake)


# ===========================================================================
# 关键验收：用户实际命中的"薄数据"场景被清晰跳过
# Feature: cnn-stock-screening, Tier-2 data-sufficiency precheck
# ===========================================================================


def test_thin_data_symbol_skipped_with_clear_note(monkeypatch) -> None:
    # Feature: cnn-stock-screening, Tier-2 data-sufficiency precheck
    """本地仅约 365 天 30m 数据 + 默认规则 → Tier-2 跳过，note 含"数据不足"与天数，
    且**不**调用 run_walk_forward_evaluate（不触发 governance 的晦涩 load error）。

    复现真实失败路径：local [2025-01-06, 2026-01-06]=365 天，
    默认 eval_window_days=900 → desired_start=2023-07，被夹回 local_start=2025-01-06，
    available_days≈365 < needed=train(480)+test(90)=570 → 预检判定不足。
    """
    syms = ["000001.SZSE"]
    _patch_profiler(monkeypatch, syms)
    captured: list[CNNWalkForwardRequest] = []
    _patch_wf(monkeypatch, captured)

    lab = FakeLab(syms, local_start=datetime(2025, 1, 6), local_end=datetime(2026, 1, 6))
    runner = ScreeningRunner(lab)  # 默认规则
    result = runner.run(_make_request(top_k=1))

    # 预检跳过：WF 绝不被调用
    assert captured == []

    row = next(r for r in result.leaderboard if r.tier1.vt_symbol == "000001.SZSE")
    assert row.promoted_to_tier2 is True  # Tier-1 仍入围（资格满足）
    assert row.tier2 is not None
    assert row.tier2.evaluable is False
    assert row.tier2.edge_ok is False
    note = row.tier2.note or ""
    assert "数据不足" in note
    # note 含真实天数：可用约 365、最少需 570、train 480 / test 90
    assert "570" in note  # needed = train + test
    assert "480" in note and "90" in note


def test_missing_local_data_skipped_with_clear_note(monkeypatch) -> None:
    # Feature: cnn-stock-screening, Tier-2 data-sufficiency precheck
    """本地完全无数据（load_local_range 返回 (None, None)）→ Tier-2 跳过 + 数据不足 note，
    不调用 WF。"""
    syms = ["000001.SZSE"]
    _patch_profiler(monkeypatch, syms)
    captured: list[CNNWalkForwardRequest] = []
    _patch_wf(monkeypatch, captured)

    lab = FakeLab(syms, local_start=None, local_end=None)
    runner = ScreeningRunner(lab)
    result = runner.run(_make_request(top_k=1))

    assert captured == []
    row = next(r for r in result.leaderboard if r.tier1.vt_symbol == "000001.SZSE")
    assert row.tier2 is not None and row.tier2.evaluable is False
    assert "数据不足" in (row.tier2.note or "")


# ===========================================================================
# 充足数据 → 预检通过、WF 被调用、edge 结论产出
# ===========================================================================


def test_sufficient_data_runs_wf_and_derives_edge(monkeypatch) -> None:
    """本地充足（约 3 年）→ 预检通过、调用 WF、产出 evaluable edge 结论。"""
    syms = ["000001.SZSE"]
    _patch_profiler(monkeypatch, syms)
    captured: list[CNNWalkForwardRequest] = []
    _patch_wf(monkeypatch, captured)

    lab = FakeLab(syms, local_start=datetime(2022, 1, 1), local_end=_AS_OF)
    runner = ScreeningRunner(lab)
    result = runner.run(_make_request(top_k=1))

    assert len(captured) == 1
    row = next(r for r in result.leaderboard if r.tier1.vt_symbol == "000001.SZSE")
    assert row.tier2 is not None
    assert row.tier2.evaluable is True
    assert row.tier2.edge_ok is True  # 正分


# ===========================================================================
# 请求级覆盖解析：覆盖优先、None 回退规则
# ===========================================================================


def test_request_overrides_resolved_into_wf_request(monkeypatch) -> None:
    """请求 train_days/fold_test_days/eval_window_days/n_seeds 覆盖规则，
    构造的 CNNWalkForwardRequest 携带解析后的值。"""
    syms = ["000001.SZSE"]
    _patch_profiler(monkeypatch, syms)
    captured: list[CNNWalkForwardRequest] = []
    _patch_wf(monkeypatch, captured)

    # 本地足够大，避免被预检卡住；覆盖更小的窗口
    lab = FakeLab(syms, local_start=datetime(2020, 1, 1), local_end=_AS_OF)
    runner = ScreeningRunner(lab)
    result = runner.run(
        _make_request(
            top_k=1,
            train_days=120,
            fold_test_days=30,
            eval_window_days=300,
            n_seeds=3,
        )
    )

    assert len(captured) == 1
    wf = captured[0]
    assert wf.train_days == 120
    assert wf.test_days == 30
    assert wf.n_seeds == 3
    # eval_window_days=300 → start = end - 300
    assert (wf.end - wf.start).days == 300
    # tier2_config 回显解析后的值
    cfg = result.input["tier2_config"]
    assert cfg["train_days"] == 120
    assert cfg["fold_test_days"] == 30
    assert cfg["eval_window_days"] == 300
    assert cfg["n_seeds"] == 3


def test_none_overrides_fall_back_to_rules(monkeypatch) -> None:
    """请求未提供覆盖（None）→ 解析后取 ScreeningRules 默认值。"""
    syms = ["000001.SZSE"]
    _patch_profiler(monkeypatch, syms)
    captured: list[CNNWalkForwardRequest] = []
    _patch_wf(monkeypatch, captured)

    lab = FakeLab(syms, local_start=datetime(2020, 1, 1), local_end=_AS_OF)
    runner = ScreeningRunner(lab)  # 默认规则
    result = runner.run(_make_request(top_k=1))  # 不传任何覆盖

    wf = captured[0]
    rules = DEFAULT_SCREENING_RULES
    assert wf.train_days == rules.train_days
    assert wf.test_days == rules.fold_test_days
    assert wf.n_seeds == rules.n_seeds
    cfg = result.input["tier2_config"]
    assert cfg["train_days"] == rules.train_days
    assert cfg["fold_test_days"] == rules.fold_test_days
    assert cfg["eval_window_days"] == rules.eval_window_days
    assert cfg["n_seeds"] == rules.n_seeds
    assert cfg["step_days"] == rules.step_days
    assert cfg["epochs"] == rules.epochs
    assert cfg["objective"] == "classification"


# ===========================================================================
# 窗口夹取：desired_start < local_start → start == local_start，end <= as_of
# Feature: cnn-stock-screening, Property 2: 无前视
# ===========================================================================


def test_window_clamped_to_local_start(monkeypatch) -> None:
    # Feature: cnn-stock-screening, Property 2: 无前视
    """desired_start 早于 local_start 时窗口左界被夹到 local_start，end 仍 <= as_of。"""
    syms = ["000001.SZSE"]
    _patch_profiler(monkeypatch, syms)
    captured: list[CNNWalkForwardRequest] = []
    _patch_wf(monkeypatch, captured)

    # 本地 [2024-01-01, as_of]：约 2 年，足够 train(120)+test(30)=150；
    # 但 eval_window_days=900 → desired_start ~ 2023-07 < local_start 2024-01-01 → 夹取
    local_start = datetime(2024, 1, 1)
    lab = FakeLab(syms, local_start=local_start, local_end=_AS_OF)
    runner = ScreeningRunner(lab)
    runner.run(
        _make_request(top_k=1, train_days=120, fold_test_days=30, eval_window_days=900)
    )

    wf = captured[0]
    assert wf.start == local_start.date()  # 夹到 local_start
    assert wf.end <= _AS_OF.date()  # Property 2 仍成立
    assert wf.start < wf.end


def test_explicit_eval_start_also_clamped(monkeypatch) -> None:
    """显式 eval_start 早于 local_start 时同样被夹到 local_start。"""
    syms = ["000001.SZSE"]
    _patch_profiler(monkeypatch, syms)
    captured: list[CNNWalkForwardRequest] = []
    _patch_wf(monkeypatch, captured)

    local_start = datetime(2024, 1, 1)
    lab = FakeLab(syms, local_start=local_start, local_end=_AS_OF)
    runner = ScreeningRunner(lab)
    runner.run(
        _make_request(
            top_k=1,
            train_days=120,
            fold_test_days=30,
            eval_window_days=300,
            eval_start=date(2023, 1, 1),  # 早于 local_start
        )
    )

    wf = captured[0]
    assert wf.start == local_start.date()


# ===========================================================================
# 配置不变量校验：eval_window_days < train_days + fold_test_days → 清晰 ValueError
# ===========================================================================


def test_config_invariant_violation_raises_clear_error(monkeypatch) -> None:
    """eval_window_days=100 而 train_days=480 → run() 抛清晰 ValueError 快速失败。"""
    syms = ["000001.SZSE"]
    _patch_profiler(monkeypatch, syms)

    lab = FakeLab(syms, local_start=datetime(2020, 1, 1), local_end=_AS_OF)
    runner = ScreeningRunner(lab)  # 默认 train_days=480, fold_test_days=90

    with pytest.raises(ValueError) as exc_info:
        runner.run(_make_request(top_k=1, eval_window_days=100))

    msg = str(exc_info.value)
    assert "Tier-2 配置无效" in msg
    assert "100" in msg  # eval_window_days
    assert "480" in msg  # train_days
    assert "90" in msg  # fold_test_days


# ===========================================================================
# tier2_config 回显始终存在（即便预检跳过）
# ===========================================================================


def test_tier2_config_echo_present_even_when_skipped(monkeypatch) -> None:
    """预检跳过的薄数据场景下 tier2_config 回显仍存在（审计/复现）。"""
    syms = ["000001.SZSE"]
    _patch_profiler(monkeypatch, syms)
    captured: list[CNNWalkForwardRequest] = []
    _patch_wf(monkeypatch, captured)

    lab = FakeLab(syms, local_start=datetime(2025, 1, 6), local_end=datetime(2026, 1, 6))
    runner = ScreeningRunner(lab)
    result = runner.run(_make_request(top_k=1))

    assert captured == []  # 预检跳过
    cfg = result.input["tier2_config"]
    assert cfg["train_days"] == DEFAULT_SCREENING_RULES.train_days
    assert cfg["eval_window_days"] == DEFAULT_SCREENING_RULES.eval_window_days


def test_resolve_window_returns_none_when_no_local_data(monkeypatch) -> None:
    """_resolve_tier2_window：本地无数据时返回 None（供预检判定数据不足）。"""
    syms = ["000001.SZSE"]
    lab = FakeLab(syms, local_start=None, local_end=None)
    runner = ScreeningRunner(lab)
    req = _make_request(top_k=1)
    window = runner._resolve_tier2_window("000001.SZSE", req)
    assert window is None
