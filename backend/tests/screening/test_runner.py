"""
CNN 选股编排器 ScreeningRunner 测试。

全部测试**桩化重/外部依赖**，不跑真实训练、不要求真实行情：

- ``FakeLab``：仅实现 ``list_data_resources()`` 供 ``discover_universe`` 发现 universe，
  以及 ``load_bar_frame_any_range``（供 ``load_local_range`` 探测本地范围）。
- monkeypatch ``aitrade.screening.runner.Profiler`` → 返回合成 ``SymbolProfile``。
- monkeypatch ``aitrade.screening.runner._load_window_frame`` → 返回合成窗口 frame，
  使 Tier-1 代理指标确定性（或被画像 stub 直接覆盖）。
- monkeypatch ``aitrade.screening.runner.run_walk_forward_evaluate`` → 返回合成 WF 报告
  并捕获所构造的 ``CNNWalkForwardRequest``（用于 Property 2 无前视断言）。

覆盖：
- Property 6（入围选择正确）
- Property 10（批量鲁棒性：单只失败降级、空池结构化）
- Property 2（无前视：WF 请求 end <= as_of）
- run_tier2=False 仅 Tier-1
- 端到端（含桩 Tier-2）：榜单排序、eval_window、status draft
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import polars as pl

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
from aitrade.screening.types import ScreeningResult, Tier1Score

_AS_OF = datetime(2025, 6, 1)


# ---------------------------------------------------------------------------
# 测试辅助：FakeLab（universe 发现 + 本地范围探测）
# ---------------------------------------------------------------------------


class FakeLab:
    """最小伪 AlphaLab：暴露 list_data_resources 与 load_bar_frame_any_range。

    Args:
        symbols: 进入 ``raw_bars`` 的标的列表，每只默认 row_count=500、interval="d"。
        local_end: ``load_bar_frame_any_range`` 返回的本地最晚 datetime（探测窗口右界）；
            None 时返回空 frame（载入范围探测得 (None, None)）。
        local_start: 本地最早 datetime；None 时默认取 ``local_end`` 前约 6 年，
            保证本地范围足够宽、能通过 Tier-2 数据充足性预检（这些用例聚焦
            无前视/草稿等性质，不测薄数据）。
    """

    def __init__(
        self,
        symbols: list[str],
        local_end: datetime | None = None,
        local_start: datetime | None = None,
    ) -> None:
        self._symbols = symbols
        self._local_end = local_end
        # 默认提供约 6 年的宽范围，确保默认规则（eval_window_days=900）下预检通过。
        if local_start is None and local_end is not None:
            local_start = local_end - timedelta(days=365 * 6)
        self._local_start = local_start

    def list_data_resources(self) -> dict[str, Any]:
        """返回合成数据资源摘要（仅 raw_bars 有内容）。"""
        return {
            "raw_bars": [
                {
                    "vt_symbol": s,
                    "interval": "d",
                    "row_count": 500,
                    "start": "2023-01-01",
                    "end": "2025-12-31",
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
        if self._local_end is None:
            return None
        dts = [d for d in (self._local_start, self._local_end) if d is not None]
        return pl.DataFrame({"datetime": dts, "close": [10.0] * len(dts)})


# ---------------------------------------------------------------------------
# 测试辅助：合成 SymbolProfile / 窗口 frame
# ---------------------------------------------------------------------------


def _make_profile(
    vt_symbol: str,
    *,
    available: bool = True,
    liq_level: str = "high",
    vol_level: str = "high",
    pred_level: str = "trending",
    confidence: str = "high",
    erb: datetime | None = _AS_OF,
) -> SymbolProfile:
    """构造一个 available 可控、四块齐全的合成 SymbolProfile。

    Args:
        vt_symbol: 标的代码。
        available: 数据是否可用；False 时 blocks 为空。
        liq_level/vol_level/pred_level: 三个有等级块的 level。
        confidence: 各指标置信度（同时作为 overall_confidence）。
        erb: input.effective_right_bound；用于审计右边界断言。

    Returns:
        合成 ``SymbolProfile``。
    """
    inp = ProfileInput(
        vt_symbol=vt_symbol,
        interval="d",
        as_of=_AS_OF,
        lookback_days=365,
        effective_bar_count=300 if available else 0,
        effective_right_bound=erb if available else None,
        rules_id="builtin-v1",
    )
    blocks: list[MetricBlock] = []
    if available:
        blocks = [
            MetricBlock(
                block="data_quality",
                metrics=[
                    MetricValue(
                        key="count_valid_bars",
                        value=300.0,
                        effective_sample=300,
                        confidence=confidence,
                    )
                ],
            ),
            MetricBlock(
                block="liquidity",
                metrics=[
                    MetricValue(
                        key="avg_turnover", value=1e7, effective_sample=300, confidence=confidence
                    )
                ],
                level=liq_level,
            ),
            MetricBlock(
                block="volatility",
                metrics=[
                    MetricValue(
                        key="realized_volatility",
                        value=0.02,
                        effective_sample=300,
                        confidence=confidence,
                    )
                ],
                level=vol_level,
            ),
            MetricBlock(
                block="predictability",
                metrics=[
                    MetricValue(
                        key="hurst_exponent",
                        value=0.6,
                        effective_sample=300,
                        confidence=confidence,
                    )
                ],
                level=pred_level,
            ),
        ]
    return SymbolProfile(
        input=inp,
        available=available,
        unavailable_reason=None if available else "本地无数据",
        blocks=blocks,
        overall_confidence=confidence if available else "insufficient",
    )


def _synthetic_window() -> pl.DataFrame:
    """构造一段确定性窗口 frame（trend + 波动），供 Tier-1 代理指标计算。"""
    import numpy as np

    n = 300
    rng = np.random.default_rng(42)
    # 带漂移与 ARCH 风味的合成价格，保证代理指标有非平凡值。
    rets = 0.001 + 0.01 * rng.standard_normal(n)
    close = 10.0 * np.exp(np.cumsum(rets))
    dts = [datetime(2024, 1, 1) for _ in range(n)]
    return pl.DataFrame({"datetime": dts, "close": close.tolist()})


def _make_request(**overrides: Any) -> CNNScreeningRequest:
    """构造默认 CNNScreeningRequest，可覆盖任意字段。"""
    params: dict[str, Any] = {
        "name": "test-run",
        "interval": "d",
        "as_of": _AS_OF,
        "lookback_days": 365,
        "min_bar_count": 250,
        "top_k": 3,
        "run_tier2": False,
    }
    params.update(overrides)
    return CNNScreeningRequest(**params)


# ---------------------------------------------------------------------------
# 桩工具：用预设 profile / WF 报告替换 Profiler / run_walk_forward_evaluate
# ---------------------------------------------------------------------------


def _patch_profiler(monkeypatch, profile_map: dict[str, SymbolProfile]) -> None:
    """monkeypatch runner 内的 Profiler，使其 profile() 按 vt_symbol 返回预设画像。

    Args:
        monkeypatch: pytest fixture。
        profile_map: ``{vt_symbol: SymbolProfile}``；未命中的标的抛 KeyError
            （用于模拟 Tier-1 单只异常时改用 _raising_profile_map）。
    """

    class _StubProfiler:
        def __init__(self, lab: Any) -> None:
            self.lab = lab

        def profile(self, *, vt_symbol: str, **_kw: Any) -> SymbolProfile:
            prof = profile_map[vt_symbol]
            if isinstance(prof, Exception):
                raise prof
            return prof

    monkeypatch.setattr(runner_mod, "Profiler", _StubProfiler)
    # 让代理指标取到确定性窗口（available 路径才会调用）。
    monkeypatch.setattr(
        runner_mod, "_load_window_frame", lambda *a, **k: _synthetic_window()
    )


def _patch_wf(monkeypatch, captured: list[CNNWalkForwardRequest], score: float = 1.0):
    """monkeypatch run_walk_forward_evaluate：捕获请求并返回合成正分 WF 报告。

    Args:
        monkeypatch: pytest fixture。
        captured: 收集所有传入的 CNNWalkForwardRequest（供 Property 2 断言）。
        score: 每折 candidate_score 值（>0 → edge_ok=True）。
    """

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
# Property 6: 入围选择正确
# Feature: cnn-stock-screening, Property 6: 入围选择正确
# ===========================================================================


def test_property6_promoted_is_topk_by_score_filtered(monkeypatch) -> None:
    # Feature: cnn-stock-screening, Property 6: 入围选择正确
    """入围集合 == 按分降序、available + 置信度过滤后的前 top_k。"""
    # 直接对 _select_promoted 喂合成 Tier1Score，隔离漏斗逻辑。
    runner = ScreeningRunner(FakeLab([]))
    scores = [
        Tier1Score(vt_symbol="A.SSE", fitness_score=0.9, contributions=[], overall_confidence="high", available=True),
        Tier1Score(vt_symbol="B.SSE", fitness_score=0.8, contributions=[], overall_confidence="medium", available=True),
        Tier1Score(vt_symbol="C.SSE", fitness_score=0.7, contributions=[], overall_confidence="low", available=True),
        Tier1Score(vt_symbol="D.SSE", fitness_score=0.95, contributions=[], overall_confidence="insufficient", available=True),  # 低置信
        Tier1Score(vt_symbol="E.SSE", fitness_score=None, contributions=[], overall_confidence="insufficient", available=False),  # 不可用
        Tier1Score(vt_symbol="F.SSE", fitness_score=0.6, contributions=[], overall_confidence="high", available=True),
    ]
    req = _make_request(top_k=3, min_confidence="low")
    promoted, reasons = runner._select_promoted(scores, req)
    promoted_syms = [s.vt_symbol for s in promoted]

    # 合格者（available 且 >= low）：A(0.9,high) B(0.8,med) C(0.7,low) F(0.6,high)；
    # D 低置信(insufficient<low) / E 不可用 → 不合格。top_k=3 → 取前三 A,B,C。
    assert promoted_syms == ["A.SSE", "B.SSE", "C.SSE"]
    # 不可用 / 低置信绝不入围
    assert "D.SSE" not in promoted_syms
    assert "E.SSE" not in promoted_syms
    # 未入围原因齐全
    assert "置信度不足" in reasons["D.SSE"]
    assert reasons["E.SSE"]  # 数据不可用原因
    assert "未进 top_k" in reasons["F.SSE"]


def test_property6_promoted_count_eq_min_topk_qualified(monkeypatch) -> None:
    # Feature: cnn-stock-screening, Property 6: 入围选择正确
    """入围数 == min(top_k, 合格数)：合格者少于 top_k 时只取合格者。"""
    runner = ScreeningRunner(FakeLab([]))
    scores = [
        Tier1Score(vt_symbol="A.SSE", fitness_score=0.9, contributions=[], overall_confidence="high", available=True),
        Tier1Score(vt_symbol="B.SSE", fitness_score=None, contributions=[], overall_confidence="insufficient", available=False),
    ]
    req = _make_request(top_k=5, min_confidence="low")
    promoted, _ = runner._select_promoted(scores, req)
    assert len(promoted) == 1  # min(5, 1 合格)


def test_property6_min_confidence_medium_filters_low(monkeypatch) -> None:
    # Feature: cnn-stock-screening, Property 6: 入围选择正确
    """min_confidence=medium 时 low 标的不入围（置信度排序生效）。"""
    runner = ScreeningRunner(FakeLab([]))
    scores = [
        Tier1Score(vt_symbol="A.SSE", fitness_score=0.9, contributions=[], overall_confidence="high", available=True),
        Tier1Score(vt_symbol="B.SSE", fitness_score=0.95, contributions=[], overall_confidence="low", available=True),
        Tier1Score(vt_symbol="C.SSE", fitness_score=0.8, contributions=[], overall_confidence="medium", available=True),
    ]
    req = _make_request(top_k=5, min_confidence="medium")
    promoted, _ = runner._select_promoted(scores, req)
    assert [s.vt_symbol for s in promoted] == ["A.SSE", "C.SSE"]


def test_property6_tier2_cap_truncates_and_logs(monkeypatch, caplog) -> None:
    # Feature: cnn-stock-screening, Property 6: 入围选择正确
    """入围数超过 tier2_cap 时按分截断并 log（不静默丢弃，Requirement 4.5）。"""
    rules = ScreeningRules(tier2_cap=2)
    runner = ScreeningRunner(FakeLab([]), rules=rules)
    scores = [
        Tier1Score(vt_symbol=f"{c}.SSE", fitness_score=v, contributions=[], overall_confidence="high", available=True)
        for c, v in [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)]
    ]
    req = _make_request(top_k=4, min_confidence="low")
    with caplog.at_level("WARNING"):
        promoted, reasons = runner._select_promoted(scores, req)
    assert [s.vt_symbol for s in promoted] == ["A.SSE", "B.SSE"]  # cap=2
    assert "C.SSE" in reasons and "上限" in reasons["C.SSE"]
    assert any("上限" in r.message for r in caplog.records)


# ===========================================================================
# Property 10: 批量鲁棒性
# Feature: cnn-stock-screening, Property 10: 批量鲁棒性
# ===========================================================================


def test_property10_empty_universe_returns_structured(monkeypatch) -> None:
    # Feature: cnn-stock-screening, Property 10: 批量鲁棒性
    """universe 过滤后为空 → 结构化结果（universe_size 0、空榜单），不抛异常。"""
    # FakeLab 无任何标的 → discover_universe 返回 []
    runner = ScreeningRunner(FakeLab([]))
    req = _make_request()
    result = runner.run(req)
    assert isinstance(result, ScreeningResult)
    assert result.universe_size == 0
    assert result.leaderboard == []
    assert result.status == "draft"
    assert result.eval_window is None


def test_property10_tier1_failure_degraded_continues(monkeypatch) -> None:
    # Feature: cnn-stock-screening, Property 10: 批量鲁棒性
    """某标的 Tier-1 画像抛异常 → 降级为失败行，其余标的照常打分。"""
    syms = ["000001.SZSE", "000002.SZSE", "000003.SZSE"]
    profile_map: dict[str, Any] = {
        "000001.SZSE": _make_profile("000001.SZSE"),
        "000002.SZSE": RuntimeError("boom"),  # 触发异常
        "000003.SZSE": _make_profile("000003.SZSE"),
    }
    _patch_profiler(monkeypatch, profile_map)
    runner = ScreeningRunner(FakeLab(syms))
    result = runner.run(_make_request())

    assert result.universe_size == 3
    assert len(result.leaderboard) == 3
    # 失败标的降级为 available=False，note 含"打分失败"
    failed = next(r for r in result.leaderboard if r.tier1.vt_symbol == "000002.SZSE")
    assert failed.tier1.available is False
    assert "打分失败" in (failed.tier1.note or "")
    # 其余标的正常出分
    ok = [r for r in result.leaderboard if r.tier1.available]
    assert len(ok) == 2
    assert all(r.tier1.fitness_score is not None for r in ok)


def test_property10_tier2_failure_degraded_continues(monkeypatch) -> None:
    # Feature: cnn-stock-screening, Property 10: 批量鲁棒性
    """某标的 Tier-2 WF 抛异常 → Tier2Verdict(evaluable=False)，整批继续返回。"""
    syms = ["000001.SZSE", "000002.SZSE"]
    _patch_profiler(
        monkeypatch,
        {s: _make_profile(s) for s in syms},
    )

    def _wf(req, on_progress=None, store=None):  # noqa: ANN001
        if req.target_symbol == "000002.SZSE":
            raise RuntimeError("wf boom")
        return {
            "report_id": "wf_ok",
            "request": {"target_symbol": req.target_symbol},
            "folds": [{"candidate_score": 1.0}],
            "summary": {},
        }

    monkeypatch.setattr(runner_mod, "run_walk_forward_evaluate", _wf)
    runner = ScreeningRunner(FakeLab(syms, local_end=_AS_OF))
    result = runner.run(_make_request(run_tier2=True, top_k=2))

    verdicts = {r.tier1.vt_symbol: r.tier2 for r in result.leaderboard if r.tier2 is not None}
    assert verdicts["000001.SZSE"].evaluable is True
    assert verdicts["000002.SZSE"].evaluable is False
    assert "Tier-2 失败" in (verdicts["000002.SZSE"].note or "")


# ===========================================================================
# Property 2: 无前视（WF 请求 end <= as_of）
# Feature: cnn-stock-screening, Property 2: 无前视
# ===========================================================================


def test_property2_wf_end_not_after_as_of(monkeypatch) -> None:
    # Feature: cnn-stock-screening, Property 2: 无前视
    """所有构造的 CNNWalkForwardRequest.end <= req.as_of（按日界）。"""
    syms = ["000001.SZSE", "000002.SZSE"]
    _patch_profiler(monkeypatch, {s: _make_profile(s) for s in syms})
    captured: list[CNNWalkForwardRequest] = []
    _patch_wf(monkeypatch, captured)

    # 本地最新晚于 as_of：end 应被夹到 as_of，绝不越界。
    lab = FakeLab(syms, local_end=datetime(2025, 12, 31))
    runner = ScreeningRunner(lab)
    runner.run(_make_request(run_tier2=True, top_k=2))

    assert len(captured) == 2
    for wf_req in captured:
        assert wf_req.end <= _AS_OF.date()
        assert wf_req.start < wf_req.end


def test_property2_wf_end_clamped_to_local_when_earlier(monkeypatch) -> None:
    # Feature: cnn-stock-screening, Property 2: 无前视
    """本地最新早于 as_of 时 end 收窄到本地范围（R9.5），仍 <= as_of。"""
    syms = ["000001.SZSE"]
    _patch_profiler(monkeypatch, {s: _make_profile(s) for s in syms})
    captured: list[CNNWalkForwardRequest] = []
    _patch_wf(monkeypatch, captured)

    local_end = datetime(2025, 3, 15)  # 早于 as_of 2025-06-01
    runner = ScreeningRunner(FakeLab(syms, local_end=local_end))
    runner.run(_make_request(run_tier2=True, top_k=1))

    assert captured[0].end == local_end.date()
    assert captured[0].end <= _AS_OF.date()


# ===========================================================================
# run_tier2=False → 仅 Tier-1，无 WF 调用
# ===========================================================================


def test_run_tier2_false_no_wf_calls(monkeypatch) -> None:
    """run_tier2=False 时只产 Tier-1 榜单，不调用 run_walk_forward_evaluate。"""
    syms = ["000001.SZSE", "000002.SZSE"]
    _patch_profiler(monkeypatch, {s: _make_profile(s) for s in syms})
    called: list[Any] = []
    monkeypatch.setattr(
        runner_mod,
        "run_walk_forward_evaluate",
        lambda *a, **k: called.append(1),
    )
    runner = ScreeningRunner(FakeLab(syms))
    result = runner.run(_make_request(run_tier2=False, top_k=2))

    assert called == []  # 无 WF 调用
    assert result.eval_window is None
    assert all(r.tier2 is None for r in result.leaderboard)
    # promoted_to_tier2 仍可标记入围资格，但未跑 Tier-2 → tier2 为 None
    assert len(result.leaderboard) == 2


# ===========================================================================
# 端到端（含桩 Tier-2）：排序 / eval_window / status draft
# ===========================================================================


def test_end_to_end_with_stubs(monkeypatch) -> None:
    """端到端：榜单按分降序、eval_window 填充、status draft、右边界审计。"""
    syms = ["000001.SZSE", "000002.SZSE", "000003.SZSE"]
    # 制造分数差异：000003 流动性低 → 分较低；000001/000002 高分。
    profile_map = {
        "000001.SZSE": _make_profile("000001.SZSE", liq_level="high", vol_level="high"),
        "000002.SZSE": _make_profile("000002.SZSE", liq_level="high", vol_level="medium"),
        "000003.SZSE": _make_profile("000003.SZSE", liq_level="low", vol_level="low"),
    }
    _patch_profiler(monkeypatch, profile_map)
    captured: list[CNNWalkForwardRequest] = []
    _patch_wf(monkeypatch, captured)

    runner = ScreeningRunner(FakeLab(syms, local_end=_AS_OF))
    progress: list[float] = []
    result = runner.run(
        _make_request(run_tier2=True, top_k=2, min_confidence="low"),
        on_progress=lambda p, m: progress.append(p),
    )

    # 草稿恒定
    assert result.status == "draft"
    # 榜单按 fitness_score 降序、rank 连续
    ranks = [r.rank for r in result.leaderboard]
    assert ranks == [1, 2, 3]
    fs = [r.tier1.fitness_score for r in result.leaderboard]
    assert fs[0] >= fs[1] >= fs[2]
    # top_k=2 → 前两名入围 Tier-2
    promoted = [r for r in result.leaderboard if r.promoted_to_tier2]
    assert len(promoted) == 2
    assert all(r.tier2 is not None and r.tier2.evaluable for r in promoted)
    assert all(r.tier2.edge_ok for r in promoted)  # 正分 → edge_ok
    # eval_window 填充且 end <= as_of
    assert result.eval_window is not None
    assert result.eval_window["objective"] == "classification"
    assert result.eval_window["end"] <= _AS_OF.date().isoformat()
    # 右边界审计：取 Tier-1 profiles 的最大 effective_right_bound
    assert result.effective_right_bound == _AS_OF
    # rules_id 回显
    assert result.rules_id == DEFAULT_SCREENING_RULES.rules_id
    assert result.input["rules_id"] == DEFAULT_SCREENING_RULES.rules_id
    # 进度推进到 100
    assert progress and progress[-1] == 100


# ===========================================================================
# Property 2/5 guard: 默认规则下 Tier-2 窗口可生成 ≥2 折（真实窗口生成，不桩化）
# Feature: cnn-stock-screening, Property 2/5 guard: 默认规则下 Tier-2 窗口可生成 ≥2 折
# ===========================================================================


def test_default_rules_wf_windows_at_least_two_folds() -> None:
    # Feature: cnn-stock-screening, Property 2/5 guard: 默认规则下 Tier-2 窗口可生成 ≥2 折
    """默认 ScreeningRules 构造出的 WF 请求参数送入真实 walk_forward_windows 可生成 ≥2 折。

    本测试是对"eval_window_days 过小导致 0 折 → 所有 Tier-2 评估失败"缺陷的回归防守：
    直接从 DEFAULT_SCREENING_RULES 读取 train_days / fold_test_days / step_days，
    并用 _build_wf_request 产生的 start/end 驱动真实 walk_forward_windows，
    断言至少生成 2 个窗口（Property 2/5 guard）。

    注意：本测试**不**桩化 walk_forward_windows，这是回归价值所在——
    只要 rules 参数不满足 eval_window_days >= train_days + test_days，此测试即失败。

    Args:
        无（纯逻辑，无外部 I/O）。
    """
    from datetime import date, timedelta

    from aitrade.backtest.validation import walk_forward_windows
    from aitrade.screening.rules import DEFAULT_SCREENING_RULES

    rules = DEFAULT_SCREENING_RULES
    as_of = date(2025, 6, 1)

    # 复现 _build_wf_request 的窗口计算逻辑（eval_start 未显式提供时的路径）
    end = as_of
    start = end - timedelta(days=rules.eval_window_days)

    windows = walk_forward_windows(
        start=start,
        end=end,
        train_days=rules.train_days,
        test_days=rules.fold_test_days,
        step_days=rules.step_days,
    )

    assert len(windows) >= 2, (
        f"默认规则下 walk_forward_windows 仅生成 {len(windows)} 折（期望 >= 2）。"
        f" eval_window_days={rules.eval_window_days}, train_days={rules.train_days},"
        f" fold_test_days={rules.fold_test_days}, step_days={rules.step_days}。"
        f" 需确保 eval_window_days >= train_days + fold_test_days 且有足够空间生成多折。"
    )


def test_persist_writes_only_when_requested(monkeypatch, tmp_path) -> None:
    """persist=True 时写 SCREENING_PATH；persist=False 时不写。"""
    from aitrade.screening.store import ScreeningStore

    syms = ["000001.SZSE"]
    _patch_profiler(monkeypatch, {s: _make_profile(s) for s in syms})

    store = ScreeningStore(base_path=tmp_path)
    runner = ScreeningRunner(FakeLab(syms), store=store)

    runner.run(_make_request(run_tier2=False, persist=False))
    assert store.list_ids() == []  # 未持久化

    result = runner.run(_make_request(run_tier2=False, persist=True))
    assert result.run_id in store.list_ids() or len(store.list_ids()) == 1
    loaded = store.load(store.list_ids()[0])
    assert loaded.status == "draft"
