"""
CNN 选股 Tier-2 绝对 edge 门禁（edge.py）的属性测试与示例测试。

覆盖：
- Property 7: edge_ok 当且仅当 mean>0 且 pos_ratio>=阈值，不依赖相对晋级门禁
- 独立于 summary.passed / summary.avg_score_delta
- 空折 → evaluable=False, edge_ok=False
- 缺失 report_id / request → 不崩溃，字段有默认值
- 示例测试：全正折、全负折、混合临界值

Feature: cnn-stock-screening, Property 7: 绝对 edge 门禁正确且不依赖相对晋级
"""

from __future__ import annotations

from statistics import mean
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.screening.edge import derive_edge
from aitrade.screening.rules import DEFAULT_SCREENING_RULES, ScreeningRules


# ---------------------------------------------------------------------------
# 辅助：合成 WF 报告
# ---------------------------------------------------------------------------

def _make_report(
    scores: list[float],
    *,
    vt_symbol: str = "000001.SZSE",
    report_id: str | None = "wf_test_001",
    summary_passed: bool = False,
    summary_avg_score_delta: float | None = None,
    avg_cross_seed_std: float | None = None,
    include_request: bool = True,
    include_cross_seed_in_folds: bool = False,
) -> dict[str, Any]:
    """构造最小合法 WF 报告字典，供测试使用。

    Args:
        scores: 各折 candidate_score 列表（跨种子均值）。
        vt_symbol: 目标标的代码，写入 request.target_symbol。
        report_id: 报告 ID；None 表示缺失该键。
        summary_passed: summary.passed 值（相对门禁，应被忽略）。
        summary_avg_score_delta: summary.avg_score_delta 值（相对门禁，应被忽略）。
        avg_cross_seed_std: summary.avg_cross_seed_std；None 时不写入 summary。
        include_request: 是否在报告中包含 request 字段。
        include_cross_seed_in_folds: 是否在每折加入 cross_seed.std 字段。

    Returns:
        模拟 run_walk_forward_evaluate 输出的字典。
    """
    folds: list[dict[str, Any]] = []
    for idx, score in enumerate(scores, start=1):
        fold: dict[str, Any] = {"fold": idx, "candidate_score": score}
        if include_cross_seed_in_folds:
            fold["cross_seed"] = {"std": abs(score) * 0.1, "mean": score, "n": 1}
        folds.append(fold)

    summary: dict[str, Any] = {
        "passed": summary_passed,
        "avg_score_delta": summary_avg_score_delta,
    }
    if avg_cross_seed_std is not None:
        summary["avg_cross_seed_std"] = avg_cross_seed_std

    report: dict[str, Any] = {"folds": folds, "summary": summary}
    if report_id is not None:
        report["report_id"] = report_id
    if include_request:
        report["request"] = {"target_symbol": vt_symbol}
    return report


def _expected_edge_ok(scores: list[float], threshold: float) -> bool:
    """独立计算绝对 edge 判据（测试参照实现），与 derive_edge 完全解耦。

    Args:
        scores: 各折 candidate_score 列表（已过滤掉 None）。
        threshold: min_positive_fold_ratio 阈值。

    Returns:
        edge_ok 的期望值。
    """
    if not scores:
        return False
    avg = mean(scores)
    pos_ratio = sum(1 for s in scores if s > 0) / len(scores)
    return (avg > 0) and (pos_ratio >= threshold)


# ---------------------------------------------------------------------------
# Property 7: edge_ok 当且仅当 mean>0 且 pos_ratio>=阈值
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    scores=st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    ),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_property7_edge_ok_iff_mean_positive_and_pos_ratio_ge_threshold(
    scores: list[float], threshold: float
) -> None:
    """Property 7: edge_ok ⟺ mean>0 且 pos_ratio>=阈值，与参照实现完全一致。

    # Feature: cnn-stock-screening, Property 7: 绝对 edge 门禁正确且不依赖相对晋级
    """
    rules = ScreeningRules(min_positive_fold_ratio=threshold)
    report = _make_report(scores)
    verdict = derive_edge(report, rules)

    expected = _expected_edge_ok(scores, threshold)
    assert verdict.edge_ok is expected, (
        f"scores={scores}, threshold={threshold}, "
        f"got edge_ok={verdict.edge_ok}, expected={expected}"
    )
    assert verdict.evaluable is True
    assert verdict.avg_score is not None
    assert verdict.pos_fold_ratio is not None


# ---------------------------------------------------------------------------
# Property 7: 独立于 summary.passed 和 summary.avg_score_delta
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    scores=st.lists(
        st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    ),
    summary_passed=st.booleans(),
    summary_delta=st.one_of(
        st.none(),
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    ),
)
def test_property7_independence_from_relative_gate(
    scores: list[float], summary_passed: bool, summary_delta: float | None
) -> None:
    """summary.passed 与 summary.avg_score_delta 任意变化时，edge_ok 不受影响。

    # Feature: cnn-stock-screening, Property 7: 绝对 edge 门禁正确且不依赖相对晋级
    """
    rules = DEFAULT_SCREENING_RULES
    threshold = rules.min_positive_fold_ratio

    # 两个报告：summary.passed 和 delta 不同，其余完全一致
    report_a = _make_report(
        scores, summary_passed=False, summary_avg_score_delta=None
    )
    report_b = _make_report(
        scores, summary_passed=summary_passed, summary_avg_score_delta=summary_delta
    )

    verdict_a = derive_edge(report_a, rules)
    verdict_b = derive_edge(report_b, rules)

    assert verdict_a.edge_ok is verdict_b.edge_ok, (
        f"edge_ok 不应随 summary.passed/avg_score_delta 变化："
        f"scores={scores}, passed={summary_passed}, delta={summary_delta}"
    )
    # 两个判断都与参照实现一致
    expected = _expected_edge_ok(scores, threshold)
    assert verdict_a.edge_ok is expected
    assert verdict_b.edge_ok is expected


# ---------------------------------------------------------------------------
# 空折 → evaluable=False, edge_ok=False
# ---------------------------------------------------------------------------

def test_empty_folds_returns_not_evaluable() -> None:
    """folds 为空列表时，evaluable=False 且 edge_ok=False。

    # Feature: cnn-stock-screening, Property 7: 绝对 edge 门禁正确且不依赖相对晋级
    """
    report: dict[str, Any] = {
        "report_id": "wf_empty",
        "request": {"target_symbol": "600000.SSE"},
        "folds": [],
        "summary": {"passed": True, "avg_score_delta": 99.9},
    }
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.evaluable is False
    assert verdict.edge_ok is False
    assert verdict.avg_score is None
    assert verdict.pos_fold_ratio is None
    assert verdict.note is not None
    assert verdict.vt_symbol == "600000.SSE"
    assert verdict.report_id == "wf_empty"


def test_folds_key_missing_treated_as_empty() -> None:
    """report 中缺少 folds 键时，等同于空折，evaluable=False。"""
    report: dict[str, Any] = {
        "report_id": "wf_no_folds",
        "request": {"target_symbol": "600030.SSE"},
        "summary": {"passed": True},
    }
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.evaluable is False
    assert verdict.edge_ok is False


def test_folds_with_all_none_scores_treated_as_empty() -> None:
    """所有折的 candidate_score 均为 None 时，等同于空折，evaluable=False。"""
    report: dict[str, Any] = {
        "report_id": "wf_none_scores",
        "request": {"target_symbol": "000001.SZSE"},
        "folds": [
            {"fold": 1, "candidate_score": None},
            {"fold": 2, "candidate_score": None},
        ],
        "summary": {},
    }
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.evaluable is False
    assert verdict.edge_ok is False


# ---------------------------------------------------------------------------
# 缺失 report_id / request → 不崩溃，字段默认合理
# ---------------------------------------------------------------------------

def test_missing_report_id_no_crash() -> None:
    """report_id 缺失时，verdict.report_id 为 None，不崩溃。"""
    report: dict[str, Any] = {
        "request": {"target_symbol": "000002.SZSE"},
        "folds": [{"fold": 1, "candidate_score": 1.0}],
        "summary": {},
    }
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.report_id is None
    assert verdict.evaluable is True


def test_missing_request_no_crash() -> None:
    """request 字段缺失时，vt_symbol 为空字符串，不崩溃。"""
    report: dict[str, Any] = {
        "report_id": "wf_no_req",
        "folds": [{"fold": 1, "candidate_score": 2.0}],
        "summary": {},
    }
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.vt_symbol == ""
    assert verdict.evaluable is True


def test_missing_target_symbol_in_request_no_crash() -> None:
    """request 存在但缺少 target_symbol 键时，vt_symbol 为空字符串，不崩溃。"""
    report: dict[str, Any] = {
        "report_id": "wf_no_symbol",
        "request": {"objective": "classification"},
        "folds": [{"fold": 1, "candidate_score": 3.0}],
        "summary": {},
    }
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.vt_symbol == ""
    assert verdict.evaluable is True


# ---------------------------------------------------------------------------
# 示例测试：全正折、全负折、混合临界值
# ---------------------------------------------------------------------------

def test_all_positive_folds_edge_ok_true() -> None:
    """所有折 candidate_score > 0 → edge_ok=True（默认阈值 0.5）。"""
    scores = [1.5, 0.3, 2.0, 0.1]
    report = _make_report(scores)
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.evaluable is True
    assert verdict.edge_ok is True
    assert verdict.avg_score is not None and verdict.avg_score > 0
    assert verdict.pos_fold_ratio == 1.0


def test_all_negative_folds_edge_ok_false() -> None:
    """所有折 candidate_score < 0 → edge_ok=False。"""
    scores = [-1.5, -0.3, -2.0, -0.1]
    report = _make_report(scores)
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.evaluable is True
    assert verdict.edge_ok is False
    assert verdict.avg_score is not None and verdict.avg_score < 0
    assert verdict.pos_fold_ratio == 0.0


def test_mixed_positive_majority_above_threshold_edge_ok_true() -> None:
    """正分折超过阈值且均值>0 → edge_ok=True。

    4 折中 3 正（pos_ratio=0.75 >= 0.5），均值 = (2+1+3-1)/4 = 1.25 > 0。
    """
    scores = [2.0, 1.0, 3.0, -1.0]
    report = _make_report(scores)
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.evaluable is True
    assert verdict.edge_ok is True
    assert abs(verdict.pos_fold_ratio - 0.75) < 1e-9  # type: ignore[arg-type]


def test_boundary_pos_ratio_exactly_at_threshold_edge_ok_true() -> None:
    """正分折比例恰好等于阈值（0.5）且均值>0 → edge_ok=True（≥ 阈值，应通过）。"""
    scores = [1.0, -0.5]  # 均值=0.25 > 0, pos_ratio=0.5 >= 0.5
    rules = ScreeningRules(min_positive_fold_ratio=0.5)
    report = _make_report(scores)
    verdict = derive_edge(report, rules)

    assert verdict.evaluable is True
    assert verdict.edge_ok is True


def test_boundary_pos_ratio_just_below_threshold_edge_ok_false() -> None:
    """正分折比例恰好低于阈值 → edge_ok=False（即使均值 > 0）。

    3 折中 2 正（pos_ratio ≈ 0.667），但阈值为 0.8 → 不通过。
    均值 = (1 + 0.5 - 0.1) / 3 = 0.467 > 0，但 pos_ratio < 阈值。
    """
    scores = [1.0, 0.5, -0.1]  # pos_ratio ≈ 0.667 < 0.8
    rules = ScreeningRules(min_positive_fold_ratio=0.8)
    report = _make_report(scores)
    verdict = derive_edge(report, rules)

    assert verdict.evaluable is True
    assert verdict.edge_ok is False


def test_positive_pos_ratio_but_negative_avg_edge_ok_false() -> None:
    """正分折占比满足阈值，但均值 < 0 → edge_ok=False（双条件均须满足）。

    4 折中 3 正（pos_ratio=0.75 >= 0.5），但有一折极差的负分拉低均值 < 0。
    均值 = (0.1 + 0.1 + 0.1 - 100) / 4 = -24.925 < 0。
    """
    scores = [0.1, 0.1, 0.1, -100.0]
    report = _make_report(scores)
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.evaluable is True
    assert verdict.edge_ok is False


def test_single_fold_positive_edge_ok_true() -> None:
    """单折且 candidate_score > 0，默认阈值 0.5 → edge_ok=True（pos_ratio=1.0 >= 0.5）。"""
    report = _make_report([5.0])
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.evaluable is True
    assert verdict.edge_ok is True
    assert verdict.pos_fold_ratio == 1.0


def test_single_fold_negative_edge_ok_false() -> None:
    """单折且 candidate_score < 0 → edge_ok=False。"""
    report = _make_report([-3.0])
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.evaluable is True
    assert verdict.edge_ok is False


# ---------------------------------------------------------------------------
# avg_cross_seed_std 来源优先级
# ---------------------------------------------------------------------------

def test_avg_cross_seed_std_from_summary() -> None:
    """summary 中存在 avg_cross_seed_std 时，优先使用 summary 的值。"""
    report = _make_report(
        [1.0, 2.0],
        avg_cross_seed_std=0.99,
        include_cross_seed_in_folds=True,
    )
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.avg_cross_seed_std is not None
    assert abs(verdict.avg_cross_seed_std - 0.99) < 1e-9  # type: ignore[arg-type]


def test_avg_cross_seed_std_fallback_from_folds() -> None:
    """summary 无 avg_cross_seed_std 时，从各折 cross_seed.std 均值推算。

    两折 std 分别为 0.1 和 0.3，期望均值 0.2。
    """
    report: dict[str, Any] = {
        "report_id": "wf_std_folds",
        "request": {"target_symbol": "000001.SZSE"},
        "folds": [
            {"fold": 1, "candidate_score": 1.0, "cross_seed": {"std": 0.1, "mean": 1.0, "n": 3}},
            {"fold": 2, "candidate_score": 2.0, "cross_seed": {"std": 0.3, "mean": 2.0, "n": 3}},
        ],
        "summary": {"passed": False},  # 无 avg_cross_seed_std
    }
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.avg_cross_seed_std is not None
    assert abs(verdict.avg_cross_seed_std - 0.2) < 1e-9  # type: ignore[arg-type]


def test_avg_cross_seed_std_none_when_no_source() -> None:
    """summary 与各折均无跨种子 std 信息时，avg_cross_seed_std 为 None。"""
    report = _make_report([1.0, 2.0], avg_cross_seed_std=None, include_cross_seed_in_folds=False)
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.avg_cross_seed_std is None


# ---------------------------------------------------------------------------
# report_id 透传
# ---------------------------------------------------------------------------

def test_report_id_propagated() -> None:
    """report_id 从报告字典透传到 Tier2Verdict.report_id。"""
    report = _make_report([1.0], report_id="wf_20250615_abc123")
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.report_id == "wf_20250615_abc123"


def test_vt_symbol_propagated() -> None:
    """target_symbol 从 request 透传到 Tier2Verdict.vt_symbol。"""
    report = _make_report([1.0], vt_symbol="600519.SSE")
    verdict = derive_edge(report, DEFAULT_SCREENING_RULES)

    assert verdict.vt_symbol == "600519.SSE"
