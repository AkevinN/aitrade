"""
CNN 适配度综合分（scoring.py）的属性测试与示例测试。

覆盖：
- Property 3: fitness_score 有界且贡献自洽（_weighted_average 直接测）
- Property 4: 综合分关于单维单调（_weighted_average 直接测）
- 排除维度（None value / insufficient confidence）不计入有效权重
- available=False 的 profile → fitness_score None
- 强 profile 比弱 profile 分高的示例测试

Feature: cnn-stock-screening, Property 3: CNN_Fitness_Score 有界且贡献自洽
Feature: cnn-stock-screening, Property 4: 综合分关于单维单调
"""

from __future__ import annotations

from datetime import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.profiling.types import MetricBlock, MetricValue, ProfileInput, SymbolProfile
from aitrade.screening.rules import DEFAULT_SCREENING_RULES, ScreeningRules
from aitrade.screening.scoring import (
    LIQUIDITY_LEVEL_VALUE,
    PREDICTABILITY_LEVEL_VALUE,
    VOLATILITY_LEVEL_VALUE,
    _weighted_average,
    compute_fitness_score,
)

# ---------------------------------------------------------------------------
# 测试辅助：合成 SymbolProfile / MetricValue
# ---------------------------------------------------------------------------

_AS_OF = datetime(2025, 1, 1)
_VT_SYMBOL = "000001.SZSE"


def _make_profile_input(interval: str = "d") -> ProfileInput:
    """构造最小 ProfileInput。"""
    return ProfileInput(
        vt_symbol=_VT_SYMBOL,
        interval=interval,
        as_of=_AS_OF,
        lookback_days=365,
        effective_bar_count=300,
        rules_id="builtin-v1",
    )


def _make_metric(key: str, value: float | None, sample: int, confidence: str) -> MetricValue:
    """构造单个 MetricValue 辅助。

    Args:
        key: 指标键。
        value: 指标值；None 表示 insufficient。
        sample: 有效样本量。
        confidence: 置信度，如 ``"high"``。

    Returns:
        构造好的 ``MetricValue``。
    """
    return MetricValue(key=key, value=value, effective_sample=sample, confidence=confidence)


def _make_block(
    block_name: str,
    metrics: list[MetricValue],
    level: str | None = None,
) -> MetricBlock:
    """构造 MetricBlock 辅助。

    Args:
        block_name: 块名，如 ``"liquidity"``。
        metrics: 该块内的指标列表。
        level: 块等级判定，如 ``"high"``。

    Returns:
        构造好的 ``MetricBlock``。
    """
    return MetricBlock(block=block_name, metrics=metrics, level=level)


def _make_full_profile(
    *,
    dq_bars: float = 300.0,
    dq_conf: str = "high",
    liq_level: str = "high",
    liq_conf: str = "high",
    vol_level: str = "medium",
    vol_conf: str = "high",
    pred_level: str = "trending",
    pred_conf: str = "medium",
    interval: str = "d",
    available: bool = True,
    unavailable_reason: str | None = None,
) -> SymbolProfile:
    """构造包含四个完整画像块的 SymbolProfile。

    Args:
        dq_bars: data_quality 块中 count_valid_bars 的值。
        dq_conf: data_quality 块置信度。
        liq_level: liquidity 块等级，如 ``"high"``。
        liq_conf: liquidity 块置信度。
        vol_level: volatility 块等级，如 ``"medium"``。
        vol_conf: volatility 块置信度。
        pred_level: predictability 块等级，如 ``"trending"``。
        pred_conf: predictability 块置信度。
        interval: 数据周期，影响 min_train_samples 基准值。
        available: 是否数据可用。
        unavailable_reason: 不可用原因。

    Returns:
        合成的 ``SymbolProfile``。
    """
    blocks = []
    if available:
        blocks = [
            _make_block(
                "data_quality",
                [_make_metric("count_valid_bars", dq_bars, int(dq_bars), dq_conf)],
            ),
            _make_block(
                "liquidity",
                [_make_metric("avg_turnover", 1e7, 200, liq_conf)],
                level=liq_level,
            ),
            _make_block(
                "volatility",
                [_make_metric("realized_volatility", 0.02, 200, vol_conf)],
                level=vol_level,
            ),
            _make_block(
                "predictability",
                [_make_metric("hurst_exponent", 0.6, 300, pred_conf)],
                level=pred_level,
            ),
        ]
    return SymbolProfile(
        input=_make_profile_input(interval=interval),
        available=available,
        unavailable_reason=unavailable_reason,
        blocks=blocks,
        overall_confidence="high" if available else "insufficient",
    )


def _make_proxies(
    *,
    nonlinearity: float | None = 0.7,
    nl_conf: str = "medium",
    pattern: float | None = 0.6,
    pat_conf: str = "medium",
    temporal: float | None = 0.8,
    temp_conf: str = "medium",
) -> dict[str, MetricValue]:
    """构造 CNN 代理指标字典辅助。

    Args:
        nonlinearity: 非线性结构值 ~[0,1]；None 表示不可用。
        nl_conf: 非线性置信度。
        pattern: 形态复现性值 ~[0,1]；None 表示不可用。
        pat_conf: 形态复现置信度。
        temporal: 时间稳定性值 ~[0,1]；None 表示不可用。
        temp_conf: 时间稳定性置信度。

    Returns:
        ``{dim: MetricValue}`` 字典，供 ``compute_fitness_score`` 消费。
    """
    return {
        "nonlinearity": _make_metric("nonlinearity", nonlinearity, 200, nl_conf),
        "pattern_recurrence": _make_metric("pattern_recurrence", pattern, 200, pat_conf),
        "temporal_stability": _make_metric("temporal_stability", temporal, 200, temp_conf),
    }


# ---------------------------------------------------------------------------
# Property 3: _weighted_average 有界且贡献自洽
# Feature: cnn-stock-screening, Property 3: CNN_Fitness_Score 有界且贡献自洽
# ---------------------------------------------------------------------------

# Hypothesis 策略：生成维度名（至多 8 个字符的字母字符串避免重复）
_dim_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Ll",)),
    min_size=1, max_size=8,
)


@settings(max_examples=100)
@given(
    items=st.lists(
        st.tuples(
            _dim_strategy,                             # 维度名
            st.floats(min_value=0.0, max_value=1.0),  # value ∈ [0,1]
            st.floats(min_value=0.0, max_value=10.0), # weight >= 0
        ),
        min_size=1,
        max_size=10,
        unique_by=lambda t: t[0],  # 维度名唯一
    )
)
def test_property3_weighted_average_bounded_and_self_consistent(
    items: list[tuple[str, float, float]],
) -> None:
    """Property 3: _weighted_average 结果有界 ∈ [0,1] 且 Σcontributions == score。

    Feature: cnn-stock-screening, Property 3: CNN_Fitness_Score 有界且贡献自洽

    对任意 value ∈ [0,1] 与 weight >= 0 的维度集合（不全为零权重），
    _weighted_average 的 score ∈ [0,1] 且贡献之和等于 score。
    """
    # 确保至少一个维度有正权重（all-zero 权重会返回 score=0，贡献为空）
    has_positive_weight = any(w > 0 for _, _, w in items)
    if not has_positive_weight:
        items = [(items[0][0], items[0][1], 1.0)] + list(items[1:])

    normalized = {d: v for d, v, _ in items}
    weights = {d: w for d, _, w in items}

    result = _weighted_average(normalized, weights)

    # 有界性
    assert 0.0 <= result.score <= 1.0 + 1e-9, f"score={result.score} 超出 [0,1]"

    # 贡献自洽：Σcontributions == score（浮点 tol）
    total_contrib = sum(result.renorm_contributions.values())
    assert abs(total_contrib - result.score) < 1e-9, (
        f"Σcontributions={total_contrib} != score={result.score}"
    )


# ---------------------------------------------------------------------------
# Property 4: _weighted_average 关于单维单调
# Feature: cnn-stock-screening, Property 4: 综合分关于单维单调
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    items=st.lists(
        st.tuples(
            _dim_strategy,
            st.floats(min_value=0.0, max_value=1.0),
            st.floats(min_value=0.0, max_value=10.0),
        ),
        min_size=2,
        max_size=10,
        unique_by=lambda t: t[0],
    ),
    idx=st.integers(min_value=0, max_value=9),
    delta=st.floats(min_value=0.0, max_value=1.0),
)
def test_property4_weighted_average_monotonic(
    items: list[tuple[str, float, float]],
    idx: int,
    delta: float,
) -> None:
    """Property 4: 提升任一维度的归一化值，_weighted_average score 不降。

    Feature: cnn-stock-screening, Property 4: 综合分关于单维单调

    固定其余维度，把选定维度的 value 增加 delta（截断到 1.0），
    score 不应下降。
    """
    # 确保有正权重
    if all(w == 0 for _, _, w in items):
        items = [(items[0][0], items[0][1], 1.0)] + list(items[1:])

    idx = idx % len(items)
    dim_name, val_orig, w = items[idx]

    normalized = {d: v for d, v, _ in items}
    weights = {d: w_ for d, _, w_ in items}

    score_before = _weighted_average(normalized, weights).score

    # 提升选定维度的 value
    new_val = min(1.0, val_orig + delta)
    normalized_bumped = {**normalized, dim_name: new_val}
    score_after = _weighted_average(normalized_bumped, weights).score

    assert score_after >= score_before - 1e-12, (
        f"score 下降了：before={score_before}, after={score_after}, "
        f"delta={delta}, dim={dim_name}, w={w}"
    )


# ---------------------------------------------------------------------------
# 排除维度：insufficient confidence → 不计入有效权重
# ---------------------------------------------------------------------------

def test_insufficient_dim_excluded_from_weights() -> None:
    """insufficient 维度（value=None 或 confidence=insufficient）应被排除，
    其 ScoreContribution.weight=0 且 contribution=0。

    验证：排除维度后，fitness_score 只取有效维度计算，权重归一正确。
    """
    # 所有代理指标置为 insufficient
    proxies_all_insufficient = {
        "nonlinearity": _make_metric("nonlinearity", None, 5, "insufficient"),
        "pattern_recurrence": _make_metric("pattern_recurrence", None, 5, "insufficient"),
        "temporal_stability": _make_metric("temporal_stability", None, 5, "insufficient"),
    }
    profile = _make_full_profile(
        dq_conf="high",
        liq_conf="high",
        vol_conf="high",
        pred_conf="high",
    )
    result = compute_fitness_score(profile, proxies_all_insufficient)

    # 代理维度的 contribution 应全为 0
    for c in result.contributions:
        if c.dimension in ("nonlinearity", "pattern_recurrence", "temporal_stability"):
            assert c.weight == 0.0, f"{c.dimension} weight 应为 0"
            assert c.contribution == 0.0, f"{c.dimension} contribution 应为 0"

    # fitness_score 由剩余四个画像块决定，应为有效值
    assert result.fitness_score is not None
    assert 0.0 <= result.fitness_score <= 1.0

    # 有效维度的 Σcontribution == fitness_score
    included_sum = sum(c.contribution for c in result.contributions if c.weight > 0)
    assert abs(included_sum - result.fitness_score) < 1e-9


def test_all_dims_insufficient_gives_none_score() -> None:
    """全部维度均为 insufficient 时，fitness_score 应为 None。"""
    # 构造一个 available=True 但所有块均无可用指标的 profile
    profile = SymbolProfile(
        input=_make_profile_input(),
        available=True,
        blocks=[
            # 无任何 metrics → _block_representative_confidence → "insufficient"
            _make_block("data_quality", []),
            _make_block("liquidity", [], level=None),
            _make_block("volatility", [], level=None),
            _make_block("predictability", [], level=None),
        ],
        overall_confidence="insufficient",
    )
    proxies = {
        "nonlinearity": _make_metric("nonlinearity", None, 5, "insufficient"),
        "pattern_recurrence": _make_metric("pattern_recurrence", None, 5, "insufficient"),
        "temporal_stability": _make_metric("temporal_stability", None, 5, "insufficient"),
    }
    result = compute_fitness_score(profile, proxies)
    assert result.fitness_score is None
    assert result.overall_confidence == "insufficient"


# ---------------------------------------------------------------------------
# available=False profile → fitness_score None
# ---------------------------------------------------------------------------

def test_unavailable_profile_returns_none_score() -> None:
    """available=False 的 profile 应立即返回 fitness_score=None。"""
    profile = _make_full_profile(available=False, unavailable_reason="本地无数据")
    proxies = _make_proxies()

    result = compute_fitness_score(profile, proxies)

    assert result.available is False
    assert result.fitness_score is None
    assert result.contributions == []
    assert result.overall_confidence == "insufficient"
    assert result.note is not None


def test_unavailable_profile_fallback_note() -> None:
    """available=False 且无 unavailable_reason 时，note 有合理默认值。"""
    profile = _make_full_profile(available=False, unavailable_reason=None)
    result = compute_fitness_score(profile, {})
    assert result.note is not None and len(result.note) > 0


# ---------------------------------------------------------------------------
# 示例测试：强 profile 比弱 profile 分高
# ---------------------------------------------------------------------------

def _make_weak_profile() -> SymbolProfile:
    """构造"弱"画像：低流动性、低波动、不确定结构、数据量少。"""
    return _make_full_profile(
        dq_bars=30.0,
        dq_conf="low",
        liq_level="low",
        liq_conf="low",
        vol_level="low",
        vol_conf="low",
        pred_level="indeterminate",
        pred_conf="low",
    )


def _make_strong_profile() -> SymbolProfile:
    """构造"强"画像：高流动性、高波动、趋势结构、充足数据量。"""
    return _make_full_profile(
        dq_bars=500.0,
        dq_conf="high",
        liq_level="high",
        liq_conf="high",
        vol_level="high",
        vol_conf="high",
        pred_level="trending",
        pred_conf="high",
    )


def test_strong_profile_scores_higher_than_weak() -> None:
    """强 profile + 强代理指标的综合分应高于弱 profile + 弱代理指标。"""
    strong_proxies = _make_proxies(
        nonlinearity=0.9, nl_conf="high",
        pattern=0.85, pat_conf="high",
        temporal=0.9, temp_conf="high",
    )
    weak_proxies = _make_proxies(
        nonlinearity=0.1, nl_conf="low",
        pattern=0.1, pat_conf="low",
        temporal=0.15, temp_conf="low",
    )

    strong_score = compute_fitness_score(_make_strong_profile(), strong_proxies)
    weak_score = compute_fitness_score(_make_weak_profile(), weak_proxies)

    assert strong_score.fitness_score is not None
    assert weak_score.fitness_score is not None
    assert strong_score.fitness_score > weak_score.fitness_score, (
        f"强 profile 分 ({strong_score.fitness_score:.4f}) "
        f"不高于弱 profile 分 ({weak_score.fitness_score:.4f})"
    )


# ---------------------------------------------------------------------------
# fitness_score 全局有界性与贡献自洽（通过 compute_fitness_score 端到端验证）
# ---------------------------------------------------------------------------

def test_fitness_score_in_unit_interval() -> None:
    """正常 profile 的 fitness_score 应在 [0,1]。"""
    profile = _make_full_profile()
    proxies = _make_proxies()
    result = compute_fitness_score(profile, proxies)

    assert result.fitness_score is not None
    assert 0.0 <= result.fitness_score <= 1.0


def test_contributions_sum_equals_fitness_score() -> None:
    """有效维度的 contribution 之和应等于 fitness_score（Property 3 端到端）。"""
    profile = _make_full_profile()
    proxies = _make_proxies()
    result = compute_fitness_score(profile, proxies)

    assert result.fitness_score is not None

    included_sum = sum(c.contribution for c in result.contributions if c.weight > 0)
    assert abs(included_sum - result.fitness_score) < 1e-9, (
        f"Σcontribution={included_sum:.9f} != fitness_score={result.fitness_score:.9f}"
    )


def test_contributions_weights_sum_to_one() -> None:
    """有效维度的归一化 weight（renorm_weight）之和应等于 1.0。"""
    profile = _make_full_profile()
    proxies = _make_proxies()
    result = compute_fitness_score(profile, proxies)

    assert result.fitness_score is not None
    weight_sum = sum(c.weight for c in result.contributions if c.weight > 0)
    assert abs(weight_sum - 1.0) < 1e-9, f"有效维度权重之和 {weight_sum:.9f} != 1.0"


# ---------------------------------------------------------------------------
# 等级映射边界检查
# ---------------------------------------------------------------------------

def test_level_value_mapping_monotonic_liquidity() -> None:
    """LIQUIDITY_LEVEL_VALUE: low < medium < high（单调性）。"""
    assert LIQUIDITY_LEVEL_VALUE["low"] < LIQUIDITY_LEVEL_VALUE["medium"]
    assert LIQUIDITY_LEVEL_VALUE["medium"] < LIQUIDITY_LEVEL_VALUE["high"]


def test_level_value_mapping_monotonic_volatility() -> None:
    """VOLATILITY_LEVEL_VALUE: low < medium < high（CNN 倾向高波动，单调上升）。"""
    assert VOLATILITY_LEVEL_VALUE["low"] < VOLATILITY_LEVEL_VALUE["medium"]
    assert VOLATILITY_LEVEL_VALUE["medium"] < VOLATILITY_LEVEL_VALUE["high"]


def test_level_value_in_range() -> None:
    """所有等级映射值应在 [0,1]。"""
    for d in (LIQUIDITY_LEVEL_VALUE, VOLATILITY_LEVEL_VALUE, PREDICTABILITY_LEVEL_VALUE):
        for k, v in d.items():
            assert 0.0 <= v <= 1.0, f"等级 {k!r} 的值 {v} 超出 [0,1]"


# ---------------------------------------------------------------------------
# vt_symbol 传递正确
# ---------------------------------------------------------------------------

def test_tier1score_vt_symbol_from_profile() -> None:
    """Tier1Score.vt_symbol 应与 profile.input.vt_symbol 一致。"""
    profile = _make_full_profile()
    result = compute_fitness_score(profile, _make_proxies())
    assert result.vt_symbol == profile.input.vt_symbol


# ---------------------------------------------------------------------------
# 自定义权重：仅含单维度，分值应等于该维度值
# ---------------------------------------------------------------------------

def test_single_included_dim_score_equals_value() -> None:
    """仅一个有效维度时，fitness_score 应等于该维度的归一化值。"""
    # 只保留 nonlinearity 有效，其余全 insufficient
    profile = SymbolProfile(
        input=_make_profile_input(),
        available=True,
        blocks=[],  # 无任何画像块
        overall_confidence="insufficient",
    )
    nl_val = 0.73
    proxies = {
        "nonlinearity": _make_metric("nonlinearity", nl_val, 200, "medium"),
        "pattern_recurrence": _make_metric("pattern_recurrence", None, 5, "insufficient"),
        "temporal_stability": _make_metric("temporal_stability", None, 5, "insufficient"),
    }
    rules = ScreeningRules(
        weights={
            "nonlinearity": 2.0,
            "data_quality": 1.0,
            "liquidity": 1.0,
            "volatility": 1.0,
            "predictability": 1.0,
            "pattern_recurrence": 1.0,
            "temporal_stability": 1.0,
        }
    )
    result = compute_fitness_score(profile, proxies, rules)

    # 只有 nonlinearity 参与，分值应等于 nl_val
    assert result.fitness_score is not None
    assert abs(result.fitness_score - nl_val) < 1e-9, (
        f"预期 {nl_val}, 实际 {result.fitness_score}"
    )
