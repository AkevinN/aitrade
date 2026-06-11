from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.profiling.rules import (
    DEFAULT_RULES,
    confidence_for,
    liquidity_level,
    overall_confidence,
    structure_judgement,
    volatility_level,
)


_rank = {"insufficient": 0, "low": 1, "medium": 2, "high": 3}
_level_rank = {"low": 0, "medium": 1, "high": 2}


# Feature: symbol-profiling, Property 6: 置信度分档单调
# Validates: Requirements 7.1, 7.2
@settings(max_examples=100)
@given(a=st.integers(min_value=0, max_value=1000), b=st.integers(min_value=0, max_value=1000))
def test_property6_confidence_is_monotonic(a: int, b: int) -> None:
    lo, hi = sorted([a, b])
    assert _rank[confidence_for("default", lo, DEFAULT_RULES)] <= _rank[
        confidence_for("default", hi, DEFAULT_RULES)
    ]


# Feature: symbol-profiling, Property 8: 综合置信度不高于关键指标最低值
# Validates: Requirements 7.4
@settings(max_examples=100)
@given(levels=st.lists(st.sampled_from(list(_rank)), min_size=1, max_size=12))
def test_property8_overall_confidence_is_lowest(levels: list[str]) -> None:
    overall = overall_confidence(levels)
    assert _rank[overall] == min(_rank[level] for level in levels)


# Feature: symbol-profiling, Property 9: 等级判定的阈值单调性
# Validates: Requirements 4.3, 5.2
@settings(max_examples=100)
@given(a=st.floats(min_value=0, max_value=1e8), b=st.floats(min_value=0, max_value=1e8))
def test_property9_levels_are_monotonic(a: float, b: float) -> None:
    lo, hi = sorted([a, b])
    assert _level_rank[liquidity_level(lo, DEFAULT_RULES)] <= _level_rank[
        liquidity_level(hi, DEFAULT_RULES)
    ]
    assert _level_rank[volatility_level(lo / 1e9, DEFAULT_RULES)] <= _level_rank[
        volatility_level(hi / 1e9, DEFAULT_RULES)
    ]


def test_structure_judgement_examples() -> None:
    assert structure_judgement(0.65, 1.2, 0.6, "medium") == "trending"
    assert structure_judgement(0.35, 0.8, 0.01, "high") == "mean_reverting"
    assert structure_judgement(0.65, 0.8, None, "low") == "indeterminate"
