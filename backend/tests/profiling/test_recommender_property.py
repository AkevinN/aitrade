from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.profiling.recommender import build_scheme_suggestion
from aitrade.profiling.rules import DEFAULT_RULES
from aitrade.profiling.types import MetricBlock, MetricValue


def _metric(key: str, confidence: str, value: float | None = 1.0) -> MetricValue:
    return MetricValue(
        key=key,
        value=value,
        effective_sample=500 if confidence in {"medium", "high"} else 10,
        confidence=confidence,
    )


def _blocks(confidence: str = "high") -> list[MetricBlock]:
    return [
        MetricBlock(
            block="data_quality",
            metrics=[_metric("count_valid_bars", confidence), _metric("gap_ratio", confidence)],
        ),
        MetricBlock(
            block="liquidity",
            level="medium",
            metrics=[_metric("avg_turnover", confidence, 10_000_000.0)],
        ),
        MetricBlock(
            block="volatility",
            level="low",
            metrics=[_metric("realized_volatility", confidence, 0.01)],
        ),
        MetricBlock(
            block="predictability",
            level="trending",
            metrics=[_metric("variance_ratio", confidence, 1.2)],
        ),
    ]


# Feature: symbol-profiling, Property 10: 建议草稿恒为只读草稿且受置信度门控
# Validates: Requirements 7.3, 8.1, 8.2, 8.4, 8.5
@settings(max_examples=100)
@given(confidence=st.sampled_from(["insufficient", "low", "medium", "high"]))
def test_property10_suggestion_is_draft_and_confidence_gated(confidence: str) -> None:
    suggestion = build_scheme_suggestion(
        vt_symbol="600030.SSE",
        interval="30m",
        blocks=_blocks(confidence),
        overall_confidence=confidence,
        rules=DEFAULT_RULES,
    )

    assert suggestion.status == "draft"
    assert all(item.reason for item in suggestion.items)
    if confidence in {"insufficient", "low"}:
        assert suggestion.degraded
        assert all(item.field in {"data.lookback_days", "interval"} for item in suggestion.items)


# Feature: symbol-profiling, Property 11: 建议结构与 Scheme 字段兼容
# Validates: Requirements 8.1
@settings(max_examples=100)
@given(overall=st.sampled_from(["medium", "high"]))
def test_property11_suggestion_fields_are_scheme_compatible(overall: str) -> None:
    suggestion = build_scheme_suggestion(
        vt_symbol="600030.SSE",
        interval="30m",
        blocks=_blocks("high"),
        overall_confidence=overall,
        rules=DEFAULT_RULES,
    )

    allowed_prefixes = (
        "label_spec.",
        "predictor.",
        "strategy.",
        "cost.",
        "data.",
        "interval",
    )
    assert suggestion.items
    assert all(item.field == "interval" or item.field.startswith(allowed_prefixes) for item in suggestion.items)


def test_low_volatility_adds_tp_sl_risk_note() -> None:
    suggestion = build_scheme_suggestion(
        vt_symbol="600030.SSE",
        interval="30m",
        blocks=_blocks("high"),
        overall_confidence="high",
        rules=DEFAULT_RULES,
    )

    fields = {item.field for item in suggestion.items}
    assert "label_spec.take_profit" in fields
    assert "label_spec.stop_loss" in fields
    assert suggestion.note and "低波动" in suggestion.note
