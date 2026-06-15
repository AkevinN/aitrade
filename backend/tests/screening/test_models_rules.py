"""
CNN 选股模块基础模型与规则的理智检验（sanity tests）。

覆盖：
- CNNScreeningRequest 拒绝缺失 as_of 与 lookback_days <= 0 的请求
- DEFAULT_SCREENING_RULES 所有权重非负且 rules_id 存在
- ScreeningResult.status 恒为 "draft"

Feature: cnn-stock-screening, Tasks 1.3 / 1.4 / 1.5
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from aitrade.models.screening import CNNScreeningRequest
from aitrade.screening import DEFAULT_SCREENING_RULES, ScreeningResult, ScreeningRules


# ---- Task 1.4: CNNScreeningRequest 校验 ----


def test_cnn_screening_request_rejects_missing_as_of() -> None:
    """as_of 为必填字段；缺失时 pydantic 应抛 ValidationError。"""
    with pytest.raises(ValidationError):
        CNNScreeningRequest(
            name="test",
            lookback_days=365,
            # as_of 故意不传
        )  # type: ignore[call-arg]


def test_cnn_screening_request_rejects_zero_lookback() -> None:
    """lookback_days 必须 > 0；传 0 时应抛 ValidationError。"""
    with pytest.raises(ValidationError):
        CNNScreeningRequest(
            name="test",
            as_of=datetime(2025, 1, 1),
            lookback_days=0,
        )


def test_cnn_screening_request_rejects_negative_lookback() -> None:
    """lookback_days 必须 > 0；传负数时应抛 ValidationError。"""
    with pytest.raises(ValidationError):
        CNNScreeningRequest(
            name="test",
            as_of=datetime(2025, 1, 1),
            lookback_days=-10,
        )


def test_cnn_screening_request_valid_defaults() -> None:
    """合法的最小请求（只传 name/as_of/lookback_days）应可构造，并采用全部文档默认值。"""
    req = CNNScreeningRequest(
        name="sanity",
        as_of=datetime(2025, 6, 1),
        lookback_days=250,
    )
    # ---- 基础参数 ----
    assert req.interval == "d"
    # ---- 漏斗参数 ----
    assert req.top_k == 15
    assert req.run_tier2 is True
    assert req.min_confidence == "low"
    # ---- Tier-2 超参 ----
    assert req.objective == "classification"
    assert req.eval_start is None
    # ---- Universe 过滤 ----
    assert req.exchange is None
    assert req.include_symbols == []
    assert req.exclude_symbols == []
    # ---- 持久化 ----
    assert req.persist is False


# ---- Task 1.5: DEFAULT_SCREENING_RULES 权重与版本 ----


def test_default_rules_has_rules_id() -> None:
    """DEFAULT_SCREENING_RULES 必须有非空的 rules_id。"""
    assert isinstance(DEFAULT_SCREENING_RULES.rules_id, str)
    assert DEFAULT_SCREENING_RULES.rules_id


def test_default_rules_weights_nonnegative() -> None:
    """DEFAULT_SCREENING_RULES 中所有权重必须 >= 0（Property 3 前置条件）。"""
    for dim, w in DEFAULT_SCREENING_RULES.weights.items():
        assert w >= 0, f"维度 {dim!r} 权重 {w} 为负"


def test_screening_rules_rejects_negative_weight() -> None:
    """ScreeningRules 构造时若传入负权重应抛 ValidationError。"""
    with pytest.raises(ValidationError):
        ScreeningRules(weights={"data_quality": -1.0})


def test_default_rules_has_expected_dimensions() -> None:
    """DEFAULT_SCREENING_RULES.weights 应同时包含四个画像块与三个 CNN 代理指标维度。"""
    expected = {
        "data_quality", "liquidity", "volatility", "predictability",
        "nonlinearity", "pattern_recurrence", "temporal_stability",
    }
    assert expected.issubset(DEFAULT_SCREENING_RULES.weights.keys())


# ---- Task 1.3: ScreeningResult.status 恒为 "draft" ----


def test_screening_result_status_is_always_draft() -> None:
    """ScreeningResult.status 固定字面量 'draft'，无论如何构造都不变（Requirement 11.1）。"""
    result = ScreeningResult(
        run_id="test-run-001",
        created_at=datetime(2025, 6, 1, 12, 0),
        rules_id="screening-builtin-v1",
        universe_size=100,
    )
    assert result.status == "draft"


def test_screening_result_status_cannot_be_overridden() -> None:
    """即使显式传 status='draft'，也应正常构造；非 'draft' 的值则不合法。"""
    # 正常路径：显式传 "draft" 也 OK
    result = ScreeningResult(
        run_id="test-run-002",
        created_at=datetime(2025, 6, 1),
        rules_id="screening-builtin-v1",
        universe_size=0,
        status="draft",
    )
    assert result.status == "draft"

    # 异常路径：传非 "draft" 值应被 pydantic 拒绝
    with pytest.raises(ValidationError):
        ScreeningResult(
            run_id="test-run-003",
            created_at=datetime(2025, 6, 1),
            rules_id="screening-builtin-v1",
            universe_size=0,
            status="live",  # type: ignore[arg-type]
        )
