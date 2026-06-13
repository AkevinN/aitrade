"""threshold_scale_check 纯函数的单元真值表测试 + 属性测试。

覆盖范围：
- 真值表：概率型（classification/path_class）合法/越界、
  regression 合法/误用/边界、None/"" 跳过。
- Property 5：阈值尺度校验规则恒成立 + 纯函数确定性（一致性子句）。

# Feature: cnn-eval-honesty-fixes, Property 5:
#   对任意 (objective, buy, sell)——概率型且 buy/sell 越出 [0,1] → 非空违规；
#   regression 且 buy≥0.5 → 非空违规；objective 为 None → 空（跳过）；
#   既有合法配置（0.6/0.4、0.005/-0.005）→ 空。
#   回测/实盘同一函数对同一输入返回同一结果（纯函数确定性）。
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.cnn.thresholds import threshold_scale_check

# ---------------------------------------------------------------------------
# 真值表测试（示例测试）
# ---------------------------------------------------------------------------


class TestThresholdScaleCheckTruthTable:
    """逐格覆盖 threshold_scale_check 的合法/违规分支。"""

    # ---- classification ---------------------------------------------------

    def test_classification_valid_buy_sell(self):
        """classification buy=0.6, sell=0.4 → 合法，返回空列表。"""
        assert threshold_scale_check("classification", 0.6, 0.4) == []

    def test_classification_buy_above_one(self):
        """classification buy=1.5 → 超出 [0,1]，违规。"""
        result = threshold_scale_check("classification", 1.5)
        assert len(result) > 0

    def test_classification_buy_below_zero(self):
        """classification buy=-0.1 → 低于 [0,1]，违规。"""
        result = threshold_scale_check("classification", -0.1)
        assert len(result) > 0

    def test_classification_sell_above_one(self):
        """classification buy=0.6, sell=1.2 → sell 超出 [0,1]，违规。"""
        result = threshold_scale_check("classification", 0.6, 1.2)
        assert len(result) > 0

    def test_classification_sell_none(self):
        """classification buy=0.6, sell=None → sell 不传，只校验 buy。"""
        assert threshold_scale_check("classification", 0.6, None) == []

    def test_classification_boundary_buy_zero(self):
        """classification buy=0.0 → 边界值合法。"""
        assert threshold_scale_check("classification", 0.0) == []

    def test_classification_boundary_buy_one(self):
        """classification buy=1.0 → 边界值合法。"""
        assert threshold_scale_check("classification", 1.0) == []

    # ---- path_class -------------------------------------------------------

    def test_path_class_valid_buy(self):
        """path_class buy=0.6 → 合法。"""
        assert threshold_scale_check("path_class", 0.6) == []

    def test_path_class_buy_above_one(self):
        """path_class buy=1.1 → 超出 [0,1]，违规。"""
        result = threshold_scale_check("path_class", 1.1)
        assert len(result) > 0

    def test_path_class_buy_below_zero(self):
        """path_class buy=-0.05 → 低于 [0,1]，违规。"""
        result = threshold_scale_check("path_class", -0.05)
        assert len(result) > 0

    # ---- regression -------------------------------------------------------

    def test_regression_valid_positive_buy(self):
        """regression buy=0.005 → 合法收益阈值，通过。"""
        assert threshold_scale_check("regression", 0.005) == []

    def test_regression_valid_with_negative_sell(self):
        """regression buy=0.005, sell=-0.005 → 收益口径，sell 为负数合法，通过。"""
        assert threshold_scale_check("regression", 0.005, -0.005) == []

    def test_regression_misuse_high_buy(self):
        """regression buy=0.6 → 疑似误用概率阈值（+60% 收益不可达），违规。"""
        result = threshold_scale_check("regression", 0.6)
        assert len(result) > 0

    def test_regression_boundary_buy_exact_half(self):
        """regression buy=0.5 → 边界，>= 0.5 违规。"""
        result = threshold_scale_check("regression", 0.5)
        assert len(result) > 0

    def test_regression_just_below_boundary(self):
        """regression buy=0.49 → 低于 0.5，合法。"""
        assert threshold_scale_check("regression", 0.49) == []

    # ---- None / empty objective -------------------------------------------

    def test_none_objective_skipped(self):
        """objective=None → 向后兼容跳过，返回空列表。"""
        assert threshold_scale_check(None, 1.5) == []

    def test_empty_string_objective_skipped(self):
        """objective="" → falsy，跳过，返回空列表。"""
        assert threshold_scale_check("", 1.5) == []

    # ---- 既有合法配置不误拦 -------------------------------------------------

    def test_existing_valid_classification_config(self):
        """既有合法 classification 配置 buy=0.6, sell=0.4 → 不误拦。"""
        assert threshold_scale_check("classification", 0.6, 0.4) == []

    def test_existing_valid_regression_config(self):
        """既有合法 regression 配置 buy=0.005, sell=-0.005 → 不误拦。"""
        assert threshold_scale_check("regression", 0.005, -0.005) == []

    # ---- 未知 objective 宽容通过 ------------------------------------------

    def test_unknown_objective_passes(self):
        """未知 objective（非 classification/path_class/regression）→ 无规则，通过。"""
        assert threshold_scale_check("future_model", 999.0) == []


# ---------------------------------------------------------------------------
# Property 5 属性测试（Hypothesis）
# ---------------------------------------------------------------------------

# Feature: cnn-eval-honesty-fixes, Property 5: 阈值尺度校验 + 回测/实盘同输入同判定一致性子句

_OBJECTIVES = st.sampled_from(["classification", "regression", "path_class", None])
_THRESHOLDS = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
_OPTIONAL_THRESHOLDS = st.one_of(st.none(), _THRESHOLDS)


def _expected_violations(objective: str | None, buy: float, sell: float | None) -> bool:
    """独立于实现的期望判定逻辑（避免同义反复）。

    Returns:
        True 表示应有至少一条违规。
    """
    if not objective:
        return False
    if objective in {"classification", "path_class"}:
        buy_bad = not (0.0 <= buy <= 1.0)
        sell_bad = sell is not None and not (0.0 <= sell <= 1.0)
        return buy_bad or sell_bad
    if objective == "regression":
        return buy >= 0.5
    return False  # 未知 objective → 无规则


@given(
    objective=_OBJECTIVES,
    buy=_THRESHOLDS,
    sell=_OPTIONAL_THRESHOLDS,
)
@settings(max_examples=100)
def test_property5_threshold_scale_rules(
    objective: str | None,
    buy: float,
    sell: float | None,
) -> None:
    """Property 5: 阈值尺度校验规则对任意输入恒成立。

    # Feature: cnn-eval-honesty-fixes, Property 5: <原文>
    独立期望逻辑：
    - 概率型且 buy/sell 越出 [0,1] → 非空违规
    - regression 且 buy>=0.5 → 非空违规
    - objective 为 None/"" → 空（跳过）
    - 其余情况 → 空（通过）
    """
    result = threshold_scale_check(objective, buy, sell)

    assert isinstance(result, list), "返回值必须是 list"
    for item in result:
        assert isinstance(item, str), "违规项必须是字符串"

    expected_has_violation = _expected_violations(objective, buy, sell)
    if expected_has_violation:
        assert len(result) > 0, (
            f"期望有违规但结果为空: objective={objective!r}, buy={buy}, sell={sell}"
        )
    else:
        assert len(result) == 0, (
            f"期望无违规但有结果: objective={objective!r}, buy={buy}, sell={sell}, "
            f"violations={result}"
        )


@given(
    objective=_OBJECTIVES,
    buy=_THRESHOLDS,
    sell=_OPTIONAL_THRESHOLDS,
)
@settings(max_examples=100)
def test_property5_consistency_pure_function(
    objective: str | None,
    buy: float,
    sell: float | None,
) -> None:
    """Property 5 一致性子句：纯函数对同一输入两次调用结果相等。

    # Feature: cnn-eval-honesty-fixes, Property 5: 回测/实盘同输入同判定的一致性子句
    因为回测 API 与实盘 service 都调用同一个 threshold_scale_check，
    同一输入必然给出同一判定——此处固化该契约：两次调用结果列表相等。
    """
    result_first = threshold_scale_check(objective, buy, sell)
    result_second = threshold_scale_check(objective, buy, sell)
    assert result_first == result_second, (
        f"纯函数同一输入两次调用结果不一致: "
        f"objective={objective!r}, buy={buy}, sell={sell}, "
        f"first={result_first}, second={result_second}"
    )
