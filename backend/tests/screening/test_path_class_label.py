"""
CNN 选股支持 path_class（oco 标签）——label_spec 透传与向后兼容测试。

覆盖 cnn-screening-path-class 特性的后端正确性属性：
- Property 1: ``_build_wf_request`` 把请求级 label_spec（非 None）透传至 Tier-2
  的 ``CNNWalkForwardRequest``，否则回退 ``ScreeningRules.label_spec`` 默认；
  objective 始终照原样透传。
- Property 3: ``objective ∈ {classification, regression}`` 且不携带 label_spec 时，
  透传出的 label_spec 恒等于规则默认（next_bar），与改造前行为逐字段一致。

测试策略：直接构造一个最小 ``Tier2Window`` 喂给 ``_build_wf_request``，不触发
任何真实行情/训练（纯接线逻辑）。用 Hypothesis 随机生成 objective 与 label_spec
组合，断言透传规则。

Feature: cnn-screening-path-class
"""

from __future__ import annotations

from datetime import date
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.models.alpha import LabelMode, LabelSpec
from aitrade.screening.rules import ScreeningRules
from aitrade.screening.runner import ScreeningRunner, Tier2Window


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


class _NoLab:
    """占位 lab：``_build_wf_request`` 不触达 lab，构造 Runner 仅为拿到方法。"""


def _window() -> Tier2Window:
    """构造一个满足不变量（start <= end <= as_of）的最小 Tier2Window。"""
    return Tier2Window(
        start=date(2024, 1, 1),
        end=date(2025, 6, 1),
        train_days=480,
        fold_test_days=90,
        step_days=90,
        eval_window_days=900,
        n_seeds=1,
        epochs=30,
        local_start=date(2018, 1, 1),
        local_end=date(2025, 6, 1),
    )


def _make_request(objective: str, label_spec: LabelSpec | None) -> Any:
    """构造一个带指定 objective / label_spec 的最小选股请求。"""
    from datetime import datetime

    from aitrade.models.screening import CNNScreeningRequest

    return CNNScreeningRequest(
        name="t",
        as_of=datetime(2025, 6, 1),
        lookback_days=365,
        objective=objective,  # type: ignore[arg-type]
        label_spec=label_spec,
    )


# label_spec 策略：None / oco / next_bar 三种代表性取值。
_label_specs = st.one_of(
    st.none(),
    st.just(LabelSpec(mode="oco", take_profit=0.03, stop_loss=0.02, max_hold=10)),
    st.just(LabelSpec(mode="next_bar")),
)
_objectives = st.sampled_from(["classification", "regression", "path_class"])


# ---------------------------------------------------------------------------
# Property 1: 请求级 label_spec 透传，缺省回退规则默认
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(objective=_objectives, label_spec=_label_specs)
def test_build_wf_request_forwards_label_spec(objective: str, label_spec: LabelSpec | None) -> None:
    # Feature: cnn-screening-path-class, Property 1: label_spec 透传一致
    """``_build_wf_request`` 的 label_spec 恒为「请求级（非 None）否则规则默认」，objective 原样透传。"""
    rules = ScreeningRules()  # 默认 label_spec = next_bar
    runner = ScreeningRunner(_NoLab())
    runner.rules = rules  # 显式锚定，便于断言回退目标

    req = _make_request(objective, label_spec)
    wf = runner._build_wf_request("000001.SZSE", req, _window())

    expected = label_spec if label_spec is not None else rules.label_spec
    assert wf.label_spec == expected, (
        f"label_spec 透传错误：objective={objective} 请求={label_spec} "
        f"期望={expected} 实际={wf.label_spec}"
    )
    assert wf.objective == objective  # objective 始终照原样透传


# ---------------------------------------------------------------------------
# Property 3: 不携带 label_spec 时与改造前等价（next_bar）
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(objective=st.sampled_from(["classification", "regression"]))
def test_build_wf_request_backward_compat_default(objective: str) -> None:
    # Feature: cnn-screening-path-class, Property 3: 向后兼容不变性
    """非 path_class 且不带 label_spec 时，透传出的 label_spec 恒为规则默认（next_bar）。"""
    runner = ScreeningRunner(_NoLab())
    req = _make_request(objective, None)
    wf = runner._build_wf_request("000001.SZSE", req, _window())

    assert wf.label_spec.mode == LabelMode.NEXT_BAR
    # 与 ScreeningRules 默认逐字段一致（向后兼容守护）。
    assert wf.label_spec == runner.rules.label_spec


def test_build_wf_request_path_class_oco_passthrough() -> None:
    # Feature: cnn-screening-path-class, Property 1: path_class 携带 oco 标签透传
    """path_class + oco label_spec 时，Tier-2 WF 请求拿到完整 oco 标签（mode/tp/sl/max_hold）。"""
    runner = ScreeningRunner(_NoLab())
    oco = LabelSpec(mode="oco", take_profit=0.05, stop_loss=0.03, max_hold=8)
    req = _make_request("path_class", oco)
    wf = runner._build_wf_request("600000.SSE", req, _window())

    assert wf.objective == "path_class"
    assert wf.label_spec.mode == LabelMode.OCO
    assert wf.label_spec.take_profit == 0.05
    assert wf.label_spec.stop_loss == 0.03
    assert wf.label_spec.max_hold == 8
