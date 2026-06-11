"""
OCO（三重障碍）路径依赖标签验收测试。

覆盖：
1. _normalize_label_spec：oco 模式默认值与校验（缺 tp/sl 抛错、max_hold 兜底）。
2. _oco_label_value：止盈/止损/时间三种出场的标签口径（分类 & 回归）、去噪、无前视丢样本。
3. consistency：oco label 的持有期解析、auto 推导 OCO 出场、一致性告警分支。
"""

from __future__ import annotations

import numpy as np
import pytest

from aitrade.cnn.dataset import _normalize_label_spec, _oco_label_value
from aitrade.cnn.consistency import (
    check_label_strategy_consistency,
    derive_strategy_exit_from_label,
    label_holding_horizon,
)


# ---------------------------------------------------------------------------
# 1. _normalize_label_spec —— oco 模式
# ---------------------------------------------------------------------------
def test_normalize_oco_defaults_and_horizon_fallback() -> None:
    spec = _normalize_label_spec(
        {"mode": "oco", "take_profit": 0.05, "stop_loss": 0.05, "horizon": 8}
    )
    assert spec["mode"] == "oco"
    assert spec["take_profit"] == 0.05
    assert spec["stop_loss"] == 0.05
    # max_hold 缺省时回退到 horizon
    assert spec["max_hold"] == 8
    # stop_first 默认 True（与回测保守口径一致）
    assert spec["stop_first"] is True


def test_normalize_oco_requires_positive_tp_sl() -> None:
    with pytest.raises(ValueError, match="take_profit 与 stop_loss"):
        _normalize_label_spec({"mode": "oco", "take_profit": 0.0, "stop_loss": 0.05})
    with pytest.raises(ValueError, match="take_profit 与 stop_loss"):
        _normalize_label_spec({"mode": "oco", "take_profit": 0.05})


def test_normalize_oco_bad_max_hold_raises() -> None:
    with pytest.raises(ValueError, match="max_hold"):
        _normalize_label_spec(
            {"mode": "oco", "take_profit": 0.05, "stop_loss": 0.05, "max_hold": 0}
        )


# ---------------------------------------------------------------------------
# 2. _oco_label_value —— 三种出场口径
# ---------------------------------------------------------------------------
# 公共序列长度 6（索引 0..5）；anchor=0 → entry_index=1，entry_price=100。
# max_hold=3 → 扫描 j∈[2,4]，兜底平仓在 index 5；fallback=anchor+max_hold+2=5<6 成立。
_SPEC = {"mode": "oco", "take_profit": 0.05, "stop_loss": 0.05, "max_hold": 3, "stop_first": True}


def _flat(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def test_oco_take_profit_hit() -> None:
    opens = _flat([100, 100, 100, 100, 100, 100])
    highs = _flat([100, 100, 106, 100, 100, 100])  # j=2 触止盈价 105
    lows = _flat([100, 100, 100, 100, 100, 100])
    # 分类：止盈触发 → 1
    status, label, ret = _oco_label_value(0, opens, highs, lows, _SPEC, "classification", 0.0, "drop")
    assert status == "ok"
    assert label == 1.0
    assert ret == pytest.approx(0.05)
    # 回归：label = 真实出场收益
    status_r, label_r, ret_r = _oco_label_value(0, opens, highs, lows, _SPEC, "regression", 0.0, "drop")
    assert status_r == "ok"
    assert label_r == pytest.approx(0.05)


def test_oco_stop_loss_hit() -> None:
    opens = _flat([100, 100, 100, 100, 100, 100])
    highs = _flat([100, 100, 100, 100, 100, 100])
    lows = _flat([100, 100, 94, 100, 100, 100])  # j=2 触止损价 95
    status, label, ret = _oco_label_value(0, opens, highs, lows, _SPEC, "classification", 0.0, "drop")
    assert status == "ok"
    assert label == 0.0
    assert ret == pytest.approx(-0.05)


def test_oco_time_exit_positive() -> None:
    # 持有期内都不触发 → 在 index 5 开盘按时间止损平仓
    opens = _flat([100, 100, 100, 100, 100, 103])
    highs = _flat([100, 100, 104, 104, 104, 103])  # 不到止盈价 105
    lows = _flat([100, 100, 96, 96, 96, 103])  # 不到止损价 95
    status, label, ret = _oco_label_value(0, opens, highs, lows, _SPEC, "classification", 0.0, "drop")
    assert status == "ok"
    assert label == 1.0  # ret>0 → 上涨
    assert ret == pytest.approx(0.03)


def test_oco_regression_neutral_dropped_by_threshold() -> None:
    # 时间止损 ret=0.03，回归阈值 0.05 → 落入去噪区，丢弃
    opens = _flat([100, 100, 100, 100, 100, 103])
    highs = _flat([100, 100, 104, 104, 104, 103])
    lows = _flat([100, 100, 96, 96, 96, 103])
    status, label, ret = _oco_label_value(0, opens, highs, lows, _SPEC, "regression", 0.05, "drop")
    assert status == "neutral"
    assert label is None
    assert ret == pytest.approx(0.03)


def test_oco_no_lookahead_skips_when_insufficient_future() -> None:
    # anchor=1 → fallback=anchor+max_hold+2=6 越界（n=6），无法形成完整样本 → skip
    opens = _flat([100, 100, 100, 100, 100, 100])
    highs = _flat([100, 100, 106, 100, 100, 100])
    lows = _flat([100, 100, 100, 100, 100, 100])
    status, label, ret = _oco_label_value(1, opens, highs, lows, _SPEC, "classification", 0.0, "drop")
    assert status == "skip"
    assert label is None and ret is None


def test_oco_stop_first_when_both_hit_same_bar() -> None:
    # 同一根 bar 同时触止盈与止损 → 保守假设止损先到（stop_first=True）
    opens = _flat([100, 100, 100, 100, 100, 100])
    highs = _flat([100, 100, 106, 100, 100, 100])  # 触止盈
    lows = _flat([100, 100, 94, 100, 100, 100])  # 同根也触止损
    status, label, ret = _oco_label_value(0, opens, highs, lows, _SPEC, "classification", 0.0, "drop")
    assert status == "ok"
    assert label == 0.0  # 止损先到
    assert ret == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# 3. consistency —— oco 扩展
# ---------------------------------------------------------------------------
def test_holding_horizon_oco_is_none() -> None:
    # OCO realized 持有期不固定 → None
    assert label_holding_horizon({"mode": "oco", "max_hold": 10}) is None


def test_derive_exit_from_oco() -> None:
    cfg = derive_strategy_exit_from_label(
        {"mode": "oco", "take_profit": 0.03, "stop_loss": 0.02, "max_hold": 10}
    )
    assert cfg == {
        "exit_mode": "oco",
        "take_profit": 0.03,
        "stop_loss": 0.02,
        "hold_days": 10,
    }


def test_consistency_oco_label_with_oco_exit_aligned() -> None:
    warnings = check_label_strategy_consistency(
        {"mode": "oco", "take_profit": 0.03, "stop_loss": 0.02, "max_hold": 10},
        exit_mode="oco", hold_days=10, input_interval="d",
    )
    assert any("对齐" in w for w in warnings)


def test_consistency_oco_exit_with_nonoco_label_warns() -> None:
    warnings = check_label_strategy_consistency(
        {"mode": "horizon_bars", "horizon": 5, "price_ref": "next_open"},
        exit_mode="oco", hold_days=5, input_interval="d",
    )
    assert any("不一致" in w for w in warnings)


# ---------------------------------------------------------------------------
# 4. API 模型 LabelSpec —— oco 字段透传与枚举规整
# ---------------------------------------------------------------------------
def test_label_spec_model_carries_oco_fields() -> None:
    from aitrade.models.alpha import LabelMode, LabelSpec

    spec = LabelSpec(mode="oco", take_profit=0.03, stop_loss=0.02, max_hold=10)
    dumped = spec.model_dump()
    assert dumped["mode"] == LabelMode.OCO
    assert dumped["take_profit"] == 0.03
    assert dumped["stop_loss"] == 0.02
    assert dumped["max_hold"] == 10
    assert dumped["stop_first"] is True


def test_normalize_coerces_enum_mode_to_str() -> None:
    # API 路径下 mode 是 pydantic 枚举；normalize 必须规整为字符串 "oco"，
    # 否则 build_dataset 的 str(mode)=="oco" 判定会失效。
    from aitrade.models.alpha import LabelMode

    spec = _normalize_label_spec(
        {"mode": LabelMode.OCO, "take_profit": 0.03, "stop_loss": 0.02, "max_hold": 5}
    )
    assert spec["mode"] == "oco"
    assert str(spec["mode"]) == "oco"
