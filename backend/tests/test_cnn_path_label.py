"""
路径多分类标签（path_class）验收测试。

测试对象：
  aitrade.cnn.dataset._oco_label_value（objective="path_class" 分支）
  aitrade.cnn.dataset.build_dataset（objective="path_class" 入口校验）

覆盖：
  2.3 示例测试：六个分支逐一断言（tp/sl/time涨/time跌/dead-zone-drop/dead-zone-negative）
  2.4 Property 1：路径标签与 OCO 判定一致（Hypothesis，max_examples=100）
  2.5 Property 6：非法组合拦截（objective=path_class 且 mode≠oco 抛 ValueError）
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.cnn.dataset import (
    PATH_SL_FIRST,
    PATH_TIME_DOWN,
    PATH_TIME_UP,
    PATH_TP_FIRST,
    _oco_label_value,
    build_dataset,
)
from aitrade.backtest.oco import simulate_oco_exit


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _flat(values: list[float]) -> np.ndarray:
    """把浮点列表转为 float64 numpy 数组。"""
    return np.asarray(values, dtype=np.float64)


# 公共 OCO 配置；序列长度 6（索引 0..5），anchor=0 → entry_index=1
_SPEC = {
    "mode": "oco",
    "take_profit": 0.05,
    "stop_loss": 0.05,
    "max_hold": 3,
    "stop_first": True,
}


# ---------------------------------------------------------------------------
# 2.3 示例测试：六个分支
# ---------------------------------------------------------------------------
class TestPathClassExamples:
    """path_class 六个出场分支的确定性验证。"""

    def test_tp_first(self) -> None:
        """止盈触发 → PATH_TP_FIRST（0.0）。"""
        opens = _flat([100, 100, 100, 100, 100, 100])
        highs = _flat([100, 100, 106, 100, 100, 100])  # j=2 触止盈价 105
        lows  = _flat([100, 100, 100, 100, 100, 100])
        status, label, ret = _oco_label_value(
            0, opens, highs, lows, _SPEC, "path_class", 0.0, "drop"
        )
        assert status == "ok"
        assert label == PATH_TP_FIRST
        assert ret == pytest.approx(0.05)

    def test_sl_first(self) -> None:
        """止损触发 → PATH_SL_FIRST（1.0）。"""
        opens = _flat([100, 100, 100, 100, 100, 100])
        highs = _flat([100, 100, 100, 100, 100, 100])
        lows  = _flat([100, 100, 94, 100, 100, 100])  # j=2 触止损价 95
        status, label, ret = _oco_label_value(
            0, opens, highs, lows, _SPEC, "path_class", 0.0, "drop"
        )
        assert status == "ok"
        assert label == PATH_SL_FIRST
        assert ret == pytest.approx(-0.05)

    def test_time_up(self) -> None:
        """时间止损 + 涨超阈值 → PATH_TIME_UP（2.0）。"""
        opens = _flat([100, 100, 100, 100, 100, 103])
        highs = _flat([100, 100, 104, 104, 104, 103])  # 不到止盈价 105
        lows  = _flat([100, 100, 96,  96,  96,  103])  # 不到止损价 95
        status, label, ret = _oco_label_value(
            0, opens, highs, lows, _SPEC, "path_class", 0.01, "drop"
        )
        assert status == "ok"
        assert label == PATH_TIME_UP
        assert ret == pytest.approx(0.03)

    def test_time_down(self) -> None:
        """时间止损 + 跌超阈值 → PATH_TIME_DOWN（3.0）。"""
        opens = _flat([100, 100, 100, 100, 100, 97])
        highs = _flat([100, 100, 104, 104, 104, 97])
        lows  = _flat([100, 100, 96,  96,  96,  97])
        status, label, ret = _oco_label_value(
            0, opens, highs, lows, _SPEC, "path_class", 0.01, "drop"
        )
        assert status == "ok"
        assert label == PATH_TIME_DOWN
        assert ret == pytest.approx(-0.03)

    def test_dead_zone_drop(self) -> None:
        """时间止损 + |ret| <= threshold（dead-zone），neutral_policy=drop → "neutral"。"""
        # entry_price=100, 兜底 open=100.2，ret=0.002 < threshold=0.01 → dead-zone
        opens = _flat([100, 100, 100, 100, 100, 100.2])
        highs = _flat([100, 100, 104, 104, 104, 100.2])
        lows  = _flat([100, 100, 96,  96,  96,  100.2])
        status, label, ret = _oco_label_value(
            0, opens, highs, lows, _SPEC, "path_class", 0.01, "drop"
        )
        assert status == "neutral"
        assert label is None

    def test_dead_zone_negative(self) -> None:
        """dead-zone + neutral_policy=negative → PATH_TIME_DOWN（3.0）。"""
        opens = _flat([100, 100, 100, 100, 100, 100.2])
        highs = _flat([100, 100, 104, 104, 104, 100.2])
        lows  = _flat([100, 100, 96,  96,  96,  100.2])
        status, label, ret = _oco_label_value(
            0, opens, highs, lows, _SPEC, "path_class", 0.01, "negative"
        )
        assert status == "ok"
        assert label == PATH_TIME_DOWN

    def test_out_of_bounds_skip(self) -> None:
        """序列不足以形成完整路径 → skip。"""
        opens = _flat([100, 100, 100, 100, 100, 100])
        highs = _flat([100, 100, 106, 100, 100, 100])
        lows  = _flat([100, 100, 100, 100, 100, 100])
        # anchor=1 → fallback=1+3+2=6 >= 6 → skip
        status, label, ret = _oco_label_value(
            1, opens, highs, lows, _SPEC, "path_class", 0.0, "drop"
        )
        assert status == "skip"
        assert label is None
        assert ret is None


# ---------------------------------------------------------------------------
# 2.4 Property 1：路径标签与 OCO 判定一致
# ---------------------------------------------------------------------------

# OHLC 合成策略：
#   - anchor=0, entry_index=1
#   - 序列长度 >= max_hold+3（保证 fallback 不越界）
#   - 每根 bar：low <= min(open,close)，high >= max(open,close)，所有价格 > 0


@st.composite
def ohlc_path_strategy(draw):
    """生成合法 OHLC 序列及 OCO 参数（供 Property 1 使用）。

    Returns:
        (opens, highs, lows, spec, threshold, neutral_policy)：
        - opens/highs/lows 是 float64 numpy 数组，长度 >= max_hold+3。
        - spec 是规整后的 OCO label_spec 字典。
        - threshold ∈ [0, 0.01]。
        - neutral_policy ∈ {"drop", "negative"}。
    """
    take_profit = draw(st.floats(min_value=0.005, max_value=0.2))
    stop_loss = draw(st.floats(min_value=0.005, max_value=0.2))
    max_hold = draw(st.integers(min_value=1, max_value=10))
    threshold = draw(st.floats(min_value=0.0, max_value=0.01))
    neutral_policy = draw(st.sampled_from(["drop", "negative"]))

    n = max_hold + 3  # 保证 fallback_index = 1 + max_hold + 1 = max_hold+2 < n

    base = draw(st.floats(min_value=10.0, max_value=1000.0))
    opens_vals = []
    highs_vals = []
    lows_vals = []
    for _ in range(n):
        o = draw(st.floats(min_value=base * 0.5, max_value=base * 1.5))
        c = draw(st.floats(min_value=base * 0.5, max_value=base * 1.5))
        hi = draw(st.floats(min_value=max(o, c), max_value=max(o, c) * 1.1 + 0.01))
        lo = draw(st.floats(min_value=min(o, c) * 0.9, max_value=min(o, c)))
        # 保证正数
        o = max(o, 0.01)
        c = max(c, 0.01)
        hi = max(hi, max(o, c))
        lo = max(min(lo, min(o, c)), 0.01)
        opens_vals.append(o)
        highs_vals.append(hi)
        lows_vals.append(lo)

    spec = {
        "mode": "oco",
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "max_hold": max_hold,
        "stop_first": True,
    }
    return (
        np.array(opens_vals, dtype=np.float64),
        np.array(highs_vals, dtype=np.float64),
        np.array(lows_vals, dtype=np.float64),
        spec,
        threshold,
        neutral_policy,
    )


# Feature: cnn-path-multiclass-head, Property 1:
# 对任意合法 OHLC 路径与合法 OCO 参数，_oco_label_value(objective="path_class") 产出的类别
# 与对同一输入直接调用 simulate_oco_exit 的结果满足映射；dead-zone 按 neutral_policy；越界→skip。
@given(ohlc_path_strategy())
@settings(max_examples=100)
def test_property1_path_class_consistent_with_simulate_oco_exit(params) -> None:
    """Property 1：path_class 标签与 simulate_oco_exit 结果逐字段对齐。"""
    opens, highs, lows, spec, threshold, neutral_policy = params

    status, label, ret = _oco_label_value(
        0, opens, highs, lows, spec, "path_class", threshold, neutral_policy
    )

    # 直接调用 simulate_oco_exit 得到"真值"
    entry_index = 1
    entry_price = float(opens[entry_index])
    if entry_price <= 0:
        assert status == "skip"
        return

    result = simulate_oco_exit(
        entry_index,
        entry_price,
        opens,
        highs,
        lows,
        float(spec["take_profit"]),
        float(spec["stop_loss"]),
        int(spec["max_hold"]),
        stop_first=True,
    )

    if result is None:
        # 越界 → _oco_label_value 也应 skip
        assert status == "skip"
        return

    sim_reason = str(result["reason"])
    sim_ret = float(result["ret"])

    if sim_reason == "tp":
        assert status == "ok"
        assert label == PATH_TP_FIRST
    elif sim_reason == "sl":
        assert status == "ok"
        assert label == PATH_SL_FIRST
    else:
        # reason == "time"
        assert ret == pytest.approx(sim_ret, abs=1e-9)
        if sim_ret > threshold:
            assert status == "ok"
            assert label == PATH_TIME_UP
        elif sim_ret < -threshold:
            assert status == "ok"
            assert label == PATH_TIME_DOWN
        else:
            # dead-zone
            if neutral_policy == "negative":
                assert status == "ok"
                assert label == PATH_TIME_DOWN
            else:
                assert status == "neutral"
                assert label is None


# ---------------------------------------------------------------------------
# 2.5 Property 6：非法组合拦截（path_class + mode≠oco → ValueError）
# ---------------------------------------------------------------------------

# Feature: cnn-path-multiclass-head, Property 6:
# 对任意 label_spec.mode ≠ "oco" 的配置，objective="path_class" 的 build_dataset 调用抛 ValueError。
@given(
    st.sampled_from(["next_bar", "horizon_bars", "session_close", "next_session_close"])
)
@settings(max_examples=20)
def test_property6_path_class_requires_oco_mode(mode: str) -> None:
    """Property 6：path_class + 非 oco 模式 → ValueError（在数据加载之前触发）。"""
    label_spec: dict = {"mode": mode}
    if mode == "horizon_bars":
        label_spec["horizon"] = 5

    with pytest.raises(ValueError, match="path_class"):
        build_dataset(
            vt_symbols=["FAKE.SSE"],
            start=__import__("datetime").date(2024, 1, 1),
            end=__import__("datetime").date(2024, 3, 31),
            lookback=10,
            label_spec=label_spec,
            objective="path_class",
        )
