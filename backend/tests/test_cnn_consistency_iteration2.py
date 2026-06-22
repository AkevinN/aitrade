"""
迭代 2 验收测试：label ↔ 策略出场 一致性自检（守住「label.exit ≡ 策略出场」红线）。

覆盖：
1. label_holding_horizon：各 label 模式蕴含的固定持有期解析。
2. derive_strategy_exit_from_label：由 label 自动推导固定持有出场，不可推导时抛错。
3. check_label_strategy_consistency：
   - 一致（fixed_hold + 持有期匹配 + next_open）→ 无告警、不抛错；
   - hold_days 与 label 持有期不符 → 抛错（硬性不一致）；
   - price_ref=close → 软性告警（研究口径）；
   - threshold → 软性告警（信号衰减式，非固定持有）；
   - fixed_hold + 非固定持有 label（分钟 session_close）→ 抛错；
   - 未知 exit_mode → 抛错。
"""

from __future__ import annotations

import pytest

from aitrade.cnn.consistency import (
    check_label_strategy_consistency,
    derive_strategy_exit_from_label,
    label_holding_horizon,
)


# ---------------------------------------------------------------------------
# 1. label_holding_horizon
# ---------------------------------------------------------------------------
def test_holding_horizon_next_bar() -> None:
    assert label_holding_horizon({"mode": "next_bar"}) == 1


def test_holding_horizon_horizon_bars() -> None:
    assert label_holding_horizon({"mode": "horizon_bars", "horizon": 3}) == 3


def test_holding_horizon_next_session_close_depends_on_interval() -> None:
    assert label_holding_horizon({"mode": "next_session_close"}, "d") == 1
    # 分钟线下次日收盘跨度不固定 → None
    assert label_holding_horizon({"mode": "next_session_close"}, "1m") is None


def test_holding_horizon_session_close_is_none() -> None:
    assert label_holding_horizon({"mode": "session_close"}, "15m") is None


# ---------------------------------------------------------------------------
# 2. derive_strategy_exit_from_label
# ---------------------------------------------------------------------------
def test_derive_exit_from_next_bar() -> None:
    cfg = derive_strategy_exit_from_label({"mode": "next_bar"})
    assert cfg == {"exit_mode": "fixed_hold", "hold_days": 1}


def test_derive_exit_from_horizon_bars() -> None:
    cfg = derive_strategy_exit_from_label({"mode": "horizon_bars", "horizon": 5})
    assert cfg == {"exit_mode": "fixed_hold", "hold_days": 5}


def test_derive_exit_unfixed_horizon_raises() -> None:
    with pytest.raises(ValueError, match="无法由 label"):
        derive_strategy_exit_from_label({"mode": "session_close"}, "15m")


# ---------------------------------------------------------------------------
# 3. check_label_strategy_consistency
# ---------------------------------------------------------------------------
def test_consistency_aligned_no_warnings() -> None:
    warnings = check_label_strategy_consistency(
        {"mode": "next_bar", "price_ref": "next_open"},
        exit_mode="fixed_hold", hold_days=1, input_interval="d",
    )
    assert warnings == []


def test_consistency_horizon_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="label-trade 不一致"):
        check_label_strategy_consistency(
            {"mode": "horizon_bars", "horizon": 3, "price_ref": "next_open"},
            exit_mode="fixed_hold", hold_days=2, input_interval="d",
        )


def test_consistency_close_price_ref_warns() -> None:
    warnings = check_label_strategy_consistency(
        {"mode": "next_bar", "price_ref": "close"},
        exit_mode="fixed_hold", hold_days=1, input_interval="d",
    )
    assert len(warnings) == 1
    assert "研究口径" in warnings[0]


def test_consistency_threshold_warns_not_raises() -> None:
    warnings = check_label_strategy_consistency(
        {"mode": "next_bar", "price_ref": "next_open"},
        exit_mode="threshold", hold_days=1, input_interval="d",
    )
    assert len(warnings) == 1
    assert "threshold" in warnings[0]


def test_consistency_fixed_hold_on_unfixed_label_raises() -> None:
    with pytest.raises(ValueError, match="需要固定持有期 label"):
        check_label_strategy_consistency(
            {"mode": "session_close", "price_ref": "close"},
            exit_mode="fixed_hold", hold_days=1, input_interval="15m",
        )


def test_consistency_unknown_exit_mode_raises() -> None:
    with pytest.raises(ValueError, match="未知 exit_mode"):
        check_label_strategy_consistency(
            {"mode": "next_bar"}, exit_mode="weird", hold_days=1, input_interval="d",
        )


# ---------------------------------------------------------------------------
# 4. 分钟线下 label(bar) ↔ 策略(交易日) 单位换算（按天统一，P2）
# ---------------------------------------------------------------------------
def test_derive_horizon_bars_minute_converts_to_days() -> None:
    # 30m 每日 8 根：horizon=10 bar → hold_days = ceil(10/8) = 2 个交易日（非 10）。
    cfg = derive_strategy_exit_from_label({"mode": "horizon_bars", "horizon": 10}, "30m")
    assert cfg == {"exit_mode": "fixed_hold", "hold_days": 2}


def test_derive_oco_max_hold_minute_converts_to_days() -> None:
    # OCO max_hold=16 bar @30m → hold_days = ceil(16/8) = 2 个交易日。
    cfg = derive_strategy_exit_from_label(
        {"mode": "oco", "take_profit": 0.03, "stop_loss": 0.02, "max_hold": 16}, "30m"
    )
    assert cfg["exit_mode"] == "oco"
    assert cfg["hold_days"] == 2


def test_consistency_minute_compares_in_days_not_bars() -> None:
    # horizon=10 bar @30m = 2 个交易日：hold_days=2 通过，hold_days=10（旧 bar 口径）抛错。
    warnings = check_label_strategy_consistency(
        {"mode": "horizon_bars", "horizon": 10, "price_ref": "next_open"},
        exit_mode="fixed_hold", hold_days=2, input_interval="30m",
    )
    assert warnings == []

    with pytest.raises(ValueError, match="label-trade 不一致"):
        check_label_strategy_consistency(
            {"mode": "horizon_bars", "horizon": 10, "price_ref": "next_open"},
            exit_mode="fixed_hold", hold_days=10, input_interval="30m",
        )


def test_daily_behavior_unchanged_after_unit_fix() -> None:
    # 回归：日线下换算恒等，derive 与既有断言一致。
    assert derive_strategy_exit_from_label({"mode": "horizon_bars", "horizon": 5}) == {
        "exit_mode": "fixed_hold", "hold_days": 5,
    }
