"""backtest/t1.py 单一事实源谓词测试（T+1 工具化 P2）。

is_t1_locked 是回测撮合引擎与 CNN 策略共用的 T+1 判定谓词，必须严格满足：
仅当 (enabled 且 非豁免 且 今日已买入) 三条件同时成立才锁定。
"""

from __future__ import annotations

from datetime import date

from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.backtest.t1 import is_t1_locked

SYM = "600000.SSE"
D = date(2025, 1, 2)


def test_locked_when_enabled_not_exempt_bought_today():
    assert is_t1_locked(SYM, {SYM: D}, D, enabled=True, exempt=set()) is True


def test_not_locked_when_disabled():
    # enabled=False（T+0/关闭 T+1）恒不锁，即便今日买入。
    assert is_t1_locked(SYM, {SYM: D}, D, enabled=False, exempt=set()) is False


def test_not_locked_when_exempt():
    # 豁免品种（可转债 T+0）恒不锁，即便 enabled=True 且今日买入。
    assert is_t1_locked(SYM, {SYM: D}, D, enabled=True, exempt={SYM}) is False


def test_not_locked_when_bought_yesterday():
    assert is_t1_locked(SYM, {SYM: date(2025, 1, 1)}, D, enabled=True, exempt=set()) is False


def test_not_locked_when_never_bought():
    assert is_t1_locked(SYM, {}, D, enabled=True, exempt=set()) is False


def test_not_locked_when_today_is_none():
    # 引擎尚未推进到任何 bar（datetime=None）时视为不锁。
    assert is_t1_locked(SYM, {SYM: D}, None, enabled=True, exempt=set()) is False


@settings(max_examples=100)
@given(
    enabled=st.booleans(),
    exempt_it=st.booleans(),
    buy_offset=st.integers(min_value=-3, max_value=3),
)
def test_property_locked_iff_all_three_conditions(enabled: bool, exempt_it: bool, buy_offset: int):
    # Feature: t-plus-1-tooling, Property: 锁定 当且仅当 enabled 且 非豁免 且 今日买入
    today = date(2025, 6, 10)
    buy_dates = {SYM: date(2025, 6, 10 + buy_offset)}  # buy_offset==0 即"今日买入"
    exempt = {SYM} if exempt_it else set()

    result = is_t1_locked(SYM, buy_dates, today, enabled=enabled, exempt=exempt)
    expected = enabled and (not exempt_it) and (buy_dates[SYM] == today)
    assert result is expected


# ── P0：T+1 默认开启，三个子系统口径与规则策略对齐（守护违规 A）────────────────


def test_cnn_backtest_request_defaults_t_plus1_true():
    from aitrade.models.alpha import CNNBacktestRequest

    assert CNNBacktestRequest.model_fields["t_plus1"].default is True


def test_cnn_governance_params_defaults_t_plus1_true():
    from aitrade.models.governance import CNNBacktestParams

    assert CNNBacktestParams.model_fields["t_plus1"].default is True


def test_scheme_cost_config_defaults_t_plus1_true():
    from aitrade.backtest.scheme import CostConfig

    assert CostConfig.model_fields["t_plus1"].default is True


def test_all_backtest_paths_agree_with_rule_strategy_default():
    # 四个子系统的 T+1 默认值统一为 True，消除"CNN 默认 False vs 规则策略默认 True"的分叉。
    from aitrade.backtest.scheme import CostConfig
    from aitrade.models.alpha import CNNBacktestRequest
    from aitrade.models.governance import CNNBacktestParams
    from aitrade.models.strategy import StrategyCost

    defaults = {
        "CNNBacktestRequest": CNNBacktestRequest.model_fields["t_plus1"].default,
        "CNNBacktestParams": CNNBacktestParams.model_fields["t_plus1"].default,
        "CostConfig": CostConfig.model_fields["t_plus1"].default,
        "StrategyCost": StrategyCost.model_fields["t_plus1"].default,
    }
    assert all(v is True for v in defaults.values()), defaults
