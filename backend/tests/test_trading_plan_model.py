"""TradingPlanRequest 校验器单元测试（决策时刻统一：bar_freq / trigger_times）。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aitrade.models.trading_plan import TradingPlanRequest


def _base(**over) -> dict:
    data = {
        "name": "计划",
        "model": "m1",
        "vt_symbol": "000001.SZSE",
        "scheme": "s1",
        "portfolio": {"portfolio_value": 1000000},
    }
    data.update(over)
    return data


def test_valid_request_defaults() -> None:
    req = TradingPlanRequest(**_base())
    assert req.bar_freq == "1d"
    assert req.trigger_times == ["15:05"]


def test_valid_request_multi_trigger_times() -> None:
    req = TradingPlanRequest(**_base(trigger_times=["09:35", "15:05"]))
    assert req.trigger_times == ["09:35", "15:05"]


def test_invalid_trigger_time() -> None:
    with pytest.raises(ValidationError):
        TradingPlanRequest(**_base(trigger_times=["25:99"]))
    with pytest.raises(ValidationError):
        TradingPlanRequest(**_base(trigger_times=["abc"]))


def test_empty_trigger_times_rejected() -> None:
    with pytest.raises(ValidationError):
        TradingPlanRequest(**_base(trigger_times=[]))


def test_unsupported_bar_freq_rejected() -> None:
    # 仅支持 SUPPORTED_BAR_FREQS（1d + 分钟频）；其它值拒绝。
    with pytest.raises(ValidationError):
        TradingPlanRequest(**_base(bar_freq="2h"))


def test_intraday_bar_freq_normalizes_trigger_times_empty() -> None:
    # 日内计划（监控模式）按 Bar_Grid 自动调度：trigger_times 归一化为空列表。
    req = TradingPlanRequest(**_base(bar_freq="30m", trigger_times=["15:05"]))
    assert req.bar_freq == "30m"
    assert req.trigger_times == []


def test_daily_plan_requires_trigger_times() -> None:
    # 日频计划必须至少一个唤醒时刻。
    with pytest.raises(ValidationError):
        TradingPlanRequest(**_base(bar_freq="1d", trigger_times=[]))


def test_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        TradingPlanRequest(name="", model="m", vt_symbol="x", scheme="s",
                           portfolio={"portfolio_value": 1})
