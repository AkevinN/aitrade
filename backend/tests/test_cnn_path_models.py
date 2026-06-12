"""CNN path_class 模型层扩展测试。

测试对象：
- ``aitrade.models.alpha.CNNTrainRequest``：objective 字段三值校验（classification / regression / path_class）
- ``aitrade.models.alpha.CNNBacktestRequest``：新增 veto_threshold 字段的边界校验
- ``aitrade.models.governance.CNNWalkForwardRequest``：objective 字段三值同步
- ``aitrade.models.governance.CNNGovernanceReplayRequest``：objective 字段三值同步

均为纯 Pydantic 校验单元测试，无 I/O，无 torch 依赖。
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from aitrade.models.alpha import CNNBacktestRequest, CNNTrainRequest
from aitrade.models.governance import (
    CNNGovernanceReplayRequest,
    CNNWalkForwardRequest,
)


# ---------------------------------------------------------------------------
# CNNTrainRequest.objective 三值校验
# ---------------------------------------------------------------------------


def test_cnn_train_request_objective_classification() -> None:
    """objective='classification' 默认值通过校验。"""
    req = CNNTrainRequest(
        name="m1",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        vt_symbols=["000001.SZSE"],
    )
    assert req.objective == "classification"


def test_cnn_train_request_objective_regression() -> None:
    """objective='regression' 显式赋值通过校验。"""
    req = CNNTrainRequest(
        name="m1",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        vt_symbols=["000001.SZSE"],
        objective="regression",
    )
    assert req.objective == "regression"


def test_cnn_train_request_objective_path_class() -> None:
    """objective='path_class' 新值通过校验。"""
    req = CNNTrainRequest(
        name="m1",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        vt_symbols=["000001.SZSE"],
        objective="path_class",
    )
    assert req.objective == "path_class"


def test_cnn_train_request_objective_invalid() -> None:
    """非法 objective 值被 Pydantic 拒绝。"""
    with pytest.raises(ValidationError):
        CNNTrainRequest(
            name="m1",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            vt_symbols=["000001.SZSE"],
            objective="unknown_mode",
        )


# ---------------------------------------------------------------------------
# CNNBacktestRequest.veto_threshold 字段校验
# ---------------------------------------------------------------------------


def test_cnn_backtest_request_veto_threshold_default() -> None:
    """veto_threshold 默认值为 1.0（向后兼容，等效关闭否决）。"""
    req = CNNBacktestRequest(
        name="bt1",
        model="cnn_model",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
    )
    assert req.veto_threshold == 1.0


def test_cnn_backtest_request_veto_threshold_valid_mid() -> None:
    """veto_threshold=0.7 在 (0, 1] 范围内通过校验。"""
    req = CNNBacktestRequest(
        name="bt1",
        model="cnn_model",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        veto_threshold=0.7,
    )
    assert req.veto_threshold == 0.7


def test_cnn_backtest_request_veto_threshold_boundary_one() -> None:
    """veto_threshold=1.0（上界，含）通过校验。"""
    req = CNNBacktestRequest(
        name="bt1",
        model="cnn_model",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        veto_threshold=1.0,
    )
    assert req.veto_threshold == 1.0


def test_cnn_backtest_request_veto_threshold_boundary_small() -> None:
    """veto_threshold=0.01（接近下界，但 > 0）通过校验。"""
    req = CNNBacktestRequest(
        name="bt1",
        model="cnn_model",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        veto_threshold=0.01,
    )
    assert req.veto_threshold == 0.01


def test_cnn_backtest_request_veto_threshold_zero_rejected() -> None:
    """veto_threshold=0 被 Pydantic 拒绝（gt=0，不含零）。"""
    with pytest.raises(ValidationError):
        CNNBacktestRequest(
            name="bt1",
            model="cnn_model",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            veto_threshold=0.0,
        )


def test_cnn_backtest_request_veto_threshold_above_one_rejected() -> None:
    """veto_threshold=1.5 超出上界被 Pydantic 拒绝。"""
    with pytest.raises(ValidationError):
        CNNBacktestRequest(
            name="bt1",
            model="cnn_model",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            veto_threshold=1.5,
        )


def test_cnn_backtest_request_veto_threshold_negative_rejected() -> None:
    """veto_threshold=-0.1 负数被 Pydantic 拒绝。"""
    with pytest.raises(ValidationError):
        CNNBacktestRequest(
            name="bt1",
            model="cnn_model",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            veto_threshold=-0.1,
        )


# ---------------------------------------------------------------------------
# CNNWalkForwardRequest.objective 三值校验
# ---------------------------------------------------------------------------


def test_cnn_walkforward_request_objective_path_class() -> None:
    """CNNWalkForwardRequest 接受 objective='path_class'。"""
    req = CNNWalkForwardRequest(
        name="wf1",
        target_symbol="000001.SZSE",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        objective="path_class",
    )
    assert req.objective == "path_class"


def test_cnn_walkforward_request_objective_default() -> None:
    """CNNWalkForwardRequest objective 默认仍为 'classification'（向后兼容）。"""
    req = CNNWalkForwardRequest(
        name="wf1",
        target_symbol="000001.SZSE",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
    )
    assert req.objective == "classification"


def test_cnn_walkforward_request_objective_invalid() -> None:
    """非法 objective 值被 Pydantic 拒绝。"""
    with pytest.raises(ValidationError):
        CNNWalkForwardRequest(
            name="wf1",
            target_symbol="000001.SZSE",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            objective="bad_value",
        )


# ---------------------------------------------------------------------------
# CNNGovernanceReplayRequest.objective 三值校验
# ---------------------------------------------------------------------------


def test_cnn_governance_replay_request_objective_path_class() -> None:
    """CNNGovernanceReplayRequest 接受 objective='path_class'。"""
    req = CNNGovernanceReplayRequest(
        name="replay1",
        target_symbol="000001.SZSE",
        start=date(2023, 1, 1),
        end=date(2024, 12, 31),
        objective="path_class",
    )
    assert req.objective == "path_class"


def test_cnn_governance_replay_request_objective_invalid() -> None:
    """非法 objective 值被 Pydantic 拒绝（CNNGovernanceReplayRequest）。"""
    with pytest.raises(ValidationError):
        CNNGovernanceReplayRequest(
            name="replay1",
            target_symbol="000001.SZSE",
            start=date(2023, 1, 1),
            end=date(2024, 12, 31),
            objective="bad_value",
        )
