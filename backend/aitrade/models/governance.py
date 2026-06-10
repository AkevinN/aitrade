"""CNN 模型治理请求/响应模型。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .alpha import LabelSpec, ObservationGroup


GovernanceBaseline = Literal[
    "fixed_initial_model",
    "always_retrain",
    "governed_promotion",
    "buy_and_hold",
]


class CNNGovernanceConfig(BaseModel):
    """CNN 治理模块配置。"""

    enabled: bool = Field(default=True, description="是否启用治理模块")
    evaluation_period_days: int = Field(default=30, ge=1, description="默认评估周期（天）")
    train_days: int = Field(default=720, ge=30, description="默认训练窗口（天）")
    test_days: int = Field(default=90, ge=1, description="默认 OOS 测试窗口（天）")
    n_seeds: int = Field(default=1, ge=1, le=10, description="默认随机种子数量")
    next_suggested_eval_date: Optional[date] = Field(default=None, description="下一次建议评估日期")
    auto_promote: bool = Field(default=False, description="是否自动晋级；第一版固定建议 false")


class CNNTrainingParams(BaseModel):
    """治理训练参数。"""

    epochs: int = Field(default=20, ge=1, le=500)
    batch_size: int = Field(default=32, ge=1, le=4096)
    learning_rate: float = Field(default=0.001, gt=0)
    lookback: int = Field(default=30, ge=2, le=4096)
    dropout: float = Field(default=0.4, ge=0, lt=1)
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    loss_weighting: Literal["none", "magnitude"] = "none"


class CNNBacktestParams(BaseModel):
    """治理回测参数。"""

    buy_threshold: float = 0.6
    sell_threshold: float = 0.4
    commission_rate: float = Field(default=0.0003, ge=0, lt=0.1)
    stamp_duty: float = Field(default=0.001, ge=0, lt=0.1)
    slippage: float = Field(default=0.0005, ge=0, lt=0.1)
    price_add: float = Field(default=0.002, ge=0, lt=0.1)
    exit_mode: Literal["threshold", "fixed_hold", "oco", "auto"] = "auto"
    hold_days: int = Field(default=1, ge=1, le=60)
    take_profit: float = Field(default=0.0, ge=0, lt=1)
    stop_loss: float = Field(default=0.0, ge=0, lt=1)
    t_plus1: bool = False


class CNNPromotionGate(BaseModel):
    """候选模型晋级门禁。"""

    min_win_rate: float = Field(default=0.5, ge=0, le=1, description="候选胜出折数比例下限")
    min_core_score_delta: float = Field(default=0.0, description="核心分数相对生产模型的平均提升下限")
    max_drawdown_worsen_pct: float = Field(default=10.0, ge=0, description="最大回撤允许劣化百分比")
    require_positive_oos: bool = Field(default=True, description="是否要求 OOS 核心指标为正")


class CNNWalkForwardRequest(BaseModel):
    """启动 CNN walk-forward/OOS 评估。"""

    name: str = Field(description="评估名称")
    target_symbol: str
    input_data_kind: str = "bar"
    input_interval: str = "d"
    start: date
    end: date
    train_days: int = Field(default=720, ge=30)
    test_days: int = Field(default=90, ge=1)
    step_days: Optional[int] = Field(default=None, ge=1)
    objective: Literal["classification", "regression"] = "classification"
    label_spec: LabelSpec = Field(default_factory=LabelSpec)
    observation_groups: list[ObservationGroup] = Field(default_factory=list)
    training_params: CNNTrainingParams = Field(default_factory=CNNTrainingParams)
    backtest_params: CNNBacktestParams = Field(default_factory=CNNBacktestParams)
    promotion_gate: CNNPromotionGate = Field(default_factory=CNNPromotionGate)
    production_model: Optional[str] = Field(default=None, description="用于相对胜出的生产模型；为空则读取当前生产模型")


class CNNCandidateTrainRequest(CNNWalkForwardRequest):
    """训练候选模型请求。"""

    final_train_start: Optional[date] = Field(default=None, description="最终候选训练起始日期；为空用请求 start")
    final_train_end: Optional[date] = Field(default=None, description="最终候选训练结束日期；为空用最后一个训练窗口结束")


class CNNPromotionRequest(BaseModel):
    """候选晋级请求。"""

    promoted_by: str = Field(default="manual")
    note: str = ""


class CNNRollbackRequest(BaseModel):
    """生产模型回滚请求。"""

    rollback_to: Optional[str] = Field(default=None, description="指定回滚模型；为空回滚上一生产模型")
    requested_by: str = Field(default="manual")
    note: str = ""


class CNNGovernanceReplayRequest(BaseModel):
    """治理回放回测请求。"""

    name: str
    target_symbol: str
    input_data_kind: str = "bar"
    input_interval: str = "d"
    start: date
    end: date
    initial_train_days: int = Field(default=720, ge=30)
    evaluation_period_days: int = Field(default=30, ge=1)
    test_period_days: int = Field(default=30, ge=1)
    capital: float = Field(default=1_000_000, gt=0)
    objective: Literal["classification", "regression"] = "classification"
    label_spec: LabelSpec = Field(default_factory=LabelSpec)
    observation_groups: list[ObservationGroup] = Field(default_factory=list)
    training_params: CNNTrainingParams = Field(default_factory=CNNTrainingParams)
    backtest_params: CNNBacktestParams = Field(default_factory=CNNBacktestParams)
    promotion_gate: CNNPromotionGate = Field(default_factory=CNNPromotionGate)
    baselines: list[GovernanceBaseline] = Field(
        default_factory=lambda: [
            "fixed_initial_model",
            "always_retrain",
            "governed_promotion",
            "buy_and_hold",
        ]
    )


class CNNProductionModel(BaseModel):
    """当前生产模型状态。"""

    model_name: str = ""
    model_version: str = ""
    target_symbol: str = ""
    input_interval: str = "d"
    objective: str = "classification"
    promoted_at: Optional[datetime] = None
    promoted_by: str = ""
    report_id: str = ""
    previous_model_name: str = ""
    previous_model_version: str = ""


class CNNGovernanceHistoryEvent(BaseModel):
    """治理历史事件。"""

    ts: datetime = Field(default_factory=datetime.now)
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)

