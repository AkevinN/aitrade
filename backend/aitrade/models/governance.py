"""
CNN 模型治理请求/响应模型。

定义 Walk-Forward 评估、候选训练、晋级/回滚、治理回放等治理流程的 Pydantic v2 模型。
"""

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
    """CNN 治理模块配置（全局开关与默认参数）。

    控制 Walk-Forward 评估周期、训练/测试窗口大小、随机种子数量等治理行为；
    ``auto_promote=False`` 要求人工确认晋级，适合第一版保守运营。
    """

    enabled: bool = Field(default=True, description="是否启用治理模块")
    evaluation_period_days: int = Field(default=30, ge=1, description="默认评估周期（天）")
    train_days: int = Field(default=720, ge=30, description="默认训练窗口（天）")
    test_days: int = Field(default=90, ge=1, description="默认 OOS 测试窗口（天）")
    n_seeds: int = Field(default=1, ge=1, le=10, description="默认随机种子数量")
    next_suggested_eval_date: Optional[date] = Field(default=None, description="下一次建议评估日期")
    auto_promote: bool = Field(default=False, description="是否自动晋级；第一版固定建议 false")


class CNNTrainingParams(BaseModel):
    """CNN 治理训练超参数（Walk-Forward 与候选训练共用）。

    与单次训练接口的参数结构对齐；各字段均带范围约束，防止异常值导致 OOM 或发散。
    """

    epochs: int = Field(default=20, ge=1, le=500)
    batch_size: int = Field(default=32, ge=1, le=4096)
    learning_rate: float = Field(default=0.001, gt=0)
    lookback: int = Field(default=30, ge=2, le=4096)
    dropout: float = Field(default=0.4, ge=0, lt=1)
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    # 损失加权策略：none=普通 BCE，各样本权重相同；
    # magnitude=按 |未来收益| 对样本加权，让大波动样本主导梯度（仅分类目标生效）。
    loss_weighting: Literal["none", "magnitude"] = "none"


class CNNBacktestParams(BaseModel):
    """CNN 治理回测参数（阈值、成本、出场规则）。

    控制信号阈值（buy_threshold / sell_threshold）、A 股交易成本（佣金/印花税/滑点）
    以及出场模式（threshold / fixed_hold / oco / auto）；T+1 默认开启贴近 A 股现实。

    ``veto_threshold`` 仅在 objective='path_class' 时生效：当「先触止损」类概率
    prob_sl >= 该值时否决买入；默认 1.0 等效关闭（向后兼容）。
    """

    buy_threshold: float = 0.6
    sell_threshold: float = 0.4
    commission_rate: float = Field(default=0.0003, ge=0, lt=0.1)
    stamp_duty: float = Field(default=0.0005, ge=0, lt=0.1)
    slippage: float = Field(default=0.0005, ge=0, lt=0.1)
    price_add: float = Field(default=0.002, ge=0, lt=0.1)
    exit_mode: Literal["threshold", "fixed_hold", "oco", "auto"] = "auto"
    hold_days: int = Field(default=1, ge=1, le=60)
    take_profit: float = Field(default=0.0, ge=0, lt=1)
    stop_loss: float = Field(default=0.0, ge=0, lt=1)
    t_plus1: bool = False
    veto_threshold: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "path_class 专用：先触止损概率 prob_sl >= 该值时否决买入；"
            "默认 1.0 等效关闭否决（向后兼容）。"
        ),
    )


class CNNPromotionGate(BaseModel):
    """候选模型晋级门禁（Walk-Forward 折内胜出条件）。

    当候选模型的 min_win_rate 折数胜出、核心指标提升 >= min_core_score_delta
    且最大回撤劣化 <= max_drawdown_worsen_pct 时，才允许晋级（或提示运营人工确认）。
    """

    min_win_rate: float = Field(default=0.5, ge=0, le=1, description="候选胜出折数比例下限")
    min_core_score_delta: float = Field(default=0.0, description="核心分数相对生产模型的平均提升下限")
    max_drawdown_worsen_pct: float = Field(default=10.0, ge=0, description="最大回撤允许劣化百分比")
    require_positive_oos: bool = Field(default=True, description="是否要求 OOS 核心指标为正")


class CNNWalkForwardRequest(BaseModel):
    """启动 CNN Walk-Forward / OOS 评估的请求体。

    分 train_days 训练窗 + test_days OOS 测试窗逐步前推；production_model 为空时
    自动读取当前生产模型作为相对胜出基准。step_days 为空时默认等于 test_days（无重叠滚动）。
    """

    name: str = Field(description="评估名称")
    target_symbol: str
    input_data_kind: str = "bar"
    input_interval: str = "d"
    start: date
    end: date
    train_days: int = Field(default=720, ge=30)
    test_days: int = Field(default=90, ge=1)
    step_days: Optional[int] = Field(default=None, ge=1)
    n_seeds: int = Field(
        default=1,
        ge=1,
        le=10,
        description="每折重复训练的随机种子数；1=单种子，>1 时折内对多个种子取均值并衡量跨种子波动",
    )
    objective: Literal["classification", "regression", "path_class"] = Field(
        default="classification",
        description=(
            "预测目标：classification=方向二分类；regression=涨跌幅回归；"
            "path_class=路径形态四分类（先触止盈/先触止损/到期小涨/到期小跌，需配合 label_spec.mode='oco'）"
        ),
    )
    label_spec: LabelSpec = Field(default_factory=LabelSpec)
    observation_groups: list[ObservationGroup] = Field(default_factory=list)
    training_params: CNNTrainingParams = Field(default_factory=CNNTrainingParams)
    backtest_params: CNNBacktestParams = Field(default_factory=CNNBacktestParams)
    promotion_gate: CNNPromotionGate = Field(default_factory=CNNPromotionGate)
    production_model: Optional[str] = Field(default=None, description="用于相对胜出的生产模型；为空则读取当前生产模型")


class CNNCandidateTrainRequest(CNNWalkForwardRequest):
    """在全量（或指定）窗口上训练候选模型的请求体。

    继承 ``CNNWalkForwardRequest`` 的所有评估配置；额外可指定 final_train_start /
    final_train_end 以覆盖最终训练窗，不指定时沿用请求 start 或最后一个训练窗结束。
    """

    final_train_start: Optional[date] = Field(default=None, description="最终候选训练起始日期；为空用请求 start")
    final_train_end: Optional[date] = Field(default=None, description="最终候选训练结束日期；为空用最后一个训练窗口结束")


class CNNPromotionRequest(BaseModel):
    """将候选模型晋级为生产模型的请求体（人工确认用）。"""

    promoted_by: str = Field(default="manual")
    note: str = ""


class CNNRollbackRequest(BaseModel):
    """将生产模型回滚到指定版本（或上一生产模型）的请求体。"""

    rollback_to: Optional[str] = Field(default=None, description="指定回滚模型；为空回滚上一生产模型")
    requested_by: str = Field(default="manual")
    note: str = ""


class CNNGovernanceReplayRequest(BaseModel):
    """治理回放回测请求：模拟历史上若使用治理流程会有何收益/风险。

    以 initial_train_days 初始训练窗启动，每 evaluation_period_days 评估一次，
    并与 baselines（固定初始模型/永远重训/治理晋级/买入持有）做横向对比。
    """

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
    objective: Literal["classification", "regression", "path_class"] = Field(
        default="classification",
        description=(
            "预测目标：classification=方向二分类；regression=涨跌幅回归；"
            "path_class=路径形态四分类（先触止盈/先触止损/到期小涨/到期小跌，需配合 label_spec.mode='oco'）"
        ),
    )
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
    """当前生产模型状态（持久化的晋级记录快照）。

    promoted_at / promoted_by 记录最近一次晋级操作的时刻与操作者；
    previous_model_name / previous_model_version 保留上一版生产模型信息（回滚用）。
    """

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
    """治理历史事件记录（审计日志条目）。

    每次晋级/回滚/评估触发时追加到历史序列；payload 含操作参数与判定结果，
    供治理回放与合规审计使用。
    """

    ts: datetime = Field(default_factory=datetime.now)
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)

