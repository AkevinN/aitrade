"""
规则策略回测/扫参/WalkForward 请求模型（Phase 2）。

风格对齐 models/alpha.py：pydantic v2，字段注释中文，类型用 X | None。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, model_validator


class StrategyCost(BaseModel):
    """回测成本与制度约束（A 股规则策略默认值）。

    commission_rate / stamp_duty / slippage 均为小数比例（如 0.0003 = 万三）；
    T+1 默认开启，复现 A 股当日买入不可当日卖出的交易规则。
    """

    commission_rate: float = Field(default=0.0003, description="单边佣金率")
    stamp_duty: float = Field(default=0.0005, description="卖出印花税率")
    slippage: float = Field(default=0.0005, description="每笔不利滑点率")
    t_plus1: bool = Field(default=True, description="T+1 卖出限制（规则策略默认开启，贴近 A 股现实）")


class StrategyBacktestRequest(BaseModel):
    """单次规则策略回测请求体。

    ``signal_source`` 对应注册表名（如 "etf_momentum"），由 ``api/strategy.py`` 路由
    解析后分派给相应信号源工厂；``strategy_params`` 透传给策略（如 rebalancing_topk 的 top_k）。
    日期合法性由 ``_check_date_order`` 校验（start 须严格早于 end）。
    """

    signal_source: str = Field(description="注册表名，如 'etf_momentum'")
    signal_params: dict[str, Any] = Field(default_factory=dict, description="透传给信号源工厂的参数")
    strategy_name: str = Field(default="rebalancing_topk", description="已注册的策略名")
    strategy_params: dict[str, Any] = Field(default_factory=dict, description="策略参数（top_k/n_drop/min_days 等）")
    interval: str = Field(default="d", description="K 线周期")
    start: date = Field(description="回测起始日（含）")
    end: date = Field(description="回测截止日（含）")
    capital: int = Field(default=1_000_000, description="初始资金")
    cost: StrategyCost = Field(default_factory=StrategyCost, description="成本与制度约束")

    @model_validator(mode="after")
    def _check_date_order(self) -> StrategyBacktestRequest:
        """校验 start 严格早于 end；否则抛 ValueError（Pydantic 会将其转为 422）。"""
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) 必须早于 end ({self.end})")
        return self


class StrategySweepRequest(StrategyBacktestRequest):
    """网格扫参请求：grid 中每项覆盖 strategy_params 和/或 signal_params。

    继承 ``StrategyBacktestRequest`` 全部字段；``grid`` 中每个 dict 作为 override
    与基础参数合并后运行独立回测，结果按顺序返回。grid 上限 50 项（防服务端任务泛洪）。
    """

    grid: list[dict[str, Any]] = Field(
        description="参数覆盖列表，每项含 strategy_params 和/或 signal_params 的 override，上限 50"
    )

    @model_validator(mode="after")
    def _check_grid_size(self) -> StrategySweepRequest:
        """校验 grid 不超过 50 项（防止服务端任务泛洪）。"""
        if len(self.grid) > 50:
            raise ValueError(f"grid 不能超过 50 项，当前 {len(self.grid)} 项")
        return self


class StrategyWalkForwardRequest(StrategyBacktestRequest):
    """Walk-Forward 验证请求（规则策略无训练，train 窗作参数观察期）。

    规则策略不涉及参数拟合，``train_days`` 作为参数稳定性观察期使用；
    ``test_days`` 为每折 OOS 测试窗口长度，折数由总时间范围自动推算。
    """

    train_days: int = Field(default=180, description="训练窗（参数观察期）天数")
    test_days: int = Field(default=60, description="测试窗天数；逐 test 窗跑回测")
