"""
交易计划自动化（Trading Plan Automation / 盘中监控决策）请求/响应模型。

把前端计划表单映射为持久化的 TradingPlan，并提供调度状态响应模型。
复用既有 `models/live.py` 的 `PortfolioSnapshotRequest` / `RiskConfigRequest`。

校验红线：
- `bar_freq` 支持日频与分钟频（`SUPPORTED_BAR_FREQS`，单一事实来源在
  `live/decision_instant.py`）；与所选模型训练间隔的一致性（间隔锁定）在 API 层校验。
- 日频计划 `trigger_times` 每项必须为合法 HH:MM 且非空；日内计划（监控模式）按
  Bar_Grid 自动调度，`trigger_times` 归一化为空列表。
- `notify_channels` 仅含受支持的通道名（不含任何凭证）。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..live.decision_instant import INTRADAY_BAR_FREQS, SUPPORTED_BAR_FREQS
from .live import PortfolioSnapshotRequest, RiskConfigRequest

# Phase 3 M2：trigger_schedule 合法枚举值（单一事实来源）。
_TRIGGER_SCHEDULE_VALUES = ("daily", "weekly_first", "monthly_first")

# 与 notifier_channels.SUPPORTED_CHANNELS 对齐的合法通道名。
NotifyChannel = Literal["dingtalk", "wecom", "serverchan", "webhook"]

_DEFAULT_TRIGGER_TIME = "15:05"


def _validate_hhmm(v: str) -> str:
    """校验并返回合法的 HH:MM 时点字符串，非法则抛 ValueError。

    Args:
        v: 待校验的时刻字符串，预期格式 "HH:MM"（如 "15:05"）。

    Returns:
        原样返回 v（校验通过时）。

    Raises:
        ValueError: 格式不符或 HH/MM 超出范围时抛出。

    Example:
        >>> _validate_hhmm("15:05")
        '15:05'
        >>> _validate_hhmm("25:00")  # 抛 ValueError
    """
    parts = v.split(":")
    if len(parts) != 2:
        raise ValueError(f"触发时刻必须为合法 HH:MM: {v!r}")
    hh, mm = parts
    if not (hh.isdigit() and mm.isdigit() and 0 <= int(hh) < 24 and 0 <= int(mm) < 60):
        raise ValueError(f"触发时刻必须为合法 HH:MM: {v!r}")
    return v


class TradingPlanRequest(BaseModel):
    """创建/更新交易计划的请求体。

    支持两种策略类型（strategy_type）：
    - "cnn"：单标的 CNN 推理决策，model / vt_symbol / scheme 必填；
    - "rule"：规则信号驱动的组合调仓，signal_source 必填，model/vt_symbol/scheme 允许空串。

    bar_freq 须在 SUPPORTED_BAR_FREQS 中，日频计划须至少有一个 trigger_times 时点；
    日内计划（INTRADAY_BAR_FREQS）trigger_times 归一化为空列表（按 Bar_Grid 自动调度）。
    trigger_schedule 控制调度粒度（daily / weekly_first / monthly_first）。
    """

    name: str = Field(min_length=1, description="计划名称（必填）")
    model: str = Field(default="", description="CNN 模型名（cnn 计划必填，rule 计划允许空串）")
    vt_symbol: str = Field(default="", description="目标标的（cnn 计划必填，rule 计划允许空串）")
    scheme: str = Field(default="", description="方案名（cnn 计划必填，rule 计划允许空串）")
    buy_threshold: float = Field(default=0.6, description="买入阈值")
    position_ratio: float = Field(default=0.95, description="目标仓位比例")
    min_volume: int = Field(default=100, description="最小成交手数")
    model_version: str = Field(default="", description="模型版本，参与 signal_id")
    data_source: Literal["upload", "pull"] = Field(default="pull", description="数据源")
    should_exit: bool = Field(default=False, description="是否触发出场")
    halted: bool = Field(default=False, description="目标标的当日是否停牌/封死")
    portfolio: PortfolioSnapshotRequest
    risk: RiskConfigRequest = Field(default_factory=RiskConfigRequest)
    enabled: bool = Field(default=False, description="是否启用自动调度")
    bar_freq: str = Field(
        default="1d",
        description="决策 bar 频率（须与所选模型训练间隔一致）；1d=日频，分钟频=盘中监控模式",
    )
    trigger_times: list[str] = Field(
        default_factory=lambda: [_DEFAULT_TRIGGER_TIME],
        description="日频计划的调度唤醒时刻 HH:MM 列表；日内计划按 Bar_Grid 自动调度（本字段为空）",
    )
    notify_channels: list[NotifyChannel] = Field(
        default_factory=list, description="通知通道名（仅名称，无凭证）"
    )
    # —— Phase 3：策略类型与调度粒度 ——
    strategy_type: Literal["cnn", "rule"] = Field(
        default="cnn", description="计划策略类型：cnn（默认，CNN模型决策）| rule（规则信号决策）"
    )
    signal_source: str = Field(
        default="", description="rule 计划的注册表信号源名（如 'etf_momentum'）；strategy_type='rule' 时必填非空"
    )
    signal_params: dict[str, Any] = Field(
        default_factory=dict, description="规则信号参数（透传给信号源）"
    )
    trigger_schedule: str = Field(
        default="daily",
        description="调度粒度：daily（每交易日）| weekly_first（每周第一个交易日）| monthly_first（每月第一个交易日）",
    )
    portfolio_id: str = Field(
        default="", description="rule 计划关联的持仓账本 id（Phase 3 后续任务消费）"
    )

    @field_validator("trigger_schedule")
    @classmethod
    def _valid_trigger_schedule(cls, v: str) -> str:
        """校验 trigger_schedule 为合法枚举值（daily / weekly_first / monthly_first）。"""
        if v not in _TRIGGER_SCHEDULE_VALUES:
            raise ValueError(f"trigger_schedule 仅支持 {_TRIGGER_SCHEDULE_VALUES}：{v!r}")
        return v

    @field_validator("bar_freq")
    @classmethod
    def _valid_bar_freq(cls, v: str) -> str:
        """校验 bar_freq 为已支持的 bar 周期（SUPPORTED_BAR_FREQS 中的值）。"""
        if v not in SUPPORTED_BAR_FREQS:
            raise ValueError(f"bar_freq 仅支持 {SUPPORTED_BAR_FREQS}：{v!r}")
        return v

    @field_validator("trigger_times")
    @classmethod
    def _valid_trigger_times(cls, v: list[str]) -> list[str]:
        """逐项校验 trigger_times 中每个时刻均为合法 HH:MM 格式。"""
        for t in v:
            _validate_hhmm(t)
        return v

    @model_validator(mode="after")
    def _validate_cnn_required_fields(self) -> TradingPlanRequest:
        """strategy_type='cnn' 时 model/vt_symbol/scheme 必填非空，保持既有 422 语义；
        strategy_type='rule' 时允许空串（调仓信号由 signal_source 驱动）。"""
        if self.strategy_type == "cnn":
            if not self.model.strip():
                raise ValueError("strategy_type='cnn' 时 model 必填非空")
            if not self.vt_symbol.strip():
                raise ValueError("strategy_type='cnn' 时 vt_symbol 必填非空")
            if not self.scheme.strip():
                raise ValueError("strategy_type='cnn' 时 scheme 必填非空")
        return self

    @model_validator(mode="after")
    def _normalize_trigger_times_by_freq(self) -> TradingPlanRequest:
        """日频计划必须有唤醒时刻；日内计划（监控模式）按 Bar_Grid 调度，归一化为空列表。
        rule 计划固定 bar_freq='1d'，trigger_times 驱动当日几点产出调仓建议（与 cnn 日频
        相同的 due_slots 路径，见 plan_scheduler._tick_daily_plan）。"""
        if self.bar_freq in INTRADAY_BAR_FREQS:
            self.trigger_times = []
        elif not self.trigger_times:
            raise ValueError("日频计划 trigger_times 至少需要一个唤醒时刻（rule 计划亦然：调度器以 trigger_times 确定当日几点产出调仓建议）")
        return self

    @model_validator(mode="after")
    def _validate_rule_requires_signal_source(self) -> TradingPlanRequest:
        """strategy_type='rule' 时 signal_source 必填非空（Phase 3 M2 校验）。"""
        if self.strategy_type == "rule" and not self.signal_source.strip():
            raise ValueError("strategy_type='rule' 时 signal_source 必填非空")
        return self


class TradingPlanSummary(BaseModel):
    """交易计划列表摘要（供前端计划表格展示）。

    ``last_triggered`` 归一化为日期字符串 "YYYY-MM-DD"（从 Last_Triggered_Map 取），
    兼容新旧状态形态；``strategy_type`` 供前端组合选择器过滤 cnn / rule 计划。
    """

    plan_id: str
    name: str
    vt_symbol: str
    scheme: str
    bar_freq: str
    trigger_times: list[str] = Field(default_factory=list)  # 唤醒时刻集合（去重升序）
    enabled: bool
    last_triggered: str | None = None  # YYYY-MM-DD，来自 Last_Triggered_Map（取 date）
    strategy_type: str = "cnn"  # 策略类型：cnn | rule（前端组合选择器过滤用）
    portfolio_id: str = ""  # rule 计划关联的持仓账本 id（组合选择器数据源）
    signal_source: str = ""  # rule 计划信号源名（前端展示用）


class SchedulerStatus(BaseModel):
    """进程内调度器运行状态（供 GET /api/live/scheduler/status 端点响应）。

    ``last_triggered`` 为 ``{plan_id: "YYYY-MM-DD"}`` 映射，从 RuntimeStateStore 读取；
    ``enabled_plan_count`` 为当前已启用计划数量（调度器不在线时从存储静态计算）。
    """

    running: bool
    tick_seconds: float
    enabled_plan_count: int
    last_triggered: dict[str, str] = Field(default_factory=dict)  # {plan_id: "YYYY-MM-DD"}
