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

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ..live.decision_instant import INTRADAY_BAR_FREQS, SUPPORTED_BAR_FREQS
from .live import PortfolioSnapshotRequest, RiskConfigRequest

# 与 notifier_channels.SUPPORTED_CHANNELS 对齐的合法通道名。
NotifyChannel = Literal["dingtalk", "wecom", "serverchan", "webhook"]

_DEFAULT_TRIGGER_TIME = "15:05"


def _validate_hhmm(v: str) -> str:
    """校验并返回合法的 HH:MM 时点字符串，非法则抛 ValueError。"""
    parts = v.split(":")
    if len(parts) != 2:
        raise ValueError(f"触发时刻必须为合法 HH:MM: {v!r}")
    hh, mm = parts
    if not (hh.isdigit() and mm.isdigit() and 0 <= int(hh) < 24 and 0 <= int(mm) < 60):
        raise ValueError(f"触发时刻必须为合法 HH:MM: {v!r}")
    return v


class TradingPlanRequest(BaseModel):
    """创建/更新交易计划的请求体。"""

    name: str = Field(min_length=1, description="计划名称（必填）")
    model: str = Field(min_length=1, description="CNN 模型名（必填）")
    vt_symbol: str = Field(min_length=1, description="目标标的（必填）")
    scheme: str = Field(min_length=1, description="方案名（必填）")
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

    @field_validator("bar_freq")
    @classmethod
    def _valid_bar_freq(cls, v: str) -> str:
        if v not in SUPPORTED_BAR_FREQS:
            raise ValueError(f"bar_freq 仅支持 {SUPPORTED_BAR_FREQS}：{v!r}")
        return v

    @field_validator("trigger_times")
    @classmethod
    def _valid_trigger_times(cls, v: list[str]) -> list[str]:
        for t in v:
            _validate_hhmm(t)
        return v

    @model_validator(mode="after")
    def _normalize_trigger_times_by_freq(self) -> "TradingPlanRequest":
        """日频计划必须有唤醒时刻；日内计划（监控模式）按 Bar_Grid 调度，归一化为空列表。"""
        if self.bar_freq in INTRADAY_BAR_FREQS:
            self.trigger_times = []
        elif not self.trigger_times:
            raise ValueError("日频计划 trigger_times 至少需要一个唤醒时刻")
        return self


class TradingPlanSummary(BaseModel):
    """计划列表项摘要。"""

    plan_id: str
    name: str
    vt_symbol: str
    scheme: str
    bar_freq: str
    trigger_times: list[str] = Field(default_factory=list)  # 唤醒时刻集合（去重升序）
    enabled: bool
    last_triggered: Optional[str] = None  # YYYY-MM-DD，来自 Last_Triggered_Map（取 date）


class SchedulerStatus(BaseModel):
    """调度器运行状态。"""

    running: bool
    tick_seconds: float
    enabled_plan_count: int
    last_triggered: dict[str, str] = Field(default_factory=dict)  # {plan_id: "YYYY-MM-DD"}
