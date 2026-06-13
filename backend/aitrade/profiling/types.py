"""
标的画像（Symbol Profiling）数据模型。

本文件集中定义画像产物的 pydantic 模型，严格对应 design.md 的
"Data Models" 与 "Components and Interfaces" 章节。所有模型仅作数据承载，
不含计算逻辑，便于序列化为 Profile_Artifact 并作为 API 的 SymbolProfile JSON 契约。

设计要点：
- MetricValue.value 允许为 None：当指标处于 Insufficient_Sample（样本不足）时
  抑制具体数值，避免输出看似精确却误导的结果（Requirement 7.2）。
- SchemeSuggestion.status 固定字面量 "draft"：建议恒为草稿，由人工确认后才可
  转为正式 Scheme，模块绝不自动写入 SCHEME_PATH（Requirement 8.4）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# 置信度等级取值集合：由有效样本量与该指标的统计有效性下限共同决定
ConfidenceLevel = Literal["high", "medium", "low", "insufficient"]


class MetricValue(BaseModel):
    """单项指标的取值与置信度。

    insufficient 时 value 为 None（抑制误导性数值），并通过 note 给出降级说明。
    """

    key: str  # 指标键，如 "gap_ratio"、"realized_volatility"
    # 指标值：标量或小结构；insufficient / 不适用时为 None（Requirement 7.2）
    value: float | dict | None = None
    effective_sample: int  # 参与该指标计算的有效样本量，用于置信度判定
    confidence: ConfidenceLevel  # 该指标的置信度等级
    note: str | None = None  # 不可靠 / 降级 / 不适用的文字说明


class MetricBlock(BaseModel):
    """一组同类指标（四个 Metric_Block 之一）。

    Attributes:
        block:   指标块类别，取数据质量 / 流动性 / 波动性 / 可预测性之一。
        metrics: 该块下的逐项指标列表；可为空（无可计算指标时）。
        level:   该块的等级判定（如流动性 高/中/低、波动等级）；无判定时为 None。
    """

    block: Literal["data_quality", "liquidity", "volatility", "predictability"]
    metrics: list[MetricValue] = Field(default_factory=list)
    level: str | None = None  # 该块的等级判定（如流动性 高/中/低、波动等级）


class GroupProfile(BaseModel):
    """多标的关联性画像，作为独立维度，不混入单标的 Metric_Block（Requirement 13）。"""

    target: str  # 目标标的
    members: list[str] = Field(default_factory=list)  # 参与对齐的观测标的
    alignment_coverage: float  # 目标与观测标的按公共时间轴对齐后的覆盖率
    correlation_summary: dict[str, float] = Field(default_factory=dict)  # 相关性概要


class SuggestionItem(BaseModel):
    """一条方案建议条目：被建议的字段、取值、理由与依据置信度。"""

    field: str  # Scheme 字段路径，如 "label_spec.tp"、"predictor.label_type"
    value: Any  # 建议取值
    reason: str  # Suggestion_Reason：命中的指标 / 规则 / 阈值（Requirement 8.2）
    based_on_confidence: str  # 该建议所依据的置信度等级（Requirement 7.3）


class SchemeSuggestion(BaseModel):
    """与 Scheme 结构兼容的方案建议草稿（只建议、不执行）。"""

    status: Literal["draft"] = "draft"  # 恒为草稿，待人工确认（Requirement 8.4）
    interval: str  # 与画像输入一致
    vt_symbols: list[str] = Field(default_factory=list)  # 与画像输入一致
    items: list[SuggestionItem] = Field(default_factory=list)  # 各建议条目
    degraded: bool = False  # 整体置信度不足时降级标记（Requirement 8.5）
    note: str | None = None  # 降级 / 风险 / 补充说明


class ProfileInput(BaseModel):
    """画像输入回显 + 实际数据右边界（Requirement 1.3, 2.5）。"""

    vt_symbol: str
    interval: str
    as_of: datetime  # 截止时间，必填、无隐式全量默认（Requirement 2.1）
    lookback_days: int
    # 裁剪后实际参与计算的最大 datetime（可能早于 as_of）；无数据时为 None
    effective_right_bound: datetime | None = None
    effective_bar_count: int = 0  # 实际参与计算的有效 bar 数
    rules_id: str  # 本次画像所用 Profiling_Rules 标识（Requirement 10.4）


class SymbolProfile(BaseModel):
    """单标的完整画像产物。

    available=False 时为"数据不可用"结构化结果（Requirement 1.5），此时
    blocks 通常为空且通过 unavailable_reason 给出原因。
    """

    input: ProfileInput  # 输入回显与有效右边界
    available: bool  # False 时为数据不可用（Requirement 1.5）
    unavailable_reason: str | None = None  # 数据不可用原因 + 本地实际可用区间提示
    blocks: list[MetricBlock] = Field(default_factory=list)  # 四个 Metric_Block
    group_profile: GroupProfile | None = None  # 多标的关联性画像（可选）
    suggestion: SchemeSuggestion | None = None  # 方案建议草稿（可关闭）
    overall_confidence: str  # 综合置信度，不高于关键指标最低值（Requirement 7.4）
    created_at: datetime = Field(default_factory=datetime.now)  # 计算时间戳
    artifact_id: str | None = None  # persist=True 且保存成功时返回，可用于历史画像读取
