"""
CNN 选股（CNN Stock Screening）响应数据模型。

本文件集中定义选股产物的 pydantic 模型，严格对应 design.md 的
"Data Models" 与 "Components and Interfaces" 章节。所有模型仅作数据承载，
不含计算逻辑，便于序列化为 Screening_Result 产物并作为 API 的 JSON 契约。

设计要点：
- ScreeningResult.status 固定字面量 "draft"：选股结论恒为草稿，
  不自动开训、不下真实单（Requirement 11.1）。
- LeaderboardRow 嵌套 Tier1Score + Tier2Verdict，榜单完整回显逐维贡献明细。
- Tier2Verdict.evaluable=False 时其余字段为 None；edge_ok 默认 False（Requirement 5.4）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ScoreContribution(BaseModel):
    """单维度对 CNN_Fitness_Score 的贡献明细。

    每个参与合成的维度（画像块或 CNN 代理指标）对应一条记录，
    前端可据此渲染"为什么这只分高/分低"的解释（Requirement 6.5）。
    """

    dimension: str  # 维度名，如 "volatility" / "nonlinearity" / "data_quality"
    raw_value: float | None  # 该维度的原始指标值；insufficient 时为 None
    level: str | None  # 该维度的等级判定，如 "high" / "medium" / "low"；无判定时为 None
    weight: float  # 在 ScreeningRules.weights 中配置的权重；恒 >= 0
    contribution: float  # 本维度对总分的贡献量 = weight × normalized_value（归一后）
    confidence: str  # 该维度的置信度等级："insufficient" / "low" / "medium" / "high"


class Tier1Score(BaseModel):
    """单标的 Tier-1 廉价预筛打分结果。

    available=False 时 fitness_score 为 None，该行排在榜单末尾且不入围 Tier-2
    （Requirement 2.4）。
    """

    vt_symbol: str  # 标的代码，如 "600030.SSE"
    # CNN 适配度综合分 [0,1]；available=False 或所有维度均 insufficient 时为 None
    fitness_score: float | None
    contributions: list[ScoreContribution]  # 逐维贡献明细（Requirement 6.5）
    overall_confidence: str  # 综合置信度，不高于所有参与维度的最低置信度（Requirement 2.3）
    available: bool  # False 表示本地数据不可用（Requirement 2.4）
    note: str | None = None  # 不可用原因 / 置信度降级说明


class Tier2Verdict(BaseModel):
    """单标的 Tier-2 WF/OOS 实证结论。

    evaluable=False 时（无折或抛异常）不输出 edge_ok/avg_score 等数值（Requirement 5.4）。
    edge_ok 基于绝对判据（跨折平均 candidate_score > 0 且 正折占比 >= min_positive_fold_ratio），
    不依赖相对晋级门禁 summary.passed（Requirement 5.2）。
    """

    vt_symbol: str  # 标的代码
    evaluable: bool  # False 表示 Tier-2 失败或折数为 0，不可派生 edge 结论
    edge_ok: bool = False  # 绝对 edge 门禁结论；evaluable=False 时恒 False
    avg_score: float | None = None  # 跨折跨种子平均 candidate_score；evaluable=False 时为 None
    pos_fold_ratio: float | None = None  # candidate_score > 0 的折数占比；evaluable=False 时为 None
    avg_cross_seed_std: float | None = None  # 跨种子 candidate_score 标准差均值；无多种子时为 None
    report_id: str | None = None  # WF 报告的 ID，可按此在 screening governance store 回读
    note: str | None = None  # 失败原因 / 不可评估说明


class LeaderboardRow(BaseModel):
    """榜单中的一行记录，含 Tier-1 打分与可选的 Tier-2 实证结论。

    每行对应一只标的；rank 按 CNN_Fitness_Score 降序编号（从 1 开始）；
    未入围 Tier-2 的行 tier2 为 None（Requirement 6.1）。
    """

    rank: int  # 榜单排名，从 1 开始（按 fitness_score 降序；available=False 的行排末尾）
    tier1: Tier1Score  # Tier-1 打分结果
    promoted_to_tier2: bool  # 是否入围了 Tier-2 评估
    tier2: Tier2Verdict | None = None  # Tier-2 实证结论；未入围或 run_tier2=False 时为 None


class ScreeningResult(BaseModel):
    """一次 CNN 选股批量运行的完整产物（Screening_Result）。

    回显输入参数以便复现；status 恒为 "draft"，不自动写入方案或开训
    （Requirement 11.1）。写入 SCREENING_PATH 时按 run_id + 时间戳命名（Requirement 6.3）。
    """

    run_id: str  # 唯一运行 ID，如 UUID4 字符串
    status: Literal["draft"] = "draft"  # 恒为草稿；前端应以视觉标注提示（Requirement 11.1）
    created_at: datetime  # 运行完成时间戳（UTC 或本机时区）
    # 输入回显：universe 来源/过滤条件/as_of/interval/漏斗参数/Tier-2 超参
    input: dict = Field(default_factory=dict)
    rules_id: str  # 本次所用 ScreeningRules 的标识/版本，用于复现（Requirement 6.2, 12.4）
    universe_size: int  # 过滤后进入 Tier-1 的候选标的数（不含已被排除的）
    excluded: list[dict] = Field(default_factory=list)  # 被排除标的 + 原因列表
    leaderboard: list[LeaderboardRow] = Field(default_factory=list)  # 选股榜单
    # Tier-1 实际数据右边界（最大的 effective_right_bound，用于审计无前视）；无数据时为 None
    effective_right_bound: datetime | None = None
    # Tier-2 评估区间 {"start": ..., "end": ..., "objective": ...}；未跑 Tier-2 时为 None
    eval_window: dict | None = None
