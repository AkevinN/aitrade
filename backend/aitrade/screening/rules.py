"""
CNN 选股（CNN Stock Screening）规则与配置。

本文件集中管理选股的**权重、漏斗参数与 Tier-2 默认超参**（ScreeningRules），
并提供带默认值的内置实例 DEFAULT_SCREENING_RULES（Requirement 12.1 / 12.2）。

设计原则（延续 profiling/rules.py 的 ProfilingRules + DEFAULT_RULES 风格）：
- 所有选股阈值与超参来源于 ScreeningRules，计算逻辑（scoring.py / edge.py）
  内不得硬编码。
- weights 中权重恒非负，合成时归一化处理（Property 3）。
- rules_id 版本化：写入 ScreeningResult 用于复现（Requirement 12.4）。
- DEFAULT_SCREENING_RULES 为"开箱即用"实例，Tier-2 默认超参偏保守快速（design.md 3.3）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ScreeningRules(BaseModel):
    """集中管理 CNN 选股阈值、权重与 Tier-2 超参（Requirement 12.1）。

    CNN_Fitness_Score 合成流程：对 weights 中的每个维度，把该维度的画像等级
    归一化到 [0,1]，再按权重加权求和并除以有效权重之和（insufficient 维度权重不计入）。
    """

    rules_id: str = "screening-builtin-v1"  # 规则标识/版本，写入 ScreeningResult（Requirement 12.4）

    # ---- CNN_Fitness_Score 各维权重 ----
    # 键为维度名；值为权重，恒 >= 0（由 field_validator 校验）。
    # 四个画像块（复用 profiling 判定）：data_quality / liquidity / volatility / predictability
    # 三个 CNN 代理指标（proxy_metrics.py 新增）：nonlinearity / pattern_recurrence / temporal_stability
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            # 画像四块权重（通用可建模性）
            "data_quality": 1.0,        # 数据质量：bar 完整性/缺口率/零成交比
            "liquidity": 1.5,           # 流动性：成交额分档；CNN 训练需足够成交
            "volatility": 1.5,          # 波动性：CNN 需有足够价格波动可学
            "predictability": 2.0,      # 可预测性：线性自相关/Hurst/结构判定
            # CNN 代理指标三块权重（CNN 适配增益）
            "nonlinearity": 1.5,        # 非线性结构：线性 AR 残差仍有可学结构
            "pattern_recurrence": 1.5,  # 形态复现性：卷积核可捕捉的重复 motif
            "temporal_stability": 1.0,  # 时间稳定性：样本内外形态是否漂移
        }
    )

    @field_validator("weights")
    @classmethod
    def _weights_nonnegative(cls, v: dict[str, float]) -> dict[str, float]:
        """校验 weights 中所有权重 >= 0（Property 3：权重恒非负且归一）。

        Args:
            v: 待校验的权重字典。

        Returns:
            原样返回校验通过的字典。

        Raises:
            ValueError: 任意权重 < 0 时抛出。
        """
        for dim, w in v.items():
            if w < 0:
                raise ValueError(f"权重 {dim!r} 必须 >= 0，当前值：{w}")
        return v

    # ---- 漏斗配置 ----
    top_k: int = Field(default=15, ge=1)  # Tier-1 后入围 Tier-2 的最大标的数（Requirement 4）
    # Tier-2 入围所需最低综合置信度；"insufficient" 实质上不过滤（Requirement 2.5）
    min_confidence: str = "low"

    # ---- 绝对 edge 门禁阈值 ----
    # candidate_score > 0 的折数占比需 >= 此值才判 edge_ok=True（Requirement 5.2）
    min_positive_fold_ratio: float = Field(default=0.5, ge=0.0, le=1.0)

    # ---- Tier-2 默认超参（偏保守快速，design.md 3.3）----
    objective: str = "classification"  # CNN 训练目标，传给 CNNWalkForwardRequest
    n_seeds: int = Field(default=1, ge=1)  # 每折训练种子数；偏少以控制算力
    epochs: int = Field(default=30, ge=1)  # 每次训练最大 epoch 数；偏小以加快速度
    # Tier-2 评估窗口长度（天数）：若 eval_start 未显式指定，则从 as_of 向前推此天数
    eval_window_days: int = Field(default=365, ge=30)
    # Tier-2 单折测试集长度（天数）；controls WF 切分粒度
    fold_test_days: int = Field(default=90, ge=7)
    # Tier-2 入围标的数上限保护：实际入围数超过此值时按 fitness_score 截断并 log（Requirement 4.5）
    tier2_cap: int = Field(default=30, ge=1)


# ---------------------------------------------------------------------------
# 内置默认规则（Requirement 12.2）：未自定义时开箱即用。
# ---------------------------------------------------------------------------

DEFAULT_SCREENING_RULES = ScreeningRules(
    rules_id="screening-builtin-v1",
    # 权重使用 ScreeningRules 字段默认值（见 Field default_factory）
    top_k=15,
    min_confidence="low",
    min_positive_fold_ratio=0.5,
    objective="classification",
    n_seeds=1,
    epochs=30,
    eval_window_days=365,
    fold_test_days=90,
    tier2_cap=30,
)
