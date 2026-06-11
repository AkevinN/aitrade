"""
标的画像（Symbol Profiling）的规则与判定纯函数。

本文件集中管理画像的**业务阈值与映射规则**（Profiling_Rules），并提供一组
**纯函数**用于把"有效样本量 / 指标数值"翻译为离散的置信度等级与流动性 /
波动性 / 结构判定。严格对应 design.md "3. 规则与置信度判定（rules.py）"。

设计原则（Requirement 10.1）：
- 所有等级判定阈值、统计有效性下限、判定→建议映射集中在 ``ProfilingRules`` 中，
  ``metrics.py`` 的指标计算函数内**不得**硬编码这些业务阈值。
- 判定逻辑实现为纯函数：无 I/O、无副作用，输入标量 + 规则，输出离散等级字符串，
  便于单元测试与基于属性的测试（Properties 6 / 8 / 9）。

边界语义（贯穿本文件，保证落档清晰且单调）：
- 置信度分档采用"低于阈值则降一档"的左开区间约定：
  ``n < insufficient_below`` → ``insufficient``；``< low_below`` → ``low``；
  ``< medium_below`` → ``medium``；``>= medium_below`` → ``high``。
- 流动性 / 波动性分档采用"达到（含等于）下界则进入该档"的约定：
  ``value >= 某档下界`` 即归入该档；取所有满足条件的档中的最高档。
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator

# 置信度等级由低到高的全序，用于综合置信度取最低值与单调性判定（Requirement 7.4）
_CONFIDENCE_ORDER: list[str] = ["insufficient", "low", "medium", "high"]
_CONFIDENCE_RANK: dict[str, int] = {name: i for i, name in enumerate(_CONFIDENCE_ORDER)}


class ConfidenceThreshold(BaseModel):
    """单指标的样本量分档（有效性下限）。Requirement 7.1, 10.1

    分档边界（左开右闭语义，样本量越大置信度越高、单调非降）：
    - ``effective_sample < insufficient_below`` → ``insufficient``（抑制数值，Requirement 7.2）
    - ``insufficient_below <= n < low_below``    → ``low``
    - ``low_below <= n < medium_below``          → ``medium``
    - ``n >= medium_below``                      → ``high``

    三个阈值必须非降（``insufficient_below <= low_below <= medium_below``），
    以保证 ``confidence_for`` 关于样本量单调非降（Property 6）。
    """

    insufficient_below: int  # 低于此值 → insufficient（Requirement 7.2）
    low_below: int  # 低于此值（且 >= insufficient_below）→ low
    medium_below: int  # 低于此值（且 >= low_below）→ medium；>= 此值 → high

    @field_validator("low_below")
    @classmethod
    def _low_ge_insufficient(cls, v: int, info: Any) -> int:
        """校验 low_below >= insufficient_below，确保分档单调。"""
        below = info.data.get("insufficient_below")
        if below is not None and v < below:
            raise ValueError("low_below 必须 >= insufficient_below（分档需单调非降）")
        return v

    @field_validator("medium_below")
    @classmethod
    def _medium_ge_low(cls, v: int, info: Any) -> int:
        """校验 medium_below >= low_below，确保分档单调。"""
        low = info.data.get("low_below")
        if low is not None and v < low:
            raise ValueError("medium_below 必须 >= low_below（分档需单调非降）")
        return v


# 当 metric_key 不在 rules.confidence 中时使用的合理 fallback 分档。
# 取一组适用于大多数统计量的保守默认值。
_FALLBACK_CONFIDENCE = ConfidenceThreshold(
    insufficient_below=30,
    low_below=100,
    medium_below=300,
)


class ProfilingRules(BaseModel):
    """集中管理画像阈值与映射，带内置默认值。Requirement 10.1 / 10.2

    所有业务阈值都应来源于本模型，指标计算函数（metrics.py）不重复定义。
    """

    rules_id: str = "builtin-v1"  # 规则标识 / 版本，写入 Profile_Artifact（Requirement 10.4）

    # 按 interval（或 (T, horizon) 派生键）给出训练所需最低样本量；
    # 含 "default" 兜底键，供未列出的 interval 使用（Requirement 3.1）。
    min_train_samples: dict[str, int] = Field(default_factory=dict)

    # 每个指标键 → 该指标的样本量分档阈值（Requirement 7.1）。
    confidence: dict[str, ConfidenceThreshold] = Field(default_factory=dict)

    # 流动性等级（高/中/低）的成交额下界：键为档名，值为进入该档的成交额下界。
    # 约定只需给出非最低档的下界（如 medium / high），未达任何下界者归 "low"。
    # 边界含等于：turnover >= 某档下界 即可进入该档（Requirement 4.3）。
    liquidity_levels: dict[str, float] = Field(default_factory=dict)

    # 波动等级阈值，语义同上：vol >= 某档下界 即进入该档，未达者归 "low"（Requirement 5.2）。
    volatility_levels: dict[str, float] = Field(default_factory=dict)

    # 判定 → 建议映射（波动等级 → tp/sl、horizon；结构判定 → reg/cls、策略族）。
    # 结构为开放字典，由 recommender.py 消费（Requirement 5.3 / 6.3 / 8.1）。
    suggestion_map: dict[str, Any] = Field(default_factory=dict)

    def confidence_threshold_for(self, metric_key: str) -> ConfidenceThreshold:
        """取指定指标的分档；缺失时回退到合理的 fallback 分档。

        优先级：精确键 → rules.confidence["default"] → 模块级 _FALLBACK_CONFIDENCE。
        """
        if metric_key in self.confidence:
            return self.confidence[metric_key]
        if "default" in self.confidence:
            return self.confidence["default"]
        return _FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# 内置默认规则（Requirement 10.2）：未自定义时开箱即用。
# ---------------------------------------------------------------------------

# 各指标的默认样本量分档。键名与 metrics.py 中各指标返回的 key 对齐。
# 偏统计的高阶指标（hurst / adf / variance_ratio 等）需要更大样本才可靠，
# 因此其分档阈值整体高于简单比例类指标。
_DEFAULT_CONFIDENCE: dict[str, ConfidenceThreshold] = {
    "default": ConfidenceThreshold(insufficient_below=30, low_below=100, medium_below=300),
    # 数据质量类：比例统计，样本要求相对低
    "count_valid_bars": ConfidenceThreshold(insufficient_below=20, low_below=60, medium_below=120),
    "gap_ratio": ConfidenceThreshold(insufficient_below=20, low_below=60, medium_below=120),
    "zero_volume_ratio": ConfidenceThreshold(insufficient_below=20, low_below=60, medium_below=120),
    "alignment_coverage": ConfidenceThreshold(insufficient_below=20, low_below=60, medium_below=120),
    # 流动性 / 波动性类：需要一定样本估计稳定统计量
    "avg_turnover": ConfidenceThreshold(insufficient_below=20, low_below=60, medium_below=120),
    "intraday_concentration": ConfidenceThreshold(insufficient_below=40, low_below=120, medium_below=240),
    "realized_volatility": ConfidenceThreshold(insufficient_below=30, low_below=100, medium_below=250),
    "atr_ratio": ConfidenceThreshold(insufficient_below=30, low_below=100, medium_below=250),
    "amplitude_quantiles": ConfidenceThreshold(insufficient_below=30, low_below=100, medium_below=250),
    # 可预测性 / 结构类：统计量收敛慢，要求更大样本
    "return_autocorr": ConfidenceThreshold(insufficient_below=50, low_below=150, medium_below=400),
    "hurst_exponent": ConfidenceThreshold(insufficient_below=100, low_below=250, medium_below=500),
    "variance_ratio": ConfidenceThreshold(insufficient_below=60, low_below=200, medium_below=500),
    "adf_pvalue": ConfidenceThreshold(insufficient_below=60, low_below=200, medium_below=500),
    "skewness": ConfidenceThreshold(insufficient_below=50, low_below=150, medium_below=400),
    "kurtosis": ConfidenceThreshold(insufficient_below=50, low_below=150, medium_below=400),
}

DEFAULT_RULES = ProfilingRules(
    rules_id="builtin-v1",
    # 按 interval 的最低训练样本量；default 供未列出周期兜底（Requirement 3.1）。
    min_train_samples={
        "default": 240,
        "1m": 2000,
        "5m": 1000,
        "10m": 600,
        "30m": 480,
        "d": 250,
    },
    confidence=_DEFAULT_CONFIDENCE,
    # 成交额（单位：本币金额）分档下界；未达 medium 下界者归 low。
    # high: 日均成交额 >= 2e7；medium: >= 5e6；否则 low。
    liquidity_levels={
        "medium": 5_000_000.0,
        "high": 20_000_000.0,
    },
    # 已实现波动率分档下界；未达 medium 下界者归 low。
    # high: vol >= 0.03；medium: >= 0.012；否则 low。
    volatility_levels={
        "medium": 0.012,
        "high": 0.03,
    },
    # 判定 → 建议映射的内置默认（由 recommender.py 在任务 6.1 消费）。
    suggestion_map={
        # 波动等级 → OCO tp/sl 比例与 horizon 建议区间（Requirement 5.3）
        "volatility": {
            "low": {"tp": 0.02, "sl": 0.01, "horizon": 10},
            "medium": {"tp": 0.05, "sl": 0.02, "horizon": 10},
            "high": {"tp": 0.10, "sl": 0.04, "horizon": 5},
        },
        # 结构判定 → reg/cls 与策略族倾向（Requirement 6.3）
        "structure": {
            "trending": {"label_type": "reg", "strategy_family": "trend_following"},
            "mean_reverting": {"label_type": "cls", "strategy_family": "mean_reversion"},
            "indeterminate": {"label_type": "cls", "strategy_family": "neutral"},
        },
        # 流动性等级 → 成本 / 滑点提示（Requirement 4.4）
        "liquidity": {
            "low": {"slippage_hint": "high", "intraday": False},
            "medium": {"slippage_hint": "medium", "intraday": True},
            "high": {"slippage_hint": "low", "intraday": True},
        },
    },
)


# ---------------------------------------------------------------------------
# 判定纯函数
# ---------------------------------------------------------------------------


def confidence_for(metric_key: str, effective_sample: int, rules: ProfilingRules) -> str:
    """按有效样本量分档返回置信度等级。Requirement 7.1 / 7.2

    纯函数。规则：
    - ``effective_sample < insufficient_below`` 恒返回 ``"insufficient"``；
    - 关于 ``effective_sample`` 单调非降（样本越多等级不降，Property 6）；
    - ``metric_key`` 不在 ``rules.confidence`` 时回退到合理的 fallback 分档。

    负样本量按 0 处理（防御性），同样落入 insufficient。
    """
    threshold = rules.confidence_threshold_for(metric_key)
    n = max(0, int(effective_sample))
    if n < threshold.insufficient_below:
        return "insufficient"
    if n < threshold.low_below:
        return "low"
    if n < threshold.medium_below:
        return "medium"
    return "high"


def _bucket_by_lower_bounds(value: float, bounds: dict[str, float], lowest: str) -> str:
    """通用分档：给定"档名→进入该档的下界"，返回 value 满足下界中的最高档。

    单调性保证：value 增大时，满足 ``value >= 下界`` 的档集合只增不减，
    取其中下界最大者，故返回档关于 value 单调非降（Property 9）。
    边界含等于：``value >= 下界`` 即视为进入该档。
    NaN 防御：非有限值归入最低档 ``lowest``。
    任意输入都落在 ``{lowest} ∪ bounds.keys()`` 这一有限等级集合内。
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return lowest
    chosen = lowest
    chosen_bound = -math.inf
    for level, bound in bounds.items():
        if value >= bound and bound >= chosen_bound:
            chosen = level
            chosen_bound = bound
    return chosen


def liquidity_level(turnover: float, rules: ProfilingRules) -> str:
    """依据成交额给出流动性等级（高/中/低）。Requirement 4.3

    纯函数，关于 ``turnover`` 单调非降：成交额越高等级不降。
    未达任何档下界者归 ``"low"``。
    """
    return _bucket_by_lower_bounds(turnover, rules.liquidity_levels, lowest="low")


def volatility_level(vol: float, rules: ProfilingRules) -> str:
    """依据波动值给出波动等级（高/中/低）。Requirement 5.2

    纯函数，关于 ``vol`` 单调非降：波动越高等级不降。
    未达任何档下界者归 ``"low"``。
    """
    return _bucket_by_lower_bounds(vol, rules.volatility_levels, lowest="low")


def structure_judgement(
    hurst: float | None,
    vr: float | None,
    adf_p: float | None,
    confidence: str,
) -> str:
    """给出结构判定：偏趋势 / 偏均值回复 / 不显著。Requirement 6.2

    返回 ``"trending"`` / ``"mean_reverting"`` / ``"indeterminate"``。纯函数。

    门控（Requirement 7.3）：当依据的置信度为 ``insufficient`` 或 ``low`` 时，
    不下结论，倾向返回 ``"indeterminate"``，避免样本不足时给出误导结论。

    判据（基于标准统计参考点，非业务阈值）：
    - Hurst > 0.5 → 趋势性（持久）；< 0.5 → 均值回复。
    - 方差比 VR > 1 → 趋势；< 1 → 均值回复。
    - ADF p 值 < 0.05 → 平稳（均值回复）；>= 0.05 → 非平稳（趋势倾向）。
    多个可用信号投票，票数持平或无可用信号 → indeterminate。
    """
    if confidence in ("insufficient", "low"):
        return "indeterminate"

    trend_votes = 0
    mr_votes = 0

    def _ok(x: float | None) -> bool:
        return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))

    if _ok(hurst):
        if hurst > 0.5:
            trend_votes += 1
        elif hurst < 0.5:
            mr_votes += 1
    if _ok(vr):
        if vr > 1.0:
            trend_votes += 1
        elif vr < 1.0:
            mr_votes += 1
    if _ok(adf_p):
        if adf_p < 0.05:
            mr_votes += 1
        else:
            trend_votes += 1

    if trend_votes > mr_votes:
        return "trending"
    if mr_votes > trend_votes:
        return "mean_reverting"
    return "indeterminate"


def overall_confidence(metric_confidences: list[str]) -> str:
    """综合置信度：取列表中最低等级。Requirement 7.4

    纯函数。在序 ``insufficient < low < medium < high`` 下返回最低等级，
    保证综合置信度不高于任一关键指标置信度。空列表返回 ``"insufficient"``。
    未知等级按最低（insufficient）处理，作为保守降级。
    """
    if not metric_confidences:
        return "insufficient"
    lowest_rank = min(
        _CONFIDENCE_RANK.get(c, 0) for c in metric_confidences
    )
    return _CONFIDENCE_ORDER[lowest_rank]
