"""
CNN 适配度综合分合成（scoring.py）。

本模块是 Tier-1 选股的核心纯函数：把单标的的
``SymbolProfile``（来自 profiling 模块）与 CNN 适配代理指标
（来自 ``proxy_metrics.py``）按 ``ScreeningRules.weights`` 合成
一个 [0,1] 的 ``CNN_Fitness_Score``，并附逐维贡献明细与综合置信度。

设计约束（贯穿全文件）：
- 纯函数、无 I/O：不读盘、不写盘、不调用网络。
- 复用优先：直接消费 profiling 模块产出，不重算任何画像指标。
- 等级→值映射集中在本文件顶部常量中（显式文档化），不散落在逻辑内。
- ``insufficient`` 维度排除出有效权重（权重归一分母不含其权重，
  避免 unavailable 维度稀释综合分，Requirement 2.3 / 2.5）。

等级→[0,1] 映射选择说明（design.md 3 节，决策注释）：

  ``liquidity``（流动性，越高 CNN 训练越有保障）：
    "low"            → 0.25（可勉强建模，但滑点风险大）
    "medium"         → 0.60（适中，CNN 可接受）
    "high"           → 1.00（优质流动性，最优）
    未知/None        → None（排除）

  ``volatility``（波动性，CNN 需要足够价格波动可学；
    但极低波动意味着几乎无信号可学，极高波动并不是坏事——
    CNN 擅长处理高频非线性波动，故采用单调上升映射）：
    "low"            → 0.20（趋近平稳，信号极弱）
    "medium"         → 0.60（适中，CNN 常见场景）
    "high"           → 1.00（高波动，CNN 可挖掘的结构多）
    未知/None        → None（排除）

  ``predictability``（线性可预测性，偏重"有结构"而非"线性强度"；
    有结构意味着 CNN 也能从中获益，indeterminate 表示无显著结构，
    trending / mean_reverting 均为"有结构"→高映射）：
    "indeterminate"  → 0.20（无显著线性结构，CNN 也难以发现趋势）
    "trending"       → 0.85（趋势结构，CNN 可学习持续方向）
    "mean_reverting" → 0.85（均值回复结构，CNN 可学习反转形态）
    未知/None        → None（排除）

  ``data_quality``（由 count_valid_bars 对 min_train_samples 的比例
    派生连续值 [0,1]，而非离散等级，详见 ``_data_quality_value``）：
    clip(count_valid_bars / min_train_samples[interval], 0, 1)

  ``nonlinearity`` / ``pattern_recurrence`` / ``temporal_stability``
  （CNN 代理指标）：MetricValue.value 已为 ~[0,1]，clip(value, 0, 1) 即可。
"""

from __future__ import annotations

from typing import NamedTuple

from aitrade.profiling.rules import DEFAULT_RULES as _PROFILING_RULES
from aitrade.profiling.rules import overall_confidence
from aitrade.profiling.types import MetricBlock, MetricValue, SymbolProfile
from aitrade.screening.rules import DEFAULT_SCREENING_RULES, ScreeningRules
from aitrade.screening.types import ScoreContribution, Tier1Score

# ---------------------------------------------------------------------------
# 等级→[0,1] 映射常量（显式文档化，见模块 docstring 及 design.md 3 节）
# ---------------------------------------------------------------------------

#: 流动性等级 → 归一化值。
#: 低流动性仍给正值（0.25），因为该标的可能在其他维度表现好，
#: 完全归 0 会掩盖其他有效信息。
LIQUIDITY_LEVEL_VALUE: dict[str, float] = {
    "low": 0.25,
    "medium": 0.60,
    "high": 1.00,
}

#: 波动等级 → 归一化值。
#: CNN 擅长高波动场景中提取局部非线性结构，故单调上升映射。
#: "low"=0.20 而非 0：低波动仍可能有形态，但 CNN 可学习信号极少。
VOLATILITY_LEVEL_VALUE: dict[str, float] = {
    "low": 0.20,
    "medium": 0.60,
    "high": 1.00,
}

#: 结构/可预测性等级 → 归一化值。
#: "trending" 与 "mean_reverting" 均代表"有可学结构"，映射相同高值；
#: "indeterminate" 代表无显著结构，映射低值（非零：不能完全排除隐含结构）。
PREDICTABILITY_LEVEL_VALUE: dict[str, float] = {
    "indeterminate": 0.20,
    "trending": 0.85,
    "mean_reverting": 0.85,
}


# ---------------------------------------------------------------------------
# 内部辅助：加权平均核（可独立测试，Property 3 / 4 直接作用于此）
# ---------------------------------------------------------------------------


class _WeightedAverageResult(NamedTuple):
    """``_weighted_average`` 的返回结构。

    Attributes:
        score: 归一化综合分 ∈ [0,1]，等于 Σ(w_i·v_i) / Σ(w_i)。
        renorm_contributions: 每维度的归一化贡献 ``{dim: (w_i/Σw) * v_i}``；
            各项之和在浮点精度内等于 ``score``（Property 3 自洽性）。
    """

    score: float
    renorm_contributions: dict[str, float]


def _weighted_average(
    normalized: dict[str, float],
    weights: dict[str, float],
) -> _WeightedAverageResult:
    """对有效维度做权重归一后的加权平均，并返回逐维贡献。

    取 ``normalized`` 与 ``weights`` 共同存在的键作为有效维度，
    计算归一化权重加权均值。

    结果满足以下属性（供 Hypothesis 直接验证）：

    - **有界**（Property 3）：当所有 ``value ∈ [0,1]`` 且所有 ``weight >= 0``
      时，``score ∈ [0,1]``。
    - **贡献自洽**（Property 3）：``sum(renorm_contributions.values()) ≈ score``
      在浮点精度内成立。
    - **单调性**（Property 4）：加大任意单维 value 不会降低 score。

    Args:
        normalized: 各有效维度的归一化值 ``{dim: value ∈ [0,1]}``；
            仅包含**已通过** insufficient 过滤的维度。
        weights: 各维度权重 ``{dim: weight >= 0}``；
            不在 ``normalized`` 中的键会被忽略。

    Returns:
        ``_WeightedAverageResult(score, renorm_contributions)``。
        若 ``normalized`` 为空或所有有效维度的权重之和为 0，则返回
        ``score=0.0``、``renorm_contributions={}``（调用方应据此输出 None）。

    Example:
        >>> result = _weighted_average({"a": 0.8, "b": 0.2}, {"a": 1.0, "b": 1.0})
        >>> abs(result.score - 0.5) < 1e-9
        True
        >>> abs(sum(result.renorm_contributions.values()) - result.score) < 1e-9
        True
    """
    # 取两字典的交集维度（维度必须同时出现在 normalized 和 weights 中）
    dims = [d for d in normalized if d in weights]
    total_w = sum(weights[d] for d in dims)

    if total_w <= 0.0:
        return _WeightedAverageResult(score=0.0, renorm_contributions={})

    score = sum(weights[d] * normalized[d] for d in dims) / total_w
    # renorm_weight_i = w_i / Σw；contribution_i = renorm_weight_i * v_i
    renorm_contributions = {
        d: (weights[d] / total_w) * normalized[d] for d in dims
    }
    return _WeightedAverageResult(score=score, renorm_contributions=renorm_contributions)


# ---------------------------------------------------------------------------
# 维度提取辅助（从 SymbolProfile.blocks 读取画像块）
# ---------------------------------------------------------------------------


def _find_block(blocks: list[MetricBlock], block_name: str) -> MetricBlock | None:
    """在画像块列表中按名称查找指定块。

    Args:
        blocks: ``SymbolProfile.blocks`` 的全部块。
        block_name: 目标块名，如 ``"liquidity"`` / ``"volatility"`` 等。

    Returns:
        匹配的第一个 ``MetricBlock``；未找到时返回 None。
    """
    for b in blocks:
        if b.block == block_name:
            return b
    return None


def _find_metric(block: MetricBlock, key: str) -> MetricValue | None:
    """在块内按 key 查找具体指标。

    Args:
        block: 目标 ``MetricBlock``。
        key: 指标键，如 ``"avg_turnover"`` / ``"count_valid_bars"``。

    Returns:
        匹配的第一个 ``MetricValue``；未找到时返回 None。
    """
    for m in block.metrics:
        if m.key == key:
            return m
    return None


def _block_representative_confidence(block: MetricBlock) -> str:
    """取块内所有指标的最低置信度作为该块的代表置信度。

    若块内无指标，返回 ``"insufficient"``（保守降级）。

    Args:
        block: 目标 ``MetricBlock``。

    Returns:
        该块的代表置信度字符串，取值为 ``"insufficient"`` /
        ``"low"`` / ``"medium"`` / ``"high"`` 之一。
    """
    if not block.metrics:
        return "insufficient"
    return overall_confidence([m.confidence for m in block.metrics])


# ---------------------------------------------------------------------------
# 各维度提取：返回 (value_or_none, confidence, level_or_none)
# ---------------------------------------------------------------------------


def _extract_data_quality(
    profile: SymbolProfile,
) -> tuple[float | None, str, str | None]:
    """提取 data_quality 维度的归一化值与置信度。

    由 count_valid_bars / min_train_samples[interval]（按比例，clip 到 [0,1]）
    派生连续分值，用块内所有指标的最低置信度作为代表。

    Args:
        profile: 包含 ``blocks`` 与 ``input`` 字段的单标的画像。

    Returns:
        ``(value, confidence, level)`` 三元组。
        ``value`` 为 None 表示该维度无可用数值（应被排除）；
        ``level`` 恒为 None（data_quality 无离散等级）。
    """
    block = _find_block(profile.blocks, "data_quality")
    if block is None:
        return None, "insufficient", None

    conf = _block_representative_confidence(block)
    if conf == "insufficient":
        return None, conf, None

    cvb_metric = _find_metric(block, "count_valid_bars")
    if cvb_metric is None or cvb_metric.value is None:
        return None, conf, None

    interval = profile.input.interval
    min_bars = _PROFILING_RULES.min_train_samples.get(
        interval,
        _PROFILING_RULES.min_train_samples.get("default", 240),
    )
    count = float(cvb_metric.value)
    val = min(1.0, max(0.0, count / min_bars)) if min_bars > 0 else 0.0
    return val, conf, None


def _extract_level_dim(
    profile: SymbolProfile,
    block_name: str,
    level_map: dict[str, float],
) -> tuple[float | None, str, str | None]:
    """提取以离散等级为基础的画像块维度（liquidity / volatility / predictability）。

    Args:
        profile: 单标的画像，提供 blocks。
        block_name: 块名，如 ``"liquidity"``。
        level_map: 等级名 → [0,1] 归一化值的映射常量。

    Returns:
        ``(value, confidence, level)`` 三元组；
        level 为块的 ``MetricBlock.level`` 字段（可能为 None）。
    """
    block = _find_block(profile.blocks, block_name)
    if block is None:
        return None, "insufficient", None

    conf = _block_representative_confidence(block)
    level = block.level
    val = level_map.get(level) if level else None
    return val, conf, level


def _extract_proxy(
    proxies: dict[str, MetricValue],
    dim: str,
) -> tuple[float | None, str, str | None]:
    """提取 CNN 代理指标维度。

    MetricValue.value 已为 ~[0,1]，clip 到 [0,1] 后直接使用。

    Args:
        proxies: ``{dim_name: MetricValue}`` 字典，由 proxy_metrics.py 产出。
        dim: 指标维度名，如 ``"nonlinearity"``。

    Returns:
        ``(value, confidence, None)`` 三元组；level 恒为 None（代理指标无离散等级）。
    """
    mv = proxies.get(dim)
    if mv is None:
        return None, "insufficient", None
    conf = mv.confidence
    if mv.value is None or conf == "insufficient":
        return None, conf, None
    val = min(1.0, max(0.0, float(mv.value)))
    return val, conf, None


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def compute_fitness_score(
    profile: SymbolProfile,
    proxies: dict[str, MetricValue],
    rules: ScreeningRules | None = None,
) -> Tier1Score:
    """由画像四块等级 + CNN 代理指标，按 rules.weights 合成 [0,1] 综合分。

    对 ``SymbolProfile`` 的四个画像块（data_quality / liquidity / volatility /
    predictability）做等级→值映射，对 CNN 代理指标（nonlinearity /
    pattern_recurrence / temporal_stability）直接裁剪到 [0,1]，
    按 ``ScreeningRules.weights`` 做权重归一后的加权平均，得 fitness_score。

    包含以下边界处理：

    - ``profile.available is False``：立即返回 ``fitness_score=None``
      的不可用结果（Requirement 2.4）。
    - 某维度 value 为 None 或 confidence == ``"insufficient"``：该维度排除出
      有效权重（其权重不计入分母，Requirement 2.3 / 2.5）。
    - 所有维度均被排除（有效权重为 0）：``fitness_score=None``。

    ``ScoreContribution.weight`` 填入**归一化后**的权重份额（w_i / Σw），
    使 ``sum(c.contribution for c in included_contributions) == fitness_score``
    （Property 3 贡献自洽）。

    综合置信度复用 ``profiling.rules.overall_confidence``，取所有
    **参与有效维度** confidence 的最低值。

    Args:
        profile: 由 ``Profiler.profile(...)`` 产出的单标的完整画像；
            ``available=False`` 时函数立即返回不可用结构（不使用 proxies）。
        proxies: CNN 代理指标字典，键为维度名（如 ``"nonlinearity"``），
            值为 ``MetricValue``（``value`` 已为 ~[0,1]，confidence 分档同画像块）。
            维度缺失时该维度视作 insufficient 处理（排除）。
        rules: 权重与漏斗配置；缺省时使用 ``DEFAULT_SCREENING_RULES``。

    Returns:
        ``Tier1Score``，含：

        - ``fitness_score``：[0,1] 或 None（数据不可用 / 全维度 insufficient）
        - ``contributions``：所有维度的明细列表，包含被排除维度（weight=0）
        - ``overall_confidence``：参与有效维度的最低置信度
        - ``available``：False 时为数据不可用
        - ``note``：不可用原因或 None

    Example:
        >>> score = compute_fitness_score(profile, proxies, rules)
        >>> assert score.fitness_score is None or 0.0 <= score.fitness_score <= 1.0
        >>> included = [c for c in score.contributions if c.weight > 0]
        >>> if score.fitness_score is not None:
        ...     assert abs(sum(c.contribution for c in included) - score.fitness_score) < 1e-9
    """
    if rules is None:
        rules = DEFAULT_SCREENING_RULES

    vt_symbol = profile.input.vt_symbol

    # ---- 快速路径：数据不可用 ----
    if not profile.available:
        return Tier1Score(
            vt_symbol=vt_symbol,
            fitness_score=None,
            contributions=[],
            overall_confidence="insufficient",
            available=False,
            note=profile.unavailable_reason or "标的数据不可用",
        )

    # ---- 逐维度提取 (value, confidence, level) ----
    _DIM_EXTRACTORS: list[tuple[str, tuple[float | None, str, str | None]]] = [
        ("data_quality",      _extract_data_quality(profile)),
        ("liquidity",         _extract_level_dim(profile, "liquidity",      LIQUIDITY_LEVEL_VALUE)),
        ("volatility",        _extract_level_dim(profile, "volatility",     VOLATILITY_LEVEL_VALUE)),
        ("predictability",    _extract_level_dim(profile, "predictability", PREDICTABILITY_LEVEL_VALUE)),
        ("nonlinearity",      _extract_proxy(proxies, "nonlinearity")),
        ("pattern_recurrence",_extract_proxy(proxies, "pattern_recurrence")),
        ("temporal_stability",_extract_proxy(proxies, "temporal_stability")),
    ]

    # ---- 分离有效维度（value 非 None 且 confidence != "insufficient"）----
    included_normalized: dict[str, float] = {}
    included_confidences: list[str] = []

    for dim, (val, conf, _level) in _DIM_EXTRACTORS:
        if dim not in rules.weights:
            continue
        if val is not None and conf != "insufficient":
            included_normalized[dim] = val
            included_confidences.append(conf)

    # ---- 加权平均 ----
    wa = _weighted_average(included_normalized, rules.weights)

    fitness_score: float | None = wa.score if included_normalized else None

    # ---- 组装 contributions（包含被排除维度，weight=0, contribution=0）----
    contributions: list[ScoreContribution] = []

    for dim, (val, conf, level) in _DIM_EXTRACTORS:
        if dim not in rules.weights:
            continue

        if dim in wa.renorm_contributions:
            # 有效维度：renorm_weight = w_i / Σw，contribution = renorm_weight * v_i
            total_w_eff = sum(rules.weights[d] for d in included_normalized if d in rules.weights)
            renorm_w = rules.weights[dim] / total_w_eff if total_w_eff > 0 else 0.0
            contrib_val = wa.renorm_contributions[dim]
        else:
            # 排除维度：weight=0，contribution=0
            renorm_w = 0.0
            contrib_val = 0.0

        contributions.append(
            ScoreContribution(
                dimension=dim,
                raw_value=val,
                level=level,
                weight=renorm_w,
                contribution=contrib_val,
                confidence=conf,
            )
        )

    # ---- 综合置信度 ----
    oc = overall_confidence(included_confidences) if included_confidences else "insufficient"

    return Tier1Score(
        vt_symbol=vt_symbol,
        fitness_score=fitness_score,
        contributions=contributions,
        overall_confidence=oc,
        available=True,
    )
