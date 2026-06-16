"""Scheme 建议草稿生成器。

本模块只把画像判定翻译为建议项，不写方案、不触发训练或回测。
唯一对外接口为 build_scheme_suggestion()，接收画像 blocks 与规则，
返回 SchemeSuggestion（status 恒为 "draft"，Requirement 8.4）。
"""

from __future__ import annotations

from aitrade.profiling.rules import ProfilingRules
from aitrade.profiling.types import MetricBlock, SchemeSuggestion, SuggestionItem


_CONF_RANK = {"insufficient": 0, "low": 1, "medium": 2, "high": 3}


def _block(blocks: list[MetricBlock], name: str) -> MetricBlock | None:
    """从画像块列表中按 block 名查找，未找到返回 None。"""
    return next((b for b in blocks if b.block == name), None)


def _metric_conf(block: MetricBlock | None, key: str) -> str:
    """从指定 block 中取指标 key 的置信度；block 为 None 或指标不存在时返回 "insufficient"。"""
    if block is None:
        return "insufficient"
    metric = next((m for m in block.metrics if m.key == key), None)
    return metric.confidence if metric is not None else "insufficient"


def _can_suggest(confidence: str) -> bool:
    """判断给定置信度是否达到可生成建议的下限（medium 或以上）。"""
    return _CONF_RANK.get(confidence, 0) >= _CONF_RANK["medium"]


def _item(field: str, value, reason: str, confidence: str) -> SuggestionItem:
    """快捷构造 SuggestionItem 的工厂函数。

    Args:
        field: Scheme 字段路径，如 ``"label_spec.take_profit"``。
        value: 建议取值。
        reason: 命中规则的文字说明（Requirement 8.2）。
        confidence: 该建议依据的置信度等级。

    Returns:
        SuggestionItem 对象。
    """
    return SuggestionItem(
        field=field,
        value=value,
        reason=reason,
        based_on_confidence=confidence,
    )


def build_scheme_suggestion(
    *,
    vt_symbol: str,
    interval: str,
    blocks: list[MetricBlock],
    overall_confidence: str,
    rules: ProfilingRules,
) -> SchemeSuggestion:
    """根据画像指标块生成只读方案建议草稿（status 恒为 "draft"）。

    生成逻辑：
    1. 整体置信度为 insufficient 或数据质量 insufficient → 降级，仅建议扩大 lookback；
    2. 波动等级（实现波动率）→ OCO 止盈/止损比例与持有天数建议；
    3. 可预测性结构判定（hurst/VR/ADF 投票）→ 标签类型与策略族建议；
    4. 流动性等级（日均成交额）→ 滑点提示与日内交易建议；
    5. 无任何建议项时降级（关键指标置信度不足）。
    以上各步均受置信度门控（< medium 不下结论，Requirement 7.3）。

    Args:
        vt_symbol: 目标标的合约代码（写入 SchemeSuggestion.vt_symbols）。
        interval: K 线周期（写入 SchemeSuggestion.interval）。
        blocks: 四个 MetricBlock 列表（来自 Profiler._build_blocks()）。
        overall_confidence: 综合置信度等级（来自 rules.overall_confidence()）。
        rules: 画像规则配置，包含 suggestion_map。

    Returns:
        SchemeSuggestion 草稿对象（status="draft"，不触发任何写入）。
    """
    suggestion = SchemeSuggestion(
        interval=interval,
        vt_symbols=[vt_symbol],
        degraded=overall_confidence == "insufficient",
    )

    data_conf = _metric_conf(_block(blocks, "data_quality"), "count_valid_bars")
    if overall_confidence == "insufficient" or not _can_suggest(data_conf):
        suggestion.degraded = True
        suggestion.note = "样本不足，暂不输出强参数建议；请扩大 lookback 或补齐本地行情。"
        suggestion.items.append(
            _item(
                "data.lookback_days",
                "increase",
                "有效样本不足，优先扩大回看窗口或补齐行情数据。",
                data_conf,
            )
        )
        suggestion.items.append(
            _item(
                "interval",
                interval,
                "保持当前周期，仅作为数据诊断草稿，不触发训练或回测。",
                data_conf,
            )
        )
        return suggestion

    volatility = _block(blocks, "volatility")
    vol_level = volatility.level if volatility and volatility.level else "low"
    vol_conf = _metric_conf(volatility, "realized_volatility")
    if _can_suggest(vol_conf):
        vol_map = rules.suggestion_map.get("volatility", {}).get(vol_level, {})
        if vol_map:
            suggestion.items.extend(
                [
                    _item(
                        "label_spec.mode",
                        "oco",
                        f"波动等级为 {vol_level}，建议使用 OCO 标签以显式约束盈亏边界。",
                        vol_conf,
                    ),
                    _item(
                        "label_spec.take_profit",
                        vol_map.get("tp"),
                        f"波动等级为 {vol_level}，按规则 {rules.rules_id} 映射止盈比例。",
                        vol_conf,
                    ),
                    _item(
                        "label_spec.stop_loss",
                        vol_map.get("sl"),
                        f"波动等级为 {vol_level}，按规则 {rules.rules_id} 映射止损比例。",
                        vol_conf,
                    ),
                    _item(
                        "label_spec.max_hold",
                        vol_map.get("horizon"),
                        f"波动等级为 {vol_level}，按规则 {rules.rules_id} 映射最大持有 bar 数。",
                        vol_conf,
                    ),
                ]
            )
            if vol_level == "low":
                suggestion.note = "低波动标的的 tp/sl 空间较窄，需特别关注交易成本与滑点。"

    predictability = _block(blocks, "predictability")
    structure = predictability.level if predictability and predictability.level else "indeterminate"
    pred_conf = _metric_conf(predictability, "variance_ratio")
    if _can_suggest(pred_conf):
        struct_map = rules.suggestion_map.get("structure", {}).get(structure, {})
        if struct_map:
            suggestion.items.append(
                _item(
                    "predictor.params.label_type",
                    struct_map.get("label_type"),
                    f"可预测性结构判定为 {structure}，按规则映射标签类型。",
                    pred_conf,
                )
            )
            suggestion.items.append(
                _item(
                    "strategy.params.family_hint",
                    struct_map.get("strategy_family"),
                    f"可预测性结构判定为 {structure}，按规则映射策略族倾向。",
                    pred_conf,
                )
            )

    liquidity = _block(blocks, "liquidity")
    liq_level = liquidity.level if liquidity and liquidity.level else "low"
    liq_conf = _metric_conf(liquidity, "avg_turnover")
    if _can_suggest(liq_conf):
        liq_map = rules.suggestion_map.get("liquidity", {}).get(liq_level, {})
        if liq_map:
            suggestion.items.append(
                _item(
                    "cost.slippage",
                    liq_map.get("slippage_hint"),
                    f"流动性等级为 {liq_level}，给出滑点风险提示。",
                    liq_conf,
                )
            )
            suggestion.items.append(
                _item(
                    "strategy.params.allow_intraday",
                    liq_map.get("intraday"),
                    f"流动性等级为 {liq_level}，给出日内交易适配建议。",
                    liq_conf,
                )
            )

    if not suggestion.items:
        suggestion.degraded = True
        suggestion.note = "关键指标置信度不足，建议仅作诊断参考。"
        suggestion.items.append(
            _item(
                "data.lookback_days",
                "increase",
                "关键指标置信度不足，暂不输出强参数建议。",
                overall_confidence,
            )
        )
    return suggestion
