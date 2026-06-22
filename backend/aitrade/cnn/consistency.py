"""
label ↔ 策略出场 一致性自检（迭代 2 红线守护）。

核心红线：训练 label 的「出场口径」必须与回测/实盘策略的「真实出场」一致，
否则模型学的目标与实际交易脱节，回测与实盘必然背离。

本模块提供：
- ``label_holding_horizon``：解析 label 蕴含的固定持有期（bar 数）。
- ``derive_strategy_exit_from_label``：由 label 自动推导对齐的固定持有出场配置。
- ``check_label_strategy_consistency``：校验策略出场与 label 是否一致；
  硬性不一致（fixed_hold 的 hold_days 与 label 持有期不符）直接抛错，
  软性问题（price_ref=close 研究口径、threshold 信号衰减式出场）返回告警列表。

对应文档：docs/07-CNN-Label可配置化与多方案迭代计划.md 的「label↔执行一致性矩阵」。
"""

from __future__ import annotations

from typing import Any

from .intervals import bars_to_days


def label_holding_horizon(label_spec: dict[str, Any] | None, input_interval: str = "d") -> int | None:
    """返回 label 蕴含的固定持有期（bar 数）；无法表达为固定 bar 数时返回 None。

    映射规则：
    - next_bar            → 1
    - horizon_bars        → spec["horizon"]（整型）
    - next_session_close  → 日线下每日一根 = 1；分钟线跨度不固定 → None
    - session_close       → 当日尾盘，跨度不固定 → None
    - oco                 → 路径依赖，realized 持有期不固定 → None

    Args:
        label_spec: 训练时传入的 label 配置字典；None 等价于 mode=next_bar。
        input_interval: K 线周期，如 "d"、"30m"。仅 next_session_close 模式下影响结果。

    Returns:
        固定持有期的 bar 数；持有期不固定或无法表达时返回 None。
    """
    spec = label_spec or {}
    mode = spec.get("mode") or "next_bar"

    if mode == "next_bar":
        return 1
    if mode == "horizon_bars":
        return int(spec.get("horizon") or 1)
    if mode == "next_session_close":
        return 1 if input_interval == "d" else None
    if mode == "session_close":
        return None
    return None


def derive_strategy_exit_from_label(
    label_spec: dict[str, Any] | None, input_interval: str = "d"
) -> dict[str, Any]:
    """由 label 自动推导与之精确对齐的出场配置。

    - OCO label → 返回 exit_mode=oco，含 take_profit/stop_loss/hold_days。
    - 固定持有 label（next_bar/horizon_bars/日线 next_session_close）→
      返回 exit_mode=fixed_hold，含 hold_days=持有**交易日数**
      （由 label 的 bar 跨度按 input_interval 经 bars_to_days 换算，策略按交易日计）。
    - 持有期不固定（如分钟级 session_close）→ 抛 ValueError。

    Args:
        label_spec: 训练时的 label 配置字典；None 等价于 mode=next_bar。
        input_interval: K 线周期，如 "d"、"30m"。

    Returns:
        出场配置字典，始终包含 exit_mode 键；
        fixed_hold 时含 hold_days；oco 时含 take_profit/stop_loss/hold_days。

    Raises:
        ValueError: label 的持有期不是固定 bar 数，且不属于 OCO 模式时抛出。
    """
    mode = (label_spec or {}).get("mode")
    # OCO 路径依赖标签 → 推导对齐的 OCO 出场（止盈/止损 + 最大持有兜底）。
    # max_hold 是 bar 数，策略 hold_days 按交易日计 → 用 bars_to_days 换算（日线恒等）。
    if mode == "oco":
        spec = label_spec or {}
        max_hold_bars = int(spec.get("max_hold") or spec.get("horizon") or 0)
        return {
            "exit_mode": "oco",
            "take_profit": float(spec.get("take_profit") or 0.0),
            "stop_loss": float(spec.get("stop_loss") or 0.0),
            "hold_days": bars_to_days(max_hold_bars, input_interval),
        }

    horizon = label_holding_horizon(label_spec, input_interval)
    if horizon is None:
        raise ValueError(
            f"无法由 label(mode={mode}, interval={input_interval}) 自动推导固定持有出场："
            "该 label 的持有期不是固定 bar 数；请改用固定持有期 label（next_bar/horizon_bars），"
            "或显式指定 exit_mode=threshold。"
        )
    # horizon 是 bar 数；策略 hold_days 按交易日计，按周期向上取整换算（日线 ÷1 恒等）。
    return {"exit_mode": "fixed_hold", "hold_days": bars_to_days(horizon, input_interval)}


def check_label_strategy_consistency(
    label_spec: dict[str, Any] | None,
    exit_mode: str,
    hold_days: int,
    input_interval: str = "d",
) -> list[str]:
    """校验策略出场与 label 是否一致，返回软性告警列表。

    硬性不一致 → 直接抛 ValueError（回测失败，杜绝「跑出来但口径错」的隐性失败）：
    - fixed_hold 但 label 持有期非固定 bar 数；
    - fixed_hold 的 hold_days 与 label 持有期不符。

    软性问题 → 追加到返回列表（不阻断，但记录并回传前端）：
    - price_ref=close（研究口径，按收盘价成交，实盘吃不到）；
    - price_ref=next_vwap（VWAP 执行成本提示）；
    - exit_mode=threshold（信号衰减式出场，realized 持有期会偏离训练 horizon）；
    - exit_mode=oco 与非 OCO label 的口径不一致。

    Args:
        label_spec: 训练时的 label 配置字典；None 等价于 mode=next_bar。
        exit_mode: 策略出场模式，"fixed_hold" | "threshold" | "oco" | "auto"。
        hold_days: 策略固定持有**交易日数**（仅 fixed_hold 模式下与 label 比对；
            label 的 bar 跨度会按 input_interval 换算成交易日后再比，日线下恒等）。
        input_interval: K 线周期，如 "d"、"30m"。

    Returns:
        软性告警字符串列表；无软性问题时为空列表。

    Raises:
        ValueError: 硬性 label-trade 不一致，或 exit_mode 不在支持列表内时抛出。
    """
    spec = label_spec or {}
    price_ref = str(spec.get("price_ref") or "close")
    warnings: list[str] = []

    if exit_mode == "fixed_hold":
        horizon = label_holding_horizon(spec, input_interval)
        if horizon is None:
            raise ValueError(
                f"label-trade 不一致：exit_mode=fixed_hold 需要固定持有期 label，"
                f"但 label mode={spec.get('mode')} 在 interval={input_interval} 下持有期不固定。"
            )
        # label 持有期是 bar 数，策略 hold_days 按交易日计 → 换算到同单位（天）再比对，
        # 消除分钟线下"bar 数 vs 天数"的口径错配（日线 ÷1 恒等，行为不变）。
        horizon_days = bars_to_days(horizon, input_interval)
        if int(hold_days) != horizon_days:
            raise ValueError(
                f"label-trade 不一致：label 持有期={horizon} 个 bar（={horizon_days} 个交易日"
                f"@{input_interval}），但策略 hold_days={hold_days} 个交易日。"
                f"请将 hold_days 改为 {horizon_days}，或使用 exit_mode=auto 自动对齐。"
            )
        if price_ref == "close":
            warnings.append(
                "price_ref=close 为研究口径（按收盘价成交，实盘吃不到隔夜跳空），"
                "回测撮合仍按次开盘近似，二者存在错配；"
                "若用于实盘对齐，训练请改用 price_ref=next_open / next_close / next_vwap。"
            )
        elif price_ref == "next_vwap":
            warnings.append(
                "price_ref=next_vwap 假设可按 T+1 全天均价(VWAP)成交，回测撮合已对齐；"
                "实盘需用 VWAP 算法单分批执行才能逼近该价，单笔挂单通常吃不到全天均价，"
                "请据此谨慎解读绩效。"
            )
        # next_open / next_close：可执行口径，回测撮合已按对应成交价对齐，无需告警。
    elif exit_mode == "threshold":
        warnings.append(
            "exit_mode=threshold 为信号衰减式出场（概率跌破阈值才平仓），"
            "与 label 的固定持有期不对齐，realized 持有期会偏离训练 horizon；"
            "如需与 label 精确对齐，请使用 exit_mode=fixed_hold 或 auto。"
        )
    elif exit_mode == "oco":
        label_mode = str(spec.get("mode") or "next_bar")
        if label_mode == "oco":
            # label 与策略同为 OCO：口径对齐。仅提示路径依赖的保守假设。
            warnings.append(
                "exit_mode=oco 与 OCO label 对齐：回测含「同根 bar 止盈止损先后未知 → 保守假设止损先到」"
                "及跳空按更差价成交的假设；建议以纸面交易复核触发价成交。"
            )
        else:
            # 用 OCO 出场却配非 OCO label：出场口径与 label 不一致，软性告警
            warnings.append(
                f"exit_mode=oco 为路径依赖出场（止盈/止损触发），与 label mode={label_mode} 的"
                "固定/区间持有口径不一致，realized 持有期与收益会偏离训练目标；"
                "如需精确对齐，请使用 OCO label（mode=oco）或 exit_mode=auto。"
            )
    else:
        raise ValueError(f"未知 exit_mode：{exit_mode}（应为 threshold / fixed_hold / oco / auto）")

    return warnings
