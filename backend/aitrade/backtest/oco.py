"""
OCO（One-Cancels-the-Other）止盈止损出场原语（迭代 4）。

这是「label 出场假设 ≡ 策略真实出场 ≡ 撮合成交」红线在 OCO 场景的落点：
- ``check_oco_trigger``：给定一根 bar 的 open/high/low 与止盈/止损价位，判断是否触发、
  以及保守成交价。**同一根 bar 两边都可能触发、先后未知时，保守假设止损先到**。
- ``simulate_oco_exit``：从建仓后逐根扫描，复用 ``check_oco_trigger`` 得到出场点/价/原因，
  既被「数据集 OCO label 生成」调用，也用于校验「策略 OCO 出场」是否一致。

保守口径（与实盘风险方向一致，宁可低估收益、如实暴露下行）：
- 止损（卖）：跳空低开（open ≤ 止损价）按 open 成交（更差）；否则按止损价。
- 止盈（卖）：保守只按挂单止盈价成交，不享受跳空高开带来的额外收益。
"""

from __future__ import annotations

from typing import Optional, Sequence


def oco_levels(entry_price: float, take_profit: float, stop_loss: float) -> tuple[float, float]:
    """由建仓价与止盈/止损比例算出绝对价位。

    take_profit / stop_loss 为正的收益率（如 0.03 = 3%）。
    返回 (止盈价, 止损价)。
    """
    if entry_price <= 0:
        raise ValueError("entry_price 必须为正")
    if take_profit <= 0 or stop_loss <= 0:
        raise ValueError("take_profit / stop_loss 必须为正比例（如 0.03 表示 3%）")
    tp_price = entry_price * (1.0 + take_profit)
    sl_price = entry_price * (1.0 - stop_loss)
    return tp_price, sl_price


def check_oco_trigger(
    open_: float,
    high: float,
    low: float,
    tp_price: float,
    sl_price: float,
    *,
    stop_first: bool = True,
) -> Optional[tuple[str, float]]:
    """判断一根 bar 是否触发 OCO，返回 (原因, 保守成交价) 或 None。

    原因为 ``"sl"``（止损）或 ``"tp"``（止盈）。
    """
    hit_sl = low <= sl_price
    hit_tp = high >= tp_price

    # 同一根 bar 两侧都可能触发：日内先后未知 → 保守假设止损先到
    if hit_sl and hit_tp and stop_first:
        return ("sl", min(sl_price, open_))
    if hit_sl:
        # 跳空低开则按更差的开盘价成交
        return ("sl", min(sl_price, open_))
    if hit_tp:
        # 止盈保守按挂单价成交，不享受跳空高开
        return ("tp", tp_price)
    return None


def simulate_oco_exit(
    entry_index: int,
    entry_price: float,
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    take_profit: float,
    stop_loss: float,
    max_hold: int,
    *,
    stop_first: bool = True,
) -> Optional[dict]:
    """从建仓后模拟 OCO 出场（含到期时间止损兜底）。

    扫描区间为建仓后第 1 根到第 ``max_hold`` 根 bar（即 ``entry_index+1 .. entry_index+max_hold``，
    对应 A 股 T+1：建仓当根不可卖出）。任一根触发即在该根按保守价出场；
    全程未触发则在 **第 ``max_hold+1`` 根开盘** 按时间止损平仓（与 fixed_hold 的次开盘口径一致）。

    数据不足以走完扫描 + 兜底（``entry_index+max_hold+1`` 越界）时返回 None（样本应丢弃）。

    返回 dict：``{exit_index, exit_price, reason("sl"|"tp"|"time"), ret}``。
    """
    if max_hold < 1:
        raise ValueError("max_hold 必须 >= 1")
    n = len(opens)
    # 需要兜底平仓那一根存在，才能形成完整、可执行的 label
    fallback_index = entry_index + max_hold + 1
    if entry_index < 0 or fallback_index >= n:
        return None

    tp_price, sl_price = oco_levels(entry_price, take_profit, stop_loss)

    for j in range(entry_index + 1, entry_index + max_hold + 1):
        trig = check_oco_trigger(
            float(opens[j]), float(highs[j]), float(lows[j]),
            tp_price, sl_price, stop_first=stop_first,
        )
        if trig is not None:
            reason, fill = trig
            return {
                "exit_index": j,
                "exit_price": fill,
                "reason": reason,
                "ret": (fill - entry_price) / entry_price,
            }

    # 到期时间止损：下一根开盘平仓
    fill = float(opens[fallback_index])
    return {
        "exit_index": fallback_index,
        "exit_price": fill,
        "reason": "time",
        "ret": (fill - entry_price) / entry_price,
    }
