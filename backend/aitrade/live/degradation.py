"""
优雅降级判定（迭代 10）：降级优先于猜测——信息不全/系统异常时暂停交易而非乱下单。

综合 行情新鲜度 + 健康状态 + 对账阻断 给出是否允许交易。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional


def is_data_fresh(last_data_time: Optional[datetime], now: datetime, max_staleness_seconds: float) -> bool:
    """判断行情数据是否在容忍陈旧度内（新鲜）。

    Args:
        last_data_time:        最近一次行情更新时刻；None 表示从未收到行情，直接视为过期。
        now:                   当前时刻参照。
        max_staleness_seconds: 最大允许陈旧秒数。

    Returns:
        True 表示数据新鲜（差值 ≤ 阈值），False 表示过期或 last_data_time 为 None。
    """
    if last_data_time is None:
        return False
    return (now - last_data_time).total_seconds() <= max_staleness_seconds


def decide_trading(
    *,
    now: datetime,
    last_data_time: Optional[datetime],
    max_staleness_seconds: float,
    healthy: bool = True,
    reconcile_blocked: bool = False,
) -> tuple[bool, str]:
    """综合健康/对账/行情新鲜度判定当前是否允许交易（降级优先于猜测）。

    按 系统健康 → 对账阻断 → 有无行情 → 行情新鲜度 的顺序短路检查，
    任一异常即暂停交易并给出中文原因；全部通过才放行。

    Args:
        now: 当前时刻参照。
        last_data_time: 最近一次行情更新时刻；None 表示从未收到行情。
        max_staleness_seconds: 最大允许行情陈旧秒数（传给 is_data_fresh）。
        healthy: 系统健康标志；False（心跳超时/服务异常）即暂停。默认 True。
        reconcile_blocked: 对账阻断标志；True（对账差异过大）即暂停。默认 False。

    Returns:
        (是否允许交易, 原因)。允许时为 (True, "")；
        暂停时为 (False, 中文原因文案)，原因对应首个触发的异常条件。
    """
    if not healthy:
        return False, "系统不健康（心跳超时/服务异常），降级暂停交易"
    if reconcile_blocked:
        return False, "对账差异过大，降级暂停交易"
    if last_data_time is None:
        return False, "无行情数据，降级暂停交易"
    if not is_data_fresh(last_data_time, now, max_staleness_seconds):
        return False, "行情数据过期，降级暂停（不在信息不全时下单）"
    return True, ""
