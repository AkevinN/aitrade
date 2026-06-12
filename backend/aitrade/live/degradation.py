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
    """是否允许交易。任一异常 → 暂停（返回 False + 原因）。"""
    if not healthy:
        return False, "系统不健康（心跳超时/服务异常），降级暂停交易"
    if reconcile_blocked:
        return False, "对账差异过大，降级暂停交易"
    if last_data_time is None:
        return False, "无行情数据，降级暂停交易"
    if not is_data_fresh(last_data_time, now, max_staleness_seconds):
        return False, "行情数据过期，降级暂停（不在信息不全时下单）"
    return True, ""
