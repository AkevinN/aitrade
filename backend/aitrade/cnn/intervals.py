"""A股各输入周期下每个交易日的 bar 数（bars-per-day 单一事实源）。

供推理预热（``predictor.warmup_days``）、label↔策略持有期的单位换算（``consistency``）
等共用，避免 bars-per-day 表在后端多处重复（前端 ``utils/barInterval.ts`` 为其镜像）。

口径：A 股每个完整交易日 4 小时连续竞价 = 240 分钟，故各周期每日 bar 数固定。
半日市（少数节假日前）bar 数偏少，按"完整交易日"口径不单独处理（YAGNI）。
"""

from __future__ import annotations

import math

# 各输入周期 → 每个交易日的 bar 数。键集与训练入口白名单一致
# （api/cnn.py 的 {d,1m,5m,10m,15m,30m,60m}）。
BARS_PER_TRADING_DAY: dict[str, int] = {
    "d": 1, "60m": 4, "30m": 8, "15m": 16, "10m": 24, "5m": 48, "1m": 240,
}


def bars_per_day(interval: str) -> int:
    """返回该周期每个交易日的 bar 数；未识别周期按每日 1 根处理。

    Args:
        interval: K 线周期，如 "d"、"30m"、"1m"。

    Returns:
        每交易日 bar 数（>= 1）。未在表内的周期回退为 1（当作日线）。

    Example:
        >>> bars_per_day("30m")
        8
        >>> bars_per_day("d")
        1
    """
    return BARS_PER_TRADING_DAY.get(interval, 1)


def bars_to_days(bars: int, interval: str) -> int:
    """把"bar 跨度"向上取整换算成"交易日数"（统一按天的换算方向）。

    用于把 label 蕴含的持有期（bar 数）换成策略按交易日计的 ``hold_days``。
    向上取整（ceil）：宁可多持有一日也不少持；日线（每日 1 根）下恒等于原值。
    ``bars <= 0`` 时返回 0（表示"无持有期/不限"，与 OCO max_hold=0 语义一致）。

    Args:
        bars: 持有期的 bar 数。
        interval: K 线周期，如 "d"、"30m"。

    Returns:
        交易日数；bars<=0 时为 0，否则 >= 1。

    Example:
        >>> bars_to_days(10, "30m")   # 每日 8 根 → ceil(10/8)
        2
        >>> bars_to_days(5, "d")
        5
        >>> bars_to_days(0, "30m")
        0
    """
    if bars <= 0:
        return 0
    return max(1, math.ceil(int(bars) / bars_per_day(interval)))
