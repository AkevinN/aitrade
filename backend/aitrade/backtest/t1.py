"""A股 T+1 制度的单一事实源（single source of truth）。

把"当日买入的股份当日不可卖出"这条判定收敛到一个纯函数 :func:`is_t1_locked`，
供回测撮合引擎（``engine._t_plus1_locked``）、CNN 策略出场预检（``CNNSignalStrategy``）
共用同一份逻辑，避免各路径各自实现导致豁免（T+0 品种如可转债）口径漂移。

实盘侧（live/）接入真实下单时，应在本模块新增"可卖昨仓量"原语（如
``sellable_volume(total, today_buy, ...)``）并复用，使回测、实盘、标签出场三条路径
共享同一 T+1 事实源。本期范围（P0+P2）仅覆盖回测撮合与 CNN 策略，故暂不引入实盘原语。
"""

from __future__ import annotations

from datetime import date


def is_t1_locked(
    vt_symbol: str,
    buy_dates: dict[str, date],
    today: date | None,
    *,
    enabled: bool,
    exempt: set[str],
) -> bool:
    """判断标的当日买入的股份是否受 T+1 限制（当根/当日不可卖出）。

    三个条件同时成立才锁定：

    1. ``enabled`` 总开关为 True；
    2. ``vt_symbol`` 不在 ``exempt`` 豁免集合中（可转债等 T+0 品种恒不锁）；
    3. ``buy_dates`` 记录的该标的最近买入日 == ``today``（即"今天买的"）。

    判定以"日历日"（``date``）为粒度，与 bar 时间戳粒度无关——故分钟线下当日内多根
    bar 同样整天锁死，不存在"日内买了又卖绕过 T+1"的漏洞。

    Args:
        vt_symbol: 合约代码，如 ``"000001.SZSE"``。
        buy_dates: 标的 → 最近一次买入日期 的映射（成交结算时写入）。
        today: 当前交易日；为 None（引擎尚未推进到任何 bar）时一律视为不锁定。
        enabled: T+1 总开关；False 时恒返回 False（即 T+0，如显式关闭 T+1 的回测）。
        exempt: T+0 豁免标的集合；集合内标的恒返回 False。

    Returns:
        True 表示当日买入不可当日卖出；False 表示允许卖出。

    Example:
        >>> from datetime import date
        >>> bd = {"600000.SSE": date(2025, 1, 2)}
        >>> is_t1_locked("600000.SSE", bd, date(2025, 1, 2), enabled=True, exempt=set())
        True
        >>> is_t1_locked("600000.SSE", bd, date(2025, 1, 3), enabled=True, exempt=set())
        False
        >>> is_t1_locked("600000.SSE", bd, date(2025, 1, 2), enabled=False, exempt=set())
        False
    """
    if not enabled or vt_symbol in exempt:
        return False
    if today is None:
        return False
    return buy_dates.get(vt_symbol) == today
