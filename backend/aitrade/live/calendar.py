"""
交易日历（迭代 6）：判断是否交易日，用于调度与信号触发。

默认实现：工作日（周一至周五）且不在 holidays 集合内即视为交易日；
若显式传入 trading_days 集合，则以集合为准（可由数据源 get_trade_calendar 填充）。
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Optional


class TradingCalendar:
    """简单交易日历。trading_days 优先；否则按工作日 + holidays 判断。"""

    def __init__(
        self,
        trading_days: Optional[Iterable[date]] = None,
        holidays: Optional[Iterable[date]] = None,
    ) -> None:
        """初始化交易日历。

        Args:
            trading_days: 精确交易日集合（如从数据源获取）；传入时以集合为准，忽略 holidays。
            holidays:     节假日集合，仅 trading_days 为 None 时生效；默认空（即工作日均为交易日）。
        """
        self._trading_days = set(trading_days) if trading_days is not None else None
        self._holidays = set(holidays) if holidays else set()

    def is_trading_day(self, d: date) -> bool:
        """判断指定日期是否为交易日。

        Args:
            d: 待判断的日期。

        Returns:
            若传入了精确 trading_days 集合则直接查表；否则按"工作日且非 holidays"判断。
        """
        if self._trading_days is not None:
            return d in self._trading_days
        # 工作日且非节假日
        return d.weekday() < 5 and d not in self._holidays
