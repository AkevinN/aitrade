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
        self._trading_days = set(trading_days) if trading_days is not None else None
        self._holidays = set(holidays) if holidays else set()

    def is_trading_day(self, d: date) -> bool:
        if self._trading_days is not None:
            return d in self._trading_days
        # 工作日且非节假日
        return d.weekday() < 5 and d not in self._holidays
