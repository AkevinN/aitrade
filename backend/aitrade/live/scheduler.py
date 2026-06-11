"""
调度触发判定（决策时刻统一 / 盘中监控）：纯函数，便于确定性测试。

两种调度形态：

- `due_slots`（日频计划）：交易日 + 已到达用户配置的唤醒时刻 + 当日该时刻未触发 → 触发。
- `due_bar_slot`（日内计划，监控模式）：交易日 + Bar_Grid 上已有 `<= now` 的收盘时刻 +
  该时刻未完成 → 触发。只返回**最近一根**——监控只决策当下最新已收盘 bar，停机期间
  错过的中间 bar 不补决策（`as_of=now` 的语义决定无法重构过去时刻的实时决策；
  历史验证走回测）。

注意：这里的「时刻」是**调度唤醒/网格时刻**（每交易日的本地 HH:MM），与「决策 bar」解耦——
触发后由 orchestrator 据 `DecisionInstant(as_of=now, bar_freq)` 选取 close_time<=as_of 的
Decision_Bar（无前视由 as_of 截断结构性保证）。
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Optional

from .calendar import TradingCalendar
from .decision_instant import bar_close_grid


def due_slots(
    now: datetime,
    trigger_times: list[time],
    calendar: TradingCalendar,
    triggered_slots: set[str],
) -> list[str]:
    """返回当前应触发但当日尚未触发的唤醒时刻 slot（"HH:MM" 升序）。

    - 仅交易日返回非空；
    - 仅返回 `now.time() >= t` 已到达的时刻；
    - 已在 `triggered_slots`（当日已触发集合，"HH:MM"）中的时刻不再返回（同时刻幂等）。

    `triggered_slots` 中的旧状态哨兵（如 "*" 表示整日已触发）由调用方在传入前处理，
    本函数只按字面比较 slot 字符串。
    """
    if not calendar.is_trading_day(now.date()):
        return []
    out: list[str] = []
    for t in trigger_times:
        slot = t.strftime("%H:%M")
        if now.time() >= t and slot not in triggered_slots:
            out.append(slot)
    return sorted(set(out))


def due_bar_slot(
    now: datetime,
    bar_freq: str,
    calendar: TradingCalendar,
    triggered_slots: set[str],
) -> Optional[str]:
    """监控模式判定：当日 Bar_Grid 上 `<= now` 的**最近**收盘时刻 slot（"HH:MM"）。

    - 非交易日 / 开盘前（当日网格上尚无 `<= now` 的收盘时刻）→ None；
    - 该时刻已在 `triggered_slots`（当日已完成集合）中 → None（逐 bar 幂等）；
    - 否则返回该时刻的 "HH:MM"。

    只返回最近一根（不补中间错过的 bar）。调用方在决策结果确认
    `decision_bar_dt >= 该时刻` 后才标记完成——本地数据未跟上时不标记，下一 tick 重试。
    """
    if not calendar.is_trading_day(now.date()):
        return None
    due = [t for t in bar_close_grid(bar_freq) if now.time() >= t]
    if not due:
        return None
    slot = due[-1].strftime("%H:%M")
    return None if slot in triggered_slots else slot
