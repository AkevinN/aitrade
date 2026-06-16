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
    """日频计划调度判定：返回当前应触发但当日尚未触发的唤醒时刻 slot。

    判定规则：仅交易日返回非空；仅返回 `now.time() >= t` 已到达的时刻；
    已在 `triggered_slots` 中的时刻不再返回（同时刻幂等）。纯函数、无 I/O。
    `triggered_slots` 中的旧状态哨兵（如 "*" 表示整日已触发）由调用方在传入前
    处理，本函数只按字面比较 slot 字符串。

    Args:
        now: 当前时刻；其 date 用于交易日判定，time 用于"已到达"比较。
        trigger_times: 用户配置的每日唤醒时刻列表（本地 HH:MM）。
        calendar: 交易日历，用于判定 now 当天是否交易日。
        triggered_slots: 当日已触发时刻集合（"HH:MM" 字符串），用于幂等去重。

    Returns:
        应触发的 slot 列表（"HH:MM"，去重升序）；非交易日或无到点时刻时返回空列表。
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
    """日内监控模式调度判定：返回当日 Bar_Grid 上 `<= now` 的最近收盘时刻 slot。

    只返回最近一根（不补中间错过的 bar），因 `as_of=now` 无法重构过去时刻的实时
    决策。调用方在决策结果确认 `decision_bar_dt >= 该时刻` 后才标记完成——本地数据
    未跟上时不标记，下一 tick 重试。纯函数、无 I/O。

    Args:
        now: 当前时刻；其 date 用于交易日判定，time 用于筛选已收盘的网格时刻。
        bar_freq: K 线周期（如 "1d"/"30m"），用于经 bar_close_grid 生成当日收盘网格。
        calendar: 交易日历，用于判定 now 当天是否交易日。
        triggered_slots: 当日已完成时刻集合（"HH:MM"），命中即跳过（逐 bar 幂等）。

    Returns:
        最近一根已收盘时刻的 "HH:MM"；非交易日、开盘前（网格上无 `<= now` 的收盘
        时刻）、或该时刻已在 triggered_slots 中时返回 None。
    """
    if not calendar.is_trading_day(now.date()):
        return None
    due = [t for t in bar_close_grid(bar_freq) if now.time() >= t]
    if not due:
        return None
    slot = due[-1].strftime("%H:%M")
    return None if slot in triggered_slots else slot
