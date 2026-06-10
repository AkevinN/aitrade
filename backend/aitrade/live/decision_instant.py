"""
决策时刻统一（Decision Instant Unification）核心原语。

把实盘决策的单元从「天」升级为「时刻」：决策 = `DecisionInstant(as_of, bar_freq)`
的函数，「天」是 `bar_freq="1d"` 且 `as_of=当日收盘` 的特例；分钟频（盘中监控）
是同一抽象在日内 bar 上的自然延伸。

无前视红线（结构性保证）：取数只见 `close_time <= as_of` 的 bar，任何 `as_of`
都不可能读到未来 bar。分钟 bar 复用 AlphaLab 既定事实——**end-labeled**
（`datetime` 即 bar 收盘时刻，派生 metadata `ts_convention: "end"`），故日内
close_time 就是 `datetime` 本身。

本模块只提供纯函数 / 不可变值对象，不做任何 I/O，便于确定性测试与跨层复用
（orchestrator 取价/取信号、signal_service 幂等键、plan_scheduler 触发构造与
监控模式 bar 网格）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Optional

import polars as pl

# A 股默认市场收盘时刻（可后续配置化）。任意频率下，日期 D 的最后一根 bar
# 的收盘时刻 = D 的该时刻。
SESSION_CLOSE: time = time(15, 0)

# A 股连续竞价时段（与 AlphaLab._CN_EQUITY_SESSIONS 同口径）。
SESSIONS: tuple[tuple[time, time], ...] = (
    (time(9, 30), time(11, 30)),
    (time(13, 0), time(15, 0)),
)

# 决策域支持的 bar 频率（间隔锁定的单一事实来源；models 层从这里 import）。
INTRADAY_BAR_FREQS: tuple[str, ...] = ("1m", "5m", "10m", "15m", "30m", "60m")
SUPPORTED_BAR_FREQS: tuple[str, ...] = ("1d", *INTRADAY_BAR_FREQS)


@dataclass(frozen=True)
class DecisionInstant:
    """一次决策的唯一单元。

    - `as_of`：决策时刻；仅 `close_time <= as_of` 的 bar 对该决策可见（无前视截断点）。
    - `bar_freq`：决策所用 bar 频率；`"1d"` 即日频（天 = 时刻特例），分钟频
      （`INTRADAY_BAR_FREQS`）即盘中监控的逐 bar 决策。
    """

    as_of: datetime
    bar_freq: str = "1d"


def bar_freq_of_interval(interval: str) -> str:
    """模型/行情周期（AlphaLab 口径）→ 决策域 bar_freq：`"d" -> "1d"`，分钟恒等。"""
    if interval == "d":
        return "1d"
    if interval in INTRADAY_BAR_FREQS:
        return interval
    raise ValueError(f"不受支持的行情周期: {interval!r}（支持 d/{'/'.join(INTRADAY_BAR_FREQS)}）")


def interval_of_bar_freq(bar_freq: str) -> str:
    """决策域 bar_freq → 行情周期（AlphaLab 口径）：`"1d" -> "d"`，分钟恒等。"""
    if bar_freq == "1d":
        return "d"
    if bar_freq in INTRADAY_BAR_FREQS:
        return bar_freq
    raise ValueError(f"不受支持的 bar_freq: {bar_freq!r}（支持 {'/'.join(SUPPORTED_BAR_FREQS)}）")


def _freq_minutes(bar_freq: str) -> int:
    """分钟频字符串 → 分钟数（仅 INTRADAY_BAR_FREQS）。"""
    return int(bar_freq[:-1])


def bar_close_grid(bar_freq: str) -> tuple[time, ...]:
    """该频率在 A 股交易时段内的 bar 收盘时刻网格（end-labeled，升序）。

    - `"1d"` → `(15:00,)`（一天一根 bar）。
    - 分钟频 → 各 session 内 `open + k*freq` 的收盘序列；末桶并入时段收盘
      （与 AlphaLab `_session_bucket_end` 的 ceil 语义一致），必含 11:30 与 15:00。
    """
    if bar_freq == "1d":
        return (SESSION_CLOSE,)
    if bar_freq not in INTRADAY_BAR_FREQS:
        raise ValueError(f"不受支持的 bar_freq: {bar_freq!r}（支持 {'/'.join(SUPPORTED_BAR_FREQS)}）")
    minutes = _freq_minutes(bar_freq)
    closes: list[time] = []
    for open_t, close_t in SESSIONS:
        open_min = open_t.hour * 60 + open_t.minute
        close_min = close_t.hour * 60 + close_t.minute
        t = open_min
        while t < close_min:
            t = min(t + minutes, close_min)
            closes.append(time(t // 60, t % 60))
    return tuple(closes)


def session_close(d: date, bar_freq: str) -> datetime:
    """日期 d 当日**最后一根** `bar_freq` bar 的收盘时刻 = d 的 `SESSION_CLOSE`。

    对所有受支持频率成立（`1d` 与分钟频的末 bar 都收盘于 15:00）。
    """
    if bar_freq not in SUPPORTED_BAR_FREQS:
        raise ValueError(f"不受支持的 bar_freq: {bar_freq!r}（支持 {'/'.join(SUPPORTED_BAR_FREQS)}）")
    return datetime.combine(d, SESSION_CLOSE)


def _close_time_expr(bar_freq: str) -> pl.Expr:
    """构造「bar 收盘时刻」表达式（用于 as-of 截断过滤）。

    - `1d`：把 bar 的 `datetime` 截断到当日零点再偏移到 SESSION_CLOSE，得到该日真实收盘时刻，
      与 bar 原始 datetime 的具体时分（0:00 / 15:00 / 16:00 等约定）无关，结果稳定。
    - 分钟频：AlphaLab 分钟 bar 为 end-labeled（`ts_convention: "end"`），
      `datetime` 本身即收盘时刻。
    """
    if bar_freq == "1d":
        offset = f"{SESSION_CLOSE.hour}h{SESSION_CLOSE.minute}m"
        return pl.col("datetime").dt.truncate("1d").dt.offset_by(offset)
    if bar_freq in INTRADAY_BAR_FREQS:
        return pl.col("datetime")
    raise ValueError(f"不受支持的 bar_freq: {bar_freq!r}（支持 {'/'.join(SUPPORTED_BAR_FREQS)}）")


def select_decision_bar(
    frame: Optional[pl.DataFrame], instant: DecisionInstant
) -> Optional[pl.DataFrame]:
    """取 `close_time <= as_of` 的**最后一根已收盘 bar**（无前视结构性保证）。

    返回单行 DataFrame；无任何已收盘 bar（如 `as_of` 早于全部 bar 收盘）时返回 None，
    由调用方译为「as_of 之前无已收盘行情」错误，绝不退回未收盘/未来 bar。
    """
    if frame is None or frame.is_empty() or "datetime" not in frame.columns:
        return None
    closed = frame.filter(_close_time_expr(instant.bar_freq) <= pl.lit(instant.as_of))
    if closed.is_empty():
        return None
    return closed.sort("datetime").tail(1)


def decision_bar_datetime(bar_row: pl.DataFrame) -> datetime:
    """从 `select_decision_bar` 返回的单行 DataFrame 取该 bar 的 datetime。"""
    return bar_row["datetime"][0]


def make_signal_id(
    decision_bar_dt: datetime, bar_freq: str, scheme: str, model_version: str = ""
) -> str:
    """幂等键：由 Decision_Bar + bar_freq + scheme + model_version 生成。

    - `1d`：渲染为 `YYYY-MM-DD:scheme[@version]`，与历史 `Decision.make_signal_id`
      逐位一致（旧决策文件不孤立、回测/历史口径不变）。
    - 日内：渲染为含 ISO 分钟时刻的串（如 `2026-06-08T10:30:scheme[@version]`），
      使不同 Decision_Bar 产出不同 signal_id。
    """
    tag = f"@{model_version}" if model_version else ""
    if bar_freq == "1d":
        key = decision_bar_dt.date().isoformat()
    else:
        key = decision_bar_dt.isoformat(timespec="minutes")
    return f"{key}:{scheme}{tag}"
