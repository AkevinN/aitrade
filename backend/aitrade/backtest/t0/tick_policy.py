"""可插拔档位策略（TickPolicy）：决定每个交易日的 (sell_tick, buy_tick)。

硬约束：**只许用截至前一交易日收盘的信息**。本模块用 ``DailyHistory`` 仅暴露"已完成的
历史交易日"——策略在每日开盘前向其询问档位，history 里绝不含当日及未来日，从接口层面
杜绝前视（design Property 1）。

提供：
- ``FixedTick``：固定值，支持非对称（默认）。
- ``VolScaledTick``：按近 N 日振幅缩放（对称）。
- ``TrendTiltTick``：按近 N 日动量做非对称倾斜。
（``ReversionCalibratedTick`` 依赖 T0Profile，定义在 profiler 相关模块。）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

from ..engine import round_to


@dataclass
class DailyBar:
    """单个已完成交易日的 OHLC（做 T 画像/档位策略用的最小日线单元）。"""

    d: date
    open: float
    high: float
    low: float
    close: float

    @property
    def range(self) -> float:
        """当日振幅 high − low（元）。"""
        return self.high - self.low


@dataclass
class DailyHistory:
    """只读累积的"已完成交易日"序列，供 TickPolicy 做无前视的近期统计。

    由策略在每个交易日收盘后 append 当日 bar；询问当日档位时 history 仅含更早的日，
    因此所有 ``mean_range`` / ``momentum`` 天然无前视。
    """

    bars: list[DailyBar] = field(default_factory=list)

    def append(self, bar: DailyBar) -> None:
        """追加一个已完成交易日。"""
        self.bars.append(bar)

    def __len__(self) -> int:
        return len(self.bars)

    def mean_range(self, n: int) -> float | None:
        """近 n 个已完成交易日的平均振幅（high−low）。

        Args:
            n: 回看交易日数。

        Returns:
            平均振幅（元）；历史不足 n 日时返回 None。
        """
        if len(self.bars) < n:
            return None
        window = self.bars[-n:]
        return sum(b.range for b in window) / n

    def momentum(self, n: int) -> float | None:
        """近 n 日动量：``close[-1] / close[-1-n] − 1``。

        Args:
            n: 回看交易日数。

        Returns:
            收益率；历史不足 n+1 日或基准价非正时返回 None。
        """
        if len(self.bars) < n + 1:
            return None
        base = self.bars[-1 - n].close
        if base <= 0:
            return None
        return self.bars[-1].close / base - 1.0


@dataclass
class TickContext:
    """挂单决策的无前视上下文：只含「截至昨收 + 今开」的信息。

    自定义算法可基于 ``hist``/``open`` 计算；``signals`` 为 point-in-time 信号值。
    **绝不含当日 high/low/close 或未来 bar**，从结构上杜绝前视（design Property 1/5）。

    Attributes:
        day: 当前交易日。
        open: 今日开盘价（元）——9:25 集合竞价后可知。
        prev_close: 上一交易日收盘价（元）。
        hist: 截至昨日（已完成日）的日线累积。
        signals: 该标的截至昨收的信号值字典（因子或自定义算法皆可，point-in-time）。
    """

    day: date
    open: float
    prev_close: float
    hist: DailyHistory
    signals: dict[str, float | None] = field(default_factory=dict)

    @property
    def gap(self) -> float:
        """今开相对昨收的跳空 = ``open/prev_close − 1``；昨收非正时退化为 0。"""
        return self.open / self.prev_close - 1.0 if self.prev_close else 0.0


@runtime_checkable
class TickPolicy(Protocol):
    """档位策略协议：给定无前视上下文 ``TickContext``，返回 (sell_tick, buy_tick)。"""

    def ticks_for(self, ctx: TickContext) -> tuple[float, float]:
        """返回某交易日的卖/买档位（元）。实现只许读 ctx（无前视）。"""
        ...


@dataclass
class FixedTick:
    """固定档位（默认），支持非对称 sell/buy。

    Args:
        sell_tick: 卖单相对开盘价上挂的价差（元）。
        buy_tick: 买单相对开盘价下挂的价差（元）。
    """

    sell_tick: float = 0.02
    buy_tick: float = 0.02

    def ticks_for(self, ctx: TickContext) -> tuple[float, float]:
        """忽略 ctx，恒返回配置的固定档位。"""
        return (self.sell_tick, self.buy_tick)


@dataclass
class VolScaledTick:
    """按近 N 日平均振幅缩放的对称档位：``tick = round_to(k × mean_range(N))``。

    历史不足 N 日时回退到 ``fallback``，使回测起点附近不致无档位。

    Args:
        k: 振幅缩放系数。
        n: 振幅回看交易日数；默认 20。
        pricetick: 最小价位，用于对齐；默认 0.01。
        fallback: 历史不足时的回退档位（元）；默认 0.02。
    """

    k: float
    n: int = 20
    pricetick: float = 0.01
    fallback: float = 0.02

    def ticks_for(self, ctx: TickContext) -> tuple[float, float]:
        """按近 N 日均振幅缩放出对称档位；历史不足回退 fallback。"""
        atr = ctx.hist.mean_range(self.n)
        if atr is None:
            return (self.fallback, self.fallback)
        t = max(self.pricetick, round_to(self.k * atr, self.pricetick))
        return (t, t)


@dataclass
class TrendTiltTick:
    """按近 N 日动量做非对称倾斜的档位（顺势：上涨→买近卖远）。

    上涨（动量>0）：``sell_tick = base + tilt``、``buy_tick = base − tilt``（买近卖远）；
    下跌（动量<0）：对称反向；动量不可得时退化为对称 base。所有档位夹到 ≥ pricetick。

    Args:
        base: 基准档位（元）。
        tilt: 倾斜幅度（元）。
        n: 动量回看交易日数；默认 5。
        pricetick: 最小价位下限；默认 0.01。
    """

    base: float
    tilt: float
    n: int = 5
    pricetick: float = 0.01

    def ticks_for(self, ctx: TickContext) -> tuple[float, float]:
        """按近 N 日动量符号做非对称倾斜。"""
        mom = ctx.hist.momentum(self.n)
        if mom is None or mom == 0:
            return (max(self.pricetick, self.base), max(self.pricetick, self.base))
        if mom > 0:
            sell, buy = self.base + self.tilt, self.base - self.tilt
        else:
            sell, buy = self.base - self.tilt, self.base + self.tilt
        return (max(self.pricetick, sell), max(self.pricetick, buy))


@dataclass
class Rule:
    """一条挂单规则：``condition``/``ticks`` 均为任意可调用（只读 ctx）。

    自定义算法是一等公民——可直接在回调里基于 ``ctx.hist``/``ctx.open``/``ctx.signals`` 计算，
    无需先注册为命名信号。

    Args:
        name: 规则名（报告/调试用）。
        condition: ``(TickContext) -> bool``，触发判定；只读 ctx（无前视）。
        ticks: ``(TickContext) -> (sell_tick, buy_tick)``，命中时给出卖/买档（元）。
    """

    name: str
    condition: Callable[[TickContext], bool]
    ticks: Callable[[TickContext], tuple[float, float]]


@dataclass
class ConditionalTickPolicy:
    """条件驱动的档位策略（``TickPolicy`` 实现）：按规则顺序匹配，首个命中即返回其档位。

    无任何规则命中时返回 ``default``。所有返回档位经 ``round_to`` 对齐最小价位、并夹到
    ≥ 一个最小价位，避免非法报价（design Property 4）。

    Args:
        rules: 有序规则列表；逐条评估 ``condition``，命中即用其 ``ticks``。
        default: 无规则命中时的兜底档位 ``(sell_tick, buy_tick)``（元）。
        pricetick: 最小价位，用于对齐与下限；默认 0.01。

    Example:
        >>> p = ConditionalTickPolicy(rules=gap_rules(), default=(0.03, 0.02))
        >>> p.ticks_for(ctx)  # 高开→(0.07,0.01) / 低开→(0.09,0.01) / 平开→default
    """

    rules: list[Rule]
    default: tuple[float, float]
    pricetick: float = 0.01
    signal_names: tuple[str, ...] = ()   # 策略据此为 ctx.signals 预取的命名信号

    def ticks_for(self, ctx: TickContext) -> tuple[float, float]:
        """逐条评估规则，返回首个命中规则的档位；无匹配返回 default。"""
        for rule in self.rules:
            if rule.condition(ctx):
                return self._round(rule.ticks(ctx))
        return self._round(self.default)

    def _round(self, ticks: tuple[float, float]) -> tuple[float, float]:
        """把 (sell, buy) 对齐最小价位并夹到 ≥ pricetick。"""
        pt = self.pricetick
        sell, buy = ticks
        return (max(pt, round_to(sell, pt)), max(pt, round_to(buy, pt)))


def gap_rules(thresh: float = 0.003,
              up: tuple[float, float] = (0.07, 0.01),
              down: tuple[float, float] = (0.09, 0.01)) -> list[Rule]:
    """构造「高开/低开」两条跳空规则（平开交由 ``ConditionalTickPolicy.default`` 兜底）。

    高开 = ``gap > thresh``、低开 = ``gap < −thresh``；判定只用 ``ctx.gap``（今开/昨收），无前视。

    Args:
        thresh: 跳空阈值（如 0.003 = 0.3%）。
        up: 高开档位 ``(sell_tick, buy_tick)``（元）。
        down: 低开档位（元）。

    Returns:
        ``[Rule(高开), Rule(低开)]``；平开（``|gap| ≤ thresh``）不命中、走 default。
    """
    return [
        Rule("高开", lambda c, t=thresh: c.gap > t, lambda c, v=up: v),
        Rule("低开", lambda c, t=thresh: c.gap < -t, lambda c, v=down: v),
    ]
