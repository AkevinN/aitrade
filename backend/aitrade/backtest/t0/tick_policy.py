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


@runtime_checkable
class TickPolicy(Protocol):
    """档位策略协议：给定交易日与"截至前一日"的历史，返回 (sell_tick, buy_tick)。"""

    def ticks_for(self, day: date, hist: DailyHistory) -> tuple[float, float]:
        """返回某交易日的卖/买档位（元）。实现只许读 hist（无前视）。"""
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

    def ticks_for(self, day: date, hist: DailyHistory) -> tuple[float, float]:
        """忽略 hist，恒返回配置的固定档位。"""
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

    def ticks_for(self, day: date, hist: DailyHistory) -> tuple[float, float]:
        """按近 N 日均振幅缩放出对称档位；历史不足回退 fallback。"""
        atr = hist.mean_range(self.n)
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

    def ticks_for(self, day: date, hist: DailyHistory) -> tuple[float, float]:
        """按近 N 日动量符号做非对称倾斜。"""
        mom = hist.momentum(self.n)
        if mom is None or mom == 0:
            return (max(self.pricetick, self.base), max(self.pricetick, self.base))
        if mom > 0:
            sell, buy = self.base + self.tilt, self.base - self.tilt
        else:
            sell, buy = self.base - self.tilt, self.base + self.tilt
        return (max(self.pricetick, sell), max(self.pricetick, buy))
