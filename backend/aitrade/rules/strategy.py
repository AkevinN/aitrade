"""
通用轮动策略：RebalancingTopKStrategy。

在 EquityDemoStrategy 基础上叠加调仓日历门控：
- 仅在调仓日（D/W/M 频率的首根 bar）执行截面轮动；
- 非调仓日只做 holding_days 自增（维持最短持有约束正确性），不发新单；
- 调仓日先撤全部未成交限价单，再调用父类 on_bars 执行轮动逻辑。
"""

from __future__ import annotations

from datetime import date as _date

from ..alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy
from ..backtest.types import BarData


class RebalancingTopKStrategy(EquityDemoStrategy):
    """TopK 轮动 + 调仓日历门控：仅在调仓日执行截面轮动，其余日子持仓不动。

    Parameters
    ----------
    rebalance_freq : str
        调仓频率。``"D"`` 每根 bar 都调仓（退化为父类行为）；
        ``"W"`` 每个 ISO 周的第一根 bar；``"M"`` 每个自然月的第一根 bar。
    其余参数同 EquityDemoStrategy（top_k / n_drop / min_days / ...）。
    """

    rebalance_freq: str = "W"  # 类属性声明使 setting 可注入

    def on_init(self) -> None:
        """初始化：在父类基础上记录上一根 bar 的日期（用于调仓日判定）。"""
        super().on_init()
        # 上一根 bar 对应的 date；None 表示尚未处理任何 bar（首根视为调仓日）
        self._last_bar_date: _date | None = None

    def _is_rebalance_day(self, current_date: _date) -> bool:
        """判定当前 bar 是否为调仓日。

        D  → 始终是调仓日；
        W  → current_date 所在 ISO 周编号与上一根 bar 不同（或首根 bar）；
        M  → current_date 的自然月与上一根 bar 不同（或首根 bar）。
        """
        if self.rebalance_freq == "D":
            return True

        if self._last_bar_date is None:
            # 首根 bar 视为调仓日
            return True

        if self.rebalance_freq == "W":
            # ISO 周：(year, week) 对发生变化即进入新一周
            return (current_date.isocalendar()[:2]
                    != self._last_bar_date.isocalendar()[:2])

        if self.rebalance_freq == "M":
            # 自然月：(year, month) 对发生变化即进入新一月
            return ((current_date.year, current_date.month)
                    != (self._last_bar_date.year, self._last_bar_date.month))

        # 未知频率退化为每日调仓
        return True

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K 线回调：调仓日历门控逻辑入口。

        非调仓日：
            只对当前有持仓的标的执行 holding_days 自增，然后直接返回。
            这里复制了父类 on_bars 第 38-41 行的自增逻辑（来源：
            equity_demo_strategy.py:38-41），目的是保证 min_days 约束
            在跳过调仓的日子里不因缺少自增而失真。不修改父类。

        调仓日：
            先撤掉本策略遗留的全部未成交限价单（BaseStrategy.cancel_all），
            再调 super().on_bars(bars) 执行完整截面轮动（父类内部会再次
            触发 holding_days 自增 + execute_trading）。
        """
        # 取引擎当前时间轴日期
        current_dt = self.strategy_engine.datetime
        current_date: _date = current_dt.date() if current_dt else _date.today()

        is_rebalance = self._is_rebalance_day(current_date)

        # 更新上一根 bar 日期（须在判定后更新，下一次调用时才能对比）
        self._last_bar_date = current_date

        if not is_rebalance:
            # 非调仓日：仅自增 holding_days，不发任何新单
            # 来源：equity_demo_strategy.py:38-41（不改父类，此处局部复制）
            pos_symbols: list[str] = [
                vt_symbol
                for vt_symbol, pos in self.pos_data.items()
                if pos
            ]
            for vt_symbol in pos_symbols:
                self.holding_days[vt_symbol] += 1
            return

        # 调仓日：先撤全部未成交限价单，再执行完整截面轮动
        # cancel_all 定义于 BaseStrategy（strategy.py:133-136），
        # 遍历 active_orderids 逐一调 cancel_order → engine.cancel_order。
        # 防御性显式撤单：父类 execute_trading 首行当前也会 cancel_all，
        # 但"调仓日不留旧单"的语义不应依赖父类实现细节。
        self.cancel_all()
        super().on_bars(bars)


# 注册到共享策略注册表，供配置层按名取用（与 cnn/strategy.py 模式一致）
from ..backtest.registry import register_strategy  # noqa: E402

register_strategy("rebalancing_topk", RebalancingTopKStrategy)
