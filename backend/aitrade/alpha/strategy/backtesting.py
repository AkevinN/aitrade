"""
Alpha-specific backtesting module.

Re-exports the shared BacktestingEngine and adds Jupyter/notebook-only
visualization methods (show_chart, show_performance) via a thin subclass.

For backward compatibility, importing BacktestingEngine from this module
returns AlphaBacktestingEngine which wraps the shared engine with an
AlphaLab-compatible constructor.
"""

import polars as pl

from ..lab import AlphaLab, BarData
from ..logger import logger  # noqa: F401

# Re-export shared types for backward compatibility
from ...backtest.types import Direction, Offset, OrderData, TradeData  # noqa: F401
from ...backtest.engine import BacktestingEngine as _SharedEngine
from ...backtest.pnl import ContractDailyResult, PortfolioDailyResult  # noqa: F401


class BacktestingEngine(_SharedEngine):
    """Alpha 回测引擎——在共享引擎基础上封装 AlphaLab 构造方式与 Jupyter 可视化方法。

    相比 _SharedEngine，新增：
    1. 接受 AlphaLab 实例构造（无需手动传 data_loader）；
    2. show_chart：展示净值/回撤/日盈亏分布四子图；
    3. show_performance：展示策略收益/超额/换手率/超额回撤五子图（含基准对比）。
    """

    def __init__(self, lab: AlphaLab) -> None:
        """初始化回测引擎，绑定 AlphaLab 数据源。

        Args:
            lab: AlphaLab 实例，满足 BarDataLoader 协议，
                同时用于 show_performance 中加载基准行情。
        """
        super().__init__(data_loader=lab)
        # Keep a reference for show_performance (needs load_bar_data directly)
        self._lab: AlphaLab = lab

    def show_chart(self) -> None:
        """展示回测结果四子图：净值、回撤、日盈亏柱状图、日盈亏分布（仅 Jupyter 环境）。

        需在 run_backtesting 之后调用，图表高度 1000px、宽度 1000px。
        依赖 plotly，若未安装则运行时报错。
        """
        import plotly.graph_objects as go               # type: ignore
        from plotly.subplots import make_subplots       # type: ignore

        df: pl.DataFrame = self.daily_df

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Daily Pnl", "Pnl Distribution"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df["date"],
            y=df["balance"],
            mode="lines",
            name="Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df["date"],
            y=df["drawdown"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def show_performance(self, benchmark_symbol: str) -> None:
        """展示策略绩效五子图（仅 Jupyter 环境）：收益率、超额收益、换手率、超额回撤、含费超额回撤。

        自动加载基准行情计算相对收益，同时计算累计手续费对净值的拖累。
        图表高度 1500px、宽度 1200px，白色背景。

        Args:
            benchmark_symbol: 基准合约代码，如 "000300.SSE"，
                需在 AlphaLab 数据范围内且与回测区间对齐。

        Raises:
            ImportError: plotly 未安装时抛出。
        """
        import plotly.graph_objects as go               # type: ignore
        from plotly.subplots import make_subplots       # type: ignore

        benchmark_bars: list[BarData] = self._lab.load_bar_data(benchmark_symbol, self.interval, self.start, self.end)

        benchmark_prices: list[float] = []
        for bar in benchmark_bars:
            benchmark_prices.append(bar.close_price)

        performance_df: pl.DataFrame = (
            self.daily_df.with_columns(
                cumulative_return=pl.col("balance").pct_change().cum_sum(),
                cumulative_cost=(pl.col("commission") / pl.col("balance").shift(1)).cum_sum()
            ).with_columns(
                benchmark_price=pl.Series(values=benchmark_prices, dtype=pl.Float64)
            ).with_columns(
                benchmark_return=pl.col("benchmark_price").pct_change().cum_sum()
            ).with_columns(
                excess_return=(pl.col("cumulative_return") - pl.col("benchmark_return"))
            ).with_columns(
                net_excess_return=(pl.col("excess_return") - pl.col("cumulative_cost")),
            ).with_columns(
                excess_return_drawdown=(pl.col("excess_return") - pl.col("excess_return").cum_max()),
                net_excess_return_drawdown=(pl.col("net_excess_return") - pl.col("net_excess_return").cum_max())
            )
        )

        fig: go.Figure = make_subplots(
            rows=5,
            cols=1,
            subplot_titles=["Return", "Alpha", "Turnover", "Alpha Drawdown", "Alpha Drawdown with Cost"],
            vertical_spacing=0.06
        )

        strategy_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["cumulative_return"],
            mode="lines",
            name="Strategy"
        )
        net_strategy_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["cumulative_return"] - performance_df["cumulative_cost"],
            mode="lines",
            name="Strategy with Cost"
        )
        benchmark_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["benchmark_return"],
            mode="lines",
            name="Benchmark"
        )
        excess_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["excess_return"],
            mode="lines",
            name="Alpha"
        )
        net_excess_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["net_excess_return"],
            mode="lines",
            name="Alpha with Cost"
        )
        turnover_curve: go.Scatter = go.Scatter(
            x=self.daily_df["date"],
            y=self.daily_df["turnover"] / self.daily_df["balance"].shift(1),
            name="Turnover",
        )
        excess_drawdown_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["excess_return_drawdown"],
            fill='tozeroy',
            mode="lines",
            name="Alpha Drawdown"
        )
        net_excess_drawdown_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["net_excess_return_drawdown"],
            fill='tozeroy',
            mode="lines",
            name="Alpha Drawdown with Cost"
        )

        fig.add_trace(strategy_curve, row=1, col=1)
        fig.add_trace(net_strategy_curve, row=1, col=1)
        fig.add_trace(benchmark_curve, row=1, col=1)
        fig.add_trace(excess_curve, row=2, col=1)
        fig.add_trace(net_excess_curve, row=2, col=1)
        fig.add_trace(turnover_curve, row=3, col=1)
        fig.add_trace(excess_drawdown_curve, row=4, col=1)
        fig.add_trace(net_excess_drawdown_curve, row=5, col=1)

        fig.update_layout(
            height=1500,
            width=1200,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            xaxis2=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            xaxis3=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            xaxis4=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            xaxis5=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis2=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis3=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis4=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis5=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
        )
        fig.show()
