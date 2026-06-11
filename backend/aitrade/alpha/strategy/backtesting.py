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
from ..logger import logger

# Re-export shared types for backward compatibility
from ...backtest.types import Direction, Offset, OrderData, TradeData  # noqa: F401
from ...backtest.engine import BacktestingEngine as _SharedEngine
from ...backtest.pnl import ContractDailyResult, PortfolioDailyResult  # noqa: F401


class BacktestingEngine(_SharedEngine):
    """Alpha backtesting engine — wraps shared engine with AlphaLab constructor
    and Jupyter-specific visualization methods."""

    def __init__(self, lab: AlphaLab) -> None:
        """Constructor — accepts AlphaLab which satisfies BarDataLoader protocol."""
        super().__init__(data_loader=lab)
        # Keep a reference for show_performance (needs load_bar_data directly)
        self._lab: AlphaLab = lab

    def show_chart(self) -> None:
        """Display chart (Jupyter/notebook only)"""
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
        """Display performance metrics (Jupyter/notebook only)"""
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
