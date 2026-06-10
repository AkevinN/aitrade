"""
Shared backtesting infrastructure — engine, strategy, PnL calculation.

Both Alpha and CNN modules reuse this package for strategy backtesting.
"""

from .types import Direction, Offset, OrderData, TradeData, BarData
from .strategy import BaseStrategy
from .engine import BacktestingEngine, BarDataLoader
from .pnl import ContractDailyResult, PortfolioDailyResult

__all__ = [
    "Direction",
    "Offset",
    "OrderData",
    "TradeData",
    "BarData",
    "BaseStrategy",
    "BacktestingEngine",
    "BarDataLoader",
    "ContractDailyResult",
    "PortfolioDailyResult",
]
