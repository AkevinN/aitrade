"""
Alpha strategy template — re-exports shared base types and strategy class.

This module preserves backward compatibility: EquityDemoStrategy and other
alpha-specific strategies continue to import from here without changes.
"""

from ...backtest.types import Direction, Offset, OrderData, TradeData, BarData
from ...backtest.strategy import BaseStrategy

# Re-export BaseStrategy as AlphaStrategy for backward compatibility
AlphaStrategy = BaseStrategy

__all__ = [
    "Direction",
    "Offset",
    "OrderData",
    "TradeData",
    "AlphaStrategy",
    "BarData",
]
