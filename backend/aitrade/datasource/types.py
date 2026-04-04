"""
Data source types — shared dataclasses used across all providers.
These types are self-contained and do NOT depend on vnpy.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DataCategory(str, Enum):
    """Data category — identifies what a Provider can supply."""
    CONTRACT = "contract"
    BAR_HISTORY = "bar_history"
    TICK_REALTIME = "tick_realtime"
    ACCOUNT = "account"
    POSITION = "position"
    ORDER = "order"
    TRADE = "trade"
    TRADE_CALENDAR = "trade_calendar"
    FUNDAMENTAL = "fundamental"
    REFERENCE = "reference"


class ProviderStatus(str, Enum):
    """Data source availability status."""
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"


@dataclass
class ContractInfo:
    """Unified contract metadata."""
    symbol: str
    exchange: str
    name: str
    product_type: str = ""
    size: float = 1.0
    pricetick: float = 0.01
    min_volume: float = 1.0
    list_date: str = ""
    delist_date: str = ""
    underlying: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange}"


@dataclass
class BarRecord:
    """Unified bar/K-line record."""
    symbol: str
    exchange: str
    datetime: datetime
    interval: str  # "1m", "5m", "15m", "30m", "1h", "d", "w"
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float = 0.0
    turnover: float = 0.0
    open_interest: float = 0.0


@dataclass
class CalendarDay:
    """Trade calendar entry."""
    date: str  # YYYYMMDD
    exchange: str
    is_open: bool
    pre_trade_date: str = ""


@dataclass
class FundamentalRecord:
    """Fundamental data record."""
    symbol: str
    exchange: str
    trade_date: str
    pe: float | None = None
    pe_ttm: float | None = None
    pb: float | None = None
    ps: float | None = None
    total_mv: float | None = None
    circ_mv: float | None = None
    turnover_rate: float | None = None
    volume_ratio: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class ProviderInfo:
    """Data source description."""
    name: str
    display_name: str
    status: ProviderStatus
    categories: list[DataCategory]
    priority: int = 0
    description: str = ""
    config: dict = field(default_factory=dict)
