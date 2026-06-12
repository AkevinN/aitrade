"""
Data source types — shared dataclasses used across all providers.
These types are self-contained and do NOT depend on vnpy.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DataCategory(str, Enum):
    """数据品类枚举，标识各 Provider 能提供的数据种类，用于管理器路由决策。"""
    CONTRACT = "contract"
    BAR_HISTORY = "bar_history"
    TICK_HISTORY = "tick_history"
    TICK_REALTIME = "tick_realtime"
    ACCOUNT = "account"
    POSITION = "position"
    ORDER = "order"
    TRADE = "trade"
    TRADE_CALENDAR = "trade_calendar"
    FUNDAMENTAL = "fundamental"
    REFERENCE = "reference"


class ProviderStatus(str, Enum):
    """数据源可用状态枚举：AVAILABLE（正常）/ DEGRADED（降级）/
    UNAVAILABLE（不可用）/ NOT_CONFIGURED（未配置）。"""
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"


@dataclass
class ContractInfo:
    """统一合约元信息：代码、交易所、品种类型、最小变动价位等。"""
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
        """返回合约全称：``"{symbol}.{exchange}"``，如 ``"600519.SSE"``。"""
        return f"{self.symbol}.{self.exchange}"


@dataclass
class BarRecord:
    """统一 K 线记录，兼容日线 / 分钟线，含复权口径标记。"""
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
    # 复权口径：none=不复权, qfq=前复权, hfq=后复权。
    # 用于在写入本地时校验同一资源不混用不同口径（除权日价格跳变会污染回测）。
    adjust_type: str = "none"


@dataclass
class TickRecord:
    """统一历史逐笔行情记录，含最新价、量、买一/卖一价量。"""
    symbol: str
    exchange: str
    datetime: datetime
    last_price: float
    volume: float = 0.0
    turnover: float = 0.0
    bid_price_1: float = 0.0
    ask_price_1: float = 0.0
    bid_volume_1: float = 0.0
    ask_volume_1: float = 0.0


@dataclass
class CalendarDay:
    """交易日历条目：日期、交易所与是否开市。"""
    date: str  # YYYYMMDD
    exchange: str
    is_open: bool
    pre_trade_date: str = ""


@dataclass
class FundamentalRecord:
    """单日基本面数据记录：PE/PB/市值/换手率等估值与交投指标。"""
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
    """数据源描述：名称、状态、支持品类与优先级，供管理器/前端展示。"""
    name: str
    display_name: str
    status: ProviderStatus
    categories: list[DataCategory]
    priority: int = 0
    description: str = ""
    config: dict = field(default_factory=dict)
