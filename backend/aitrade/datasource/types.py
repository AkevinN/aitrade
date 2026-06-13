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
    """统一合约元信息：代码、交易所、品种类型、最小变动价位等。

    各 Provider 拉取到的合约信息归一为本结构，屏蔽数据源字段差异。

    Attributes:
        symbol: 不含交易所后缀的代码，如 "600519"。
        exchange: 交易所代码，如 "SSE"/"SZSE"。
        name: 合约名称（中文简称）。
        product_type: 品种类型（如股票/基金/期货），空串表示未知。
        size: 合约乘数（每手对应的标的数量），股票通常为 1.0。
        pricetick: 最小变动价位（元），如 A 股 0.01。
        min_volume: 最小下单数量（手/股），默认 1.0。
        list_date: 上市日期（YYYYMMDD），空串表示未知。
        delist_date: 退市日期（YYYYMMDD），空串表示在市。
        underlying: 标的物代码，仅期权/期货等衍生品有值。
        extra: 数据源透传的额外字段，键名随 Provider 而异。
    """
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
    """统一 K 线记录，兼容日线 / 分钟线，含复权口径标记。

    各数据源的历史 K 线归一为本结构后再落盘；adjust_type 用于防止同一资源
    混入不同复权口径（除权日价格跳变会污染回测）。

    Attributes:
        symbol: 不含交易所后缀的代码，如 "600519"。
        exchange: 交易所代码，如 "SSE"/"SZSE"。
        datetime: K 线时刻；日线一般取交易日 00:00，分钟线取该 bar 的起/收时刻。
        interval: K 线周期，取值 "1m"/"5m"/"15m"/"30m"/"1h"/"d"/"w"。
        open_price: 开盘价（元）。
        high_price: 最高价（元）。
        low_price: 最低价（元）。
        close_price: 收盘价（元）。
        volume: 成交量（股/手），默认 0.0。
        turnover: 成交额（元），默认 0.0。
        open_interest: 持仓量，仅期货等有意义，股票恒为 0.0。
        adjust_type: 复权口径，"none" 不复权 / "qfq" 前复权 / "hfq" 后复权。
    """
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
    """统一历史逐笔行情记录，含最新价、量、买一/卖一价量。

    Attributes:
        symbol: 不含交易所后缀的代码，如 "600519"。
        exchange: 交易所代码，如 "SSE"/"SZSE"。
        datetime: 该笔行情时刻。
        last_price: 最新成交价（元）。
        volume: 截至该时刻的累计成交量（股/手），默认 0.0。
        turnover: 截至该时刻的累计成交额（元），默认 0.0。
        bid_price_1: 买一价（元），默认 0.0 表示缺失。
        ask_price_1: 卖一价（元），默认 0.0 表示缺失。
        bid_volume_1: 买一量（股/手），默认 0.0。
        ask_volume_1: 卖一量（股/手），默认 0.0。
    """
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
    """交易日历条目：日期、交易所与是否开市。

    Attributes:
        date: 日历日期，格式 YYYYMMDD。
        exchange: 交易所代码，如 "SSE"/"SZSE"。
        is_open: 当日是否开市（True 为交易日）。
        pre_trade_date: 上一个交易日（YYYYMMDD），空串表示未知/无。
    """
    date: str  # YYYYMMDD
    exchange: str
    is_open: bool
    pre_trade_date: str = ""


@dataclass
class FundamentalRecord:
    """单日基本面数据记录：PE/PB/市值/换手率等估值与交投指标。

    数值字段均可为 None，表示该数据源当日未提供该指标（区别于真实值 0）。

    Attributes:
        symbol: 不含交易所后缀的代码，如 "600519"。
        exchange: 交易所代码，如 "SSE"/"SZSE"。
        trade_date: 数据所属交易日，格式 YYYYMMDD。
        pe: 静态市盈率，None 表示缺失。
        pe_ttm: 滚动市盈率（最近 12 个月），None 表示缺失。
        pb: 市净率，None 表示缺失。
        ps: 市销率，None 表示缺失。
        total_mv: 总市值（元），None 表示缺失。
        circ_mv: 流通市值（元），None 表示缺失。
        turnover_rate: 换手率（百分比），None 表示缺失。
        volume_ratio: 量比，None 表示缺失。
        extra: 数据源透传的额外字段，键名随 Provider 而异。
    """
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
    """数据源描述：名称、状态、支持品类与优先级，供管理器/前端展示。

    Attributes:
        name: Provider 唯一标识（程序内引用名）。
        display_name: 面向用户的展示名。
        status: 当前可用状态，见 ProviderStatus。
        categories: 该 Provider 支持的数据品类列表。
        priority: 路由优先级，数值越大越优先选用，默认 0。
        description: 数据源说明文本。
        config: 该 Provider 的配置项（如 token/路径），键随实现而异。
    """
    name: str
    display_name: str
    status: ProviderStatus
    categories: list[DataCategory]
    priority: int = 0
    description: str = ""
    config: dict = field(default_factory=dict)
