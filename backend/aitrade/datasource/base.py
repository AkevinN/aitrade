"""
Data source abstract base class.

All providers must inherit BaseProvider and implement the methods
for data categories they support.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime

from .types import (
    DataCategory,
    ProviderStatus,
    ProviderInfo,
    ContractInfo,
    BarRecord,
    TickRecord,
    CalendarDay,
    FundamentalRecord,
)


class BaseProvider(ABC):
    """数据源抽象基类，定义所有 Provider 的统一接口。

    子类需实现 init / get_status / get_supported_categories 三个抽象方法；
    其余数据查询方法默认返回 None（表示该品类不支持），子类按需覆盖。
    """

    name: str = ""
    display_name: str = ""
    description: str = ""

    @abstractmethod
    def init(self, output: Callable = print) -> bool:
        """初始化数据源连接（如导入依赖库、验证 token）。

        Args:
            output: 日志输出函数，默认 print；可替换为 logger.info。

        Returns:
            True 表示初始化成功，False 表示失败（未抛异常的软失败）。
        """
        ...

    @abstractmethod
    def get_status(self) -> ProviderStatus:
        """返回当前数据源可用状态。

        Returns:
            ProviderStatus 枚举值：AVAILABLE / DEGRADED / UNAVAILABLE / NOT_CONFIGURED。
        """
        ...

    @abstractmethod
    def get_supported_categories(self) -> list[DataCategory]:
        """返回本数据源支持的数据品类列表。

        Returns:
            DataCategory 枚举值列表，供 DataSourceManager 路由决策使用。
        """
        ...

    def get_info(self, priority: int = 0) -> ProviderInfo:
        """返回本数据源的元信息描述（供管理器/前端展示）。

        Args:
            priority: 该数据源在管理器中的优先级（数值越小优先级越高）。

        Returns:
            含 name / display_name / status / categories / priority / description 的 ProviderInfo。
        """
        return ProviderInfo(
            name=self.name,
            display_name=self.display_name,
            status=self.get_status(),
            categories=self.get_supported_categories(),
            priority=priority,
            description=self.description,
        )

    # ---- Contract data ----

    def get_contracts(
        self,
        product_type: str = "",
        exchange: str = "",
    ) -> list[ContractInfo] | None:
        """查询合约列表。

        Args:
            product_type: 品种过滤（如 ``"股票"``/``"期货"``），空字符串表示不过滤。
            exchange: 交易所过滤（如 ``"SSE"``），空字符串表示不过滤。

        Returns:
            ContractInfo 列表；None 表示本数据源不支持该查询。
        """
        return None

    def get_contract(self, symbol: str, exchange: str) -> ContractInfo | None:
        """查询单个合约的元信息。

        Args:
            symbol: 合约代码（不含交易所后缀），如 ``"600519"``。
            exchange: 交易所代码，如 ``"SSE"``。

        Returns:
            ContractInfo；None 表示未找到或不支持。
        """
        return None

    # ---- Historical bars ----

    def get_bar_history(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: datetime,
        end: datetime | None = None,
    ) -> list[BarRecord] | None:
        """查询历史 K 线数据。

        Args:
            symbol: 合约代码（不含交易所后缀）。
            exchange: 交易所代码。
            interval: K 线周期，如 ``"d"``/``"1m"``/``"30m"``。
            start: 起始时间（含）。
            end: 截止时间（含）；None 时取当前时间。

        Returns:
            BarRecord 列表（按 datetime 升序）；None 表示不支持或请求失败。
        """
        return None

    def get_tick_history(
        self,
        symbol: str,
        exchange: str,
        start: datetime,
        end: datetime | None = None,
    ) -> list[TickRecord] | None:
        """查询历史逐笔行情。

        Args:
            symbol: 合约代码。
            exchange: 交易所代码。
            start: 起始时间（含）。
            end: 截止时间（含）；None 时取当前时间。

        Returns:
            TickRecord 列表；None 表示不支持。
        """
        return None

    # ---- Real-time tick ----

    def get_latest_tick(self, symbol: str, exchange: str) -> dict | None:
        """获取最新 tick 快照（实时行情）。

        Args:
            symbol: 合约代码。
            exchange: 交易所代码。

        Returns:
            含行情字段的 dict；None 表示不支持。
        """
        return None

    def get_all_ticks(self) -> list[dict] | None:
        """获取全部持仓合约的 tick 快照列表。

        Returns:
            tick 列表；None 表示不支持。
        """
        return None

    # ---- Account / Position / Order / Trade ----

    def get_accounts(self) -> list[dict] | None:
        """返回账户信息列表；None 表示不支持。"""
        return None

    def get_positions(self) -> list[dict] | None:
        """返回持仓信息列表；None 表示不支持。"""
        return None

    def get_orders(self) -> list[dict] | None:
        """返回当日委托列表；None 表示不支持。"""
        return None

    def get_trades(self) -> list[dict] | None:
        """返回当日成交列表；None 表示不支持。"""
        return None

    # ---- Trade calendar ----

    def get_trade_calendar(
        self,
        exchange: str,
        start: str,
        end: str,
    ) -> list[CalendarDay] | None:
        """查询交易日历。

        Args:
            exchange: 交易所代码，如 ``"SSE"``。
            start: 起始日期字符串，格式 YYYYMMDD。
            end: 截止日期字符串，格式 YYYYMMDD。

        Returns:
            CalendarDay 列表；None 表示不支持。
        """
        return None

    # ---- Fundamentals ----

    def get_fundamental(
        self,
        symbol: str,
        exchange: str,
        start: str,
        end: str,
    ) -> list[FundamentalRecord] | None:
        """查询基本面数据（PE/PB/流通市值等）。

        Args:
            symbol: 合约代码。
            exchange: 交易所代码。
            start: 起始日期字符串，格式 YYYYMMDD。
            end: 截止日期字符串，格式 YYYYMMDD。

        Returns:
            FundamentalRecord 列表；None 表示不支持。
        """
        return None

    # ---- Reference data ----

    def get_adj_factor(
        self,
        symbol: str,
        exchange: str,
        start: str = "",
        end: str = "",
    ) -> list[dict] | None:
        """查询复权因子序列。

        Args:
            symbol: 合约代码。
            exchange: 交易所代码。
            start: 起始日期字符串（YYYYMMDD），空字符串表示不限。
            end: 截止日期字符串（YYYYMMDD），空字符串表示不限。

        Returns:
            含 ``trade_date`` 与 ``adj_factor`` 字段的 dict 列表；None 表示不支持。
        """
        return None
