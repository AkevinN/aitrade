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
    """Abstract data source provider."""

    name: str = ""
    display_name: str = ""
    description: str = ""

    @abstractmethod
    def init(self, output: Callable = print) -> bool:
        """Initialize the provider connection. Return True on success."""
        ...

    @abstractmethod
    def get_status(self) -> ProviderStatus:
        """Return current provider status."""
        ...

    @abstractmethod
    def get_supported_categories(self) -> list[DataCategory]:
        """Return list of data categories this provider supports."""
        ...

    def get_info(self, priority: int = 0) -> ProviderInfo:
        """Return provider description."""
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
        """Query contract list. Return None = not supported."""
        return None

    def get_contract(self, symbol: str, exchange: str) -> ContractInfo | None:
        """Query single contract."""
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
        """Query historical K-line bars. Return None = not supported."""
        return None

    def get_tick_history(
        self,
        symbol: str,
        exchange: str,
        start: datetime,
        end: datetime | None = None,
    ) -> list[TickRecord] | None:
        """Query historical ticks. Return None = not supported."""
        return None

    # ---- Real-time tick ----

    def get_latest_tick(self, symbol: str, exchange: str) -> dict | None:
        """Get latest tick snapshot. Return None = not supported."""
        return None

    def get_all_ticks(self) -> list[dict] | None:
        """Get all tick snapshots."""
        return None

    # ---- Account / Position / Order / Trade ----

    def get_accounts(self) -> list[dict] | None:
        return None

    def get_positions(self) -> list[dict] | None:
        return None

    def get_orders(self) -> list[dict] | None:
        return None

    def get_trades(self) -> list[dict] | None:
        return None

    # ---- Trade calendar ----

    def get_trade_calendar(
        self,
        exchange: str,
        start: str,
        end: str,
    ) -> list[CalendarDay] | None:
        """Query trade calendar. Return None = not supported."""
        return None

    # ---- Fundamentals ----

    def get_fundamental(
        self,
        symbol: str,
        exchange: str,
        start: str,
        end: str,
    ) -> list[FundamentalRecord] | None:
        """Query fundamental data. Return None = not supported."""
        return None

    # ---- Reference data ----

    def get_adj_factor(
        self,
        symbol: str,
        exchange: str,
        start: str = "",
        end: str = "",
    ) -> list[dict] | None:
        """Query adjustment factors. Return None = not supported."""
        return None
