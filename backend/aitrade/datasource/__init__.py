"""
Datasource module — exports.
"""

from .akshare_provider import AkshareProvider, AkshareProviderError
from .base import BaseProvider
from .manager import DataSourceManager, datasource_manager
from .mock_provider import MockProvider
from .tushare_provider import TushareProvider
from .types import (
    BarRecord,
    CalendarDay,
    ContractInfo,
    DataCategory,
    FundamentalRecord,
    ProviderInfo,
    ProviderStatus,
    TickRecord,
)

__all__ = [
    "BaseProvider",
    "DataSourceManager",
    "datasource_manager",
    "MockProvider",
    "TushareProvider",
    "AkshareProvider",
    "AkshareProviderError",
    "BarRecord",
    "TickRecord",
    "CalendarDay",
    "ContractInfo",
    "DataCategory",
    "FundamentalRecord",
    "ProviderInfo",
    "ProviderStatus",
]
