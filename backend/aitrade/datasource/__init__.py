"""
Datasource module — exports.
"""

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
)

__all__ = [
    "BaseProvider",
    "DataSourceManager",
    "datasource_manager",
    "MockProvider",
    "TushareProvider",
    "BarRecord",
    "CalendarDay",
    "ContractInfo",
    "DataCategory",
    "FundamentalRecord",
    "ProviderInfo",
    "ProviderStatus",
]
