"""
Datasource module — exports.
"""

from .akshare_provider import AkshareProvider, AkshareProviderError
from .base import BaseProvider
from .manager import DataSourceManager, datasource_manager
from .mock_provider import MockProvider
from .qmt_bridge_provider import QmtBridgeProvider  # noqa: F401
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
    "QmtBridgeProvider",
    "BarRecord",
    "TickRecord",
    "CalendarDay",
    "ContractInfo",
    "DataCategory",
    "FundamentalRecord",
    "ProviderInfo",
    "ProviderStatus",
]
