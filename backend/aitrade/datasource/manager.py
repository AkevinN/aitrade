"""
Data source manager — orchestrates multiple providers with fallback chain.

Each DataCategory maintains an ordered list of Providers.
Queries are tried in priority order; if one returns None, the next is used.

Usage:
    manager = DataSourceManager()
    manager.register(TushareProvider(), priority=10)
    manager.register(MockProvider(), priority=100)
    bars = manager.get_bar_history("600519", "SSE", "d", start, end)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from .base import BaseProvider
from .types import (
    DataCategory,
    ProviderInfo,
    ContractInfo,
    BarRecord,
    TickRecord,
    CalendarDay,
    FundamentalRecord,
)

logger = logging.getLogger(__name__)

_DEFAULT_TTL: dict[str, int] = {
    "contracts": 300,
    "calendar": 3600,
    "fundamental": 600,
    "adj_factor": 3600,
    "providers": 60,
}


class _CacheEntry:
    """TTL cache entry."""
    __slots__ = ("data", "ts", "ttl")

    def __init__(self, data: Any, ttl: int) -> None:
        self.data = data
        self.ts = time.monotonic()
        self.ttl = ttl

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.ts) >= self.ttl


class DataSourceManager:
    """Multi-provider manager with TTL cache and per-category fallback routing."""

    def __init__(self) -> None:
        self._providers: dict[str, tuple[BaseProvider, int]] = {}
        self._initialized = False
        self._cache: dict[str, _CacheEntry] = {}
        self._ttl = dict(_DEFAULT_TTL)

    # ---- Registration ----

    def register(self, provider: BaseProvider, priority: int = 50) -> None:
        """Register a data source provider."""
        self._providers[provider.name] = (provider, priority)
        logger.info(f"DataSourceManager: registered [{provider.name}] priority={priority}")

    def unregister(self, name: str) -> None:
        """Remove a data source provider."""
        self._providers.pop(name, None)

    def init_all(self, output: Callable = print) -> None:
        """Initialize all registered providers."""
        for name, (provider, _) in self._providers.items():
            try:
                result = provider.init(output)
                logger.info(f"DataSourceManager: [{name}] init={'OK' if result else 'FAIL'}")
            except Exception as e:
                logger.warning(f"DataSourceManager: [{name}] init error: {e}")
        self._initialized = True

    def get_provider(self, name: str) -> BaseProvider | None:
        """Get a specific provider by name."""
        entry = self._providers.get(name)
        return entry[0] if entry else None

    def get_all_providers_info(self) -> list[ProviderInfo]:
        """Get description info for all registered providers."""
        result = []
        for name, (provider, priority) in self._providers.items():
            result.append(provider.get_info(priority))
        result.sort(key=lambda x: x.priority)
        return result

    # ---- Provider resolution ----

    def _get_providers_for(self, category: DataCategory) -> list[BaseProvider]:
        """Get providers supporting a category, sorted by priority."""
        candidates: list[tuple[BaseProvider, int]] = []
        for _, (provider, priority) in self._providers.items():
            if category in provider.get_supported_categories():
                candidates.append((provider, priority))
        candidates.sort(key=lambda x: x[1])
        return [p for p, _ in candidates]

    def _resolve_providers(
        self,
        category: DataCategory,
        provider_name: str = "",
    ) -> list[BaseProvider]:
        """Resolve which provider(s) to use."""
        if provider_name:
            provider = self.get_provider(provider_name)
            return [provider] if provider else []
        return self._get_providers_for(category)

    # ---- Cache ----

    def _get_cached(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry and not entry.expired:
            return entry.data
        return None

    def _set_cached(self, key: str, data: Any, ttl_key: str = "") -> None:
        ttl = self._ttl.get(ttl_key, 300)
        self._cache[key] = _CacheEntry(data, ttl)

    def invalidate_cache(self, prefix: str = "") -> int:
        """Clear cache entries matching prefix (or all if prefix='')."""
        if not prefix:
            count = len(self._cache)
            self._cache.clear()
            return count
        keys = [k for k in self._cache if k.startswith(prefix)]
        for k in keys:
            del self._cache[k]
        return len(keys)

    # ---- Contracts ----

    def get_contracts(
        self,
        product_type: str = "",
        exchange: str = "",
        provider_name: str = "",
    ) -> list[ContractInfo]:
        cache_key = f"contracts:{product_type}:{exchange}"
        if not provider_name:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        providers = self._resolve_providers(DataCategory.CONTRACT, provider_name)
        for provider in providers:
            try:
                result = provider.get_contracts(product_type=product_type, exchange=exchange)
                if result is not None:
                    if not provider_name:
                        self._set_cached(cache_key, result, "contracts")
                    return result
            except Exception as e:
                logger.warning(f"get_contracts: [{provider.name}] failed - {e}")
        return []

    def get_contract(
        self,
        symbol: str,
        exchange: str,
        provider_name: str = "",
    ) -> ContractInfo | None:
        providers = self._resolve_providers(DataCategory.CONTRACT, provider_name)
        for provider in providers:
            try:
                result = provider.get_contract(symbol, exchange)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"get_contract: [{provider.name}] failed - {e}")
        return None

    # ---- Historical bars ----

    def get_bar_history(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: datetime,
        end: datetime | None = None,
        provider_name: str = "",
    ) -> list[BarRecord]:
        """Query historical K-line bars with fallback."""
        providers = self._resolve_providers(DataCategory.BAR_HISTORY, provider_name)
        last_error: Exception | None = None
        for provider in providers:
            try:
                result = provider.get_bar_history(symbol, exchange, interval, start, end)
                if result is not None:
                    logger.debug(f"get_bar_history: [{provider.name}] returned {len(result)} bars")
                    return result
            except Exception as e:
                last_error = e
                logger.warning(f"get_bar_history: [{provider.name}] failed - {e}")
                # 用户显式指定数据源时，直接抛出真实错误，避免误报「无数据」。
                if provider_name:
                    raise
        if provider_name and last_error is not None:
            raise last_error
        return []

    def get_tick_history(
        self,
        symbol: str,
        exchange: str,
        start: datetime,
        end: datetime | None = None,
        provider_name: str = "",
    ) -> list[TickRecord]:
        """Query historical ticks with fallback."""
        providers = self._resolve_providers(DataCategory.TICK_HISTORY, provider_name)
        for provider in providers:
            try:
                result = provider.get_tick_history(symbol, exchange, start, end)
                if result is not None:
                    logger.debug(f"get_tick_history: [{provider.name}] returned {len(result)} ticks")
                    return result
            except Exception as e:
                logger.warning(f"get_tick_history: [{provider.name}] failed - {e}")
        return []

    # ---- Trade calendar ----

    def get_trade_calendar(
        self,
        exchange: str,
        start: str,
        end: str,
        provider_name: str = "",
    ) -> list[CalendarDay]:
        cache_key = f"calendar:{exchange}:{start}:{end}"
        if not provider_name:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        providers = self._resolve_providers(DataCategory.TRADE_CALENDAR, provider_name)
        for provider in providers:
            try:
                result = provider.get_trade_calendar(exchange, start, end)
                if result is not None:
                    if not provider_name:
                        self._set_cached(cache_key, result, "calendar")
                    return result
            except Exception as e:
                logger.warning(f"get_trade_calendar: [{provider.name}] failed - {e}")
        return []

    # ---- Fundamentals ----

    def get_fundamental(
        self,
        symbol: str,
        exchange: str,
        start: str,
        end: str,
        provider_name: str = "",
    ) -> list[FundamentalRecord]:
        cache_key = f"fundamental:{symbol}:{exchange}:{start}:{end}"
        if not provider_name:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        providers = self._resolve_providers(DataCategory.FUNDAMENTAL, provider_name)
        for provider in providers:
            try:
                result = provider.get_fundamental(symbol, exchange, start, end)
                if result is not None:
                    if not provider_name:
                        self._set_cached(cache_key, result, "fundamental")
                    return result
            except Exception as e:
                logger.warning(f"get_fundamental: [{provider.name}] failed - {e}")
        return []

    # ---- Adjustment factors ----

    def get_adj_factor(
        self,
        symbol: str,
        exchange: str,
        start: str = "",
        end: str = "",
        provider_name: str = "",
    ) -> list[dict]:
        providers = self._resolve_providers(DataCategory.REFERENCE, provider_name)
        for provider in providers:
            try:
                result = provider.get_adj_factor(symbol, exchange, start, end)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"get_adj_factor: [{provider.name}] failed - {e}")
        return []


# Global singleton
datasource_manager = DataSourceManager()
