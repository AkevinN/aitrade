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
    """TTL 缓存条目，存储数据与到期判定。"""

    __slots__ = ("data", "ts", "ttl")

    def __init__(self, data: Any, ttl: int) -> None:
        """初始化缓存条目。

        Args:
            data: 要缓存的数据对象。
            ttl: 有效期（秒），从创建时刻起计算。
        """
        self.data = data
        self.ts = time.monotonic()
        self.ttl = ttl

    @property
    def expired(self) -> bool:
        """判断缓存是否已过期。

        Returns:
            True 表示距创建时刻已超过 ttl 秒；False 表示仍有效。
        """
        return (time.monotonic() - self.ts) >= self.ttl


class DataSourceManager:
    """多 Provider 管理器，支持按品类优先级路由与 TTL 缓存。

    查询流程：按注册优先级依次尝试各 Provider，第一个返回非 None 的结果即采用；
    若全部失败则返回空列表。部分频繁查询（合约列表、日历等）启用 TTL 缓存。
    """

    def __init__(self) -> None:
        """初始化管理器，清空 Provider 字典与缓存。"""
        self._providers: dict[str, tuple[BaseProvider, int]] = {}
        self._initialized = False
        self._cache: dict[str, _CacheEntry] = {}
        self._ttl = dict(_DEFAULT_TTL)

    # ---- Registration ----

    def register(self, provider: BaseProvider, priority: int = 50) -> None:
        """注册一个数据源 Provider。

        同名 Provider 重复注册时覆盖旧注册（热更新场景）。

        Args:
            provider: 实现 BaseProvider 的数据源对象。
            priority: 优先级（整数，越小越先尝试），默认 50。
        """
        self._providers[provider.name] = (provider, priority)
        logger.info(f"DataSourceManager: registered [{provider.name}] priority={priority}")

    def unregister(self, name: str) -> None:
        """移除指定名称的数据源 Provider。

        Args:
            name: Provider 的 name 属性，如 ``"tushare"`` / ``"akshare"``。
        """
        self._providers.pop(name, None)

    def init_all(self, output: Callable = print) -> None:
        """依次调用所有已注册 Provider 的 init() 方法。

        单个 Provider 初始化失败（抛异常或返回 False）不影响其他 Provider。
        初始化完成后将 self._initialized 置为 True。

        Args:
            output: 日志输出函数，默认 print；可替换为 logger.info。
        """
        for name, (provider, _) in self._providers.items():
            try:
                result = provider.init(output)
                logger.info(f"DataSourceManager: [{name}] init={'OK' if result else 'FAIL'}")
            except Exception as e:
                logger.warning(f"DataSourceManager: [{name}] init error: {e}")
        self._initialized = True

    def get_provider(self, name: str) -> BaseProvider | None:
        """按名称获取指定 Provider 实例。

        Args:
            name: Provider 的 name 属性。

        Returns:
            对应的 BaseProvider 实例；未注册时返回 None。
        """
        entry = self._providers.get(name)
        return entry[0] if entry else None

    def get_all_providers_info(self) -> list[ProviderInfo]:
        """返回所有已注册 Provider 的元信息列表（按优先级升序）。

        Returns:
            ProviderInfo 列表，priority 越小排越前。
        """
        result = []
        for name, (provider, priority) in self._providers.items():
            result.append(provider.get_info(priority))
        result.sort(key=lambda x: x.priority)
        return result

    # ---- Provider resolution ----

    def _get_providers_for(self, category: DataCategory) -> list[BaseProvider]:
        """返回支持指定数据品类的 Provider 列表，按优先级升序排列。

        Args:
            category: 需要的数据品类（DataCategory 枚举值）。

        Returns:
            支持该品类的 Provider 列表，priority 越小排越前。
        """
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
        """解析本次查询应使用哪些 Provider。

        Args:
            category: 请求的数据品类。
            provider_name: 若非空则锁定使用该名称的 Provider（跳过优先级排序）。

        Returns:
            有序 Provider 列表，依次尝试直到某个返回非 None 结果。
        """
        if provider_name:
            provider = self.get_provider(provider_name)
            return [provider] if provider else []
        return self._get_providers_for(category)

    # ---- Cache ----

    def _get_cached(self, key: str) -> Any | None:
        """从内存缓存中取值；过期或不存在时返回 None。

        Args:
            key: 缓存键字符串。

        Returns:
            缓存数据对象；过期或未命中时返回 None。
        """
        entry = self._cache.get(key)
        if entry and not entry.expired:
            return entry.data
        return None

    def _set_cached(self, key: str, data: Any, ttl_key: str = "") -> None:
        """将数据写入内存缓存。

        Args:
            key: 缓存键字符串。
            data: 待缓存的数据对象。
            ttl_key: 用于查询 _ttl 字典的 TTL 类型键（如 ``"contracts"``/``"calendar"``）；
                未找到时默认 300 秒。
        """
        ttl = self._ttl.get(ttl_key, 300)
        self._cache[key] = _CacheEntry(data, ttl)

    def invalidate_cache(self, prefix: str = "") -> int:
        """清除匹配前缀的缓存条目（prefix 为空则清除全部）。

        Args:
            prefix: 缓存键前缀过滤器；空字符串表示清除全部缓存。

        Returns:
            被清除的缓存条目数量。
        """
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
        """查询合约列表（带 TTL 缓存，非指定 Provider 时缓存结果）。

        Args:
            product_type: 品种类型过滤，如 ``"股票"``；空字符串不过滤。
            exchange: 交易所过滤，如 ``"SSE"``；空字符串不过滤。
            provider_name: 锁定使用的 Provider 名；空字符串按优先级尝试所有。

        Returns:
            ContractInfo 列表；全部 Provider 失败时返回空列表。
        """
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
        """查询单个合约元信息（无缓存，每次透传到 Provider）。

        Args:
            symbol: 合约代码（不含交易所后缀），如 ``"600519"``。
            exchange: 交易所代码，如 ``"SSE"``。
            provider_name: 锁定使用的 Provider 名；空字符串按优先级尝试。

        Returns:
            ContractInfo；全部 Provider 均未找到时返回 None。
        """
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
        """查询历史K线，按优先级依次尝试 Provider 直到取到非 None 结果。

        与其他查询方法的关键差异在错误处理：当显式指定 provider_name 时，该 Provider
        抛出的异常会被原样向上抛出（而非吞掉返回空列表），避免把「数据源报错」误报成
        「无数据」；未指定时则继续回退到下一个 Provider，全部失败才返回空列表。

        Args:
            symbol: 合约代码（不含交易所后缀），如 ``"600519"``。
            exchange: 交易所代码，如 ``"SSE"``。
            interval: K线周期，``"d"`` 日线，``"1m"``/``"30m"`` 等分钟线。
            start: 起始时间（含）。
            end: 截止时间（含）；None 时由 Provider 取至最新可用数据。
            provider_name: 锁定使用的 Provider 名；空字符串表示按优先级尝试所有支持该品类的 Provider。

        Returns:
            BarRecord 列表；未指定 provider_name 且全部 Provider 失败时返回空列表。

        Raises:
            Exception: 指定了 provider_name 且该 Provider 查询过程中抛错时，原样抛出其真实异常。
        """
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
        """查询历史逐笔行情，按优先级尝试 Provider 直到成功。

        Args:
            symbol: 合约代码。
            exchange: 交易所代码。
            start: 起始时间（含）。
            end: 截止时间（含）；None 时取当前时间。
            provider_name: 锁定 Provider 名；空字符串按优先级尝试。

        Returns:
            TickRecord 列表；全部 Provider 失败时返回空列表。
        """
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
        """查询交易日历（带 TTL 缓存）。

        Args:
            exchange: 交易所代码，如 ``"SSE"``。
            start: 起始日期字符串，格式 YYYYMMDD。
            end: 截止日期字符串，格式 YYYYMMDD。
            provider_name: 锁定 Provider 名；空字符串按优先级尝试。

        Returns:
            CalendarDay 列表；全部 Provider 失败时返回空列表。
        """
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
        """查询基本面数据（PE/PB/流通市值等，带 TTL 缓存）。

        Args:
            symbol: 合约代码。
            exchange: 交易所代码。
            start: 起始日期字符串，格式 YYYYMMDD。
            end: 截止日期字符串，格式 YYYYMMDD。
            provider_name: 锁定 Provider 名；空字符串按优先级尝试。

        Returns:
            FundamentalRecord 列表；全部 Provider 失败时返回空列表。
        """
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
        """查询复权因子序列（无缓存）。

        Args:
            symbol: 合约代码。
            exchange: 交易所代码。
            start: 起始日期字符串（YYYYMMDD），空字符串表示不限。
            end: 截止日期字符串（YYYYMMDD），空字符串表示不限。
            provider_name: 锁定 Provider 名；空字符串按优先级尝试。

        Returns:
            含 ``trade_date`` 与 ``adj_factor`` 字段的 dict 列表；全部失败时返回空列表。
        """
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
