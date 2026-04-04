"""
Mock data source — full-featured fallback when Tushare/Gateway unavailable.

All data is randomly generated. For development/demo use only.
"""

import random
from collections.abc import Callable
from datetime import datetime, timedelta

from .base import BaseProvider
from .types import (
    DataCategory,
    ProviderStatus,
    ContractInfo,
    BarRecord,
    CalendarDay,
    FundamentalRecord,
)

# Predefined mock contracts
MOCK_CONTRACTS: list[dict] = [
    {"symbol": "600519", "exchange": "SSE", "name": "贵州茅台", "product_type": "股票", "pricetick": 0.01, "size": 1, "min_volume": 100},
    {"symbol": "000001", "exchange": "SZSE", "name": "平安银行", "product_type": "股票", "pricetick": 0.01, "size": 1, "min_volume": 100},
    {"symbol": "510050", "exchange": "SSE", "name": "50ETF", "product_type": "基金", "pricetick": 0.001, "size": 1, "min_volume": 100},
    {"symbol": "IF2506", "exchange": "CFFEX", "name": "沪深300股指期货2506", "product_type": "期货", "pricetick": 0.2, "size": 300, "min_volume": 1},
    {"symbol": "IC2506", "exchange": "CFFEX", "name": "中证500股指期货2506", "product_type": "期货", "pricetick": 0.2, "size": 200, "min_volume": 1},
    {"symbol": "rb2510", "exchange": "SHFE", "name": "螺纹钢2510", "product_type": "期货", "pricetick": 1.0, "size": 10, "min_volume": 1},
    {"symbol": "ag2506", "exchange": "SHFE", "name": "白银2506", "product_type": "期货", "pricetick": 1.0, "size": 15, "min_volume": 1},
    {"symbol": "au2506", "exchange": "SHFE", "name": "黄金2506", "product_type": "期货", "pricetick": 0.02, "size": 1000, "min_volume": 1},
    {"symbol": "MA2509", "exchange": "CZCE", "name": "甲醇2509", "product_type": "期货", "pricetick": 1.0, "size": 10, "min_volume": 1},
    {"symbol": "m2509", "exchange": "DCE", "name": "豆粕2509", "product_type": "期货", "pricetick": 1.0, "size": 10, "min_volume": 1},
    {"symbol": "000001", "exchange": "SSE", "name": "上证指数", "product_type": "指数", "pricetick": 0.01, "size": 1, "min_volume": 1},
]

MOCK_BASE_PRICES: dict[str, float] = {
    "600519": 1800.0, "000001": 12.5, "510050": 3.5,
    "IF2506": 3950.0, "IC2506": 5800.0, "rb2510": 3350.0,
    "ag2506": 8150.0, "au2506": 580.0, "MA2509": 2650.0, "m2509": 3100.0,
}


class MockProvider(BaseProvider):
    """Mock fallback data source — always available."""

    name = "mock"
    display_name = "Mock 模拟数据"
    description = "全功能兜底降级数据源，提供模拟合约、历史K线、交易日历等数据"

    def init(self, output: Callable = print) -> bool:
        self._inited = True
        return True

    def get_status(self) -> ProviderStatus:
        return ProviderStatus.AVAILABLE if self._inited else ProviderStatus.UNAVAILABLE

    def get_supported_categories(self) -> list[DataCategory]:
        return [
            DataCategory.CONTRACT,
            DataCategory.BAR_HISTORY,
            DataCategory.TRADE_CALENDAR,
            DataCategory.FUNDAMENTAL,
            DataCategory.REFERENCE,
        ]

    # ---- Contracts ----

    def get_contracts(
        self,
        product_type: str = "",
        exchange: str = "",
    ) -> list[ContractInfo] | None:
        result = []
        for c in MOCK_CONTRACTS:
            if product_type and c["product_type"] != product_type:
                continue
            if exchange and c["exchange"] != exchange:
                continue
            result.append(ContractInfo(**c))
        return result if result else None

    def get_contract(self, symbol: str, exchange: str) -> ContractInfo | None:
        for c in MOCK_CONTRACTS:
            if c["symbol"] == symbol and c["exchange"] == exchange:
                return ContractInfo(**c)
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
        if end is None:
            end = datetime.now()

        base_price = MOCK_BASE_PRICES.get(symbol, 100.0)
        records: list[BarRecord] = []
        current = start
        price = base_price

        step_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "d": timedelta(days=1),
            "w": timedelta(weeks=1),
        }
        step = step_map.get(interval, timedelta(days=1))

        while current <= end and len(records) < 5000:
            if interval in ("d", "w") and current.weekday() >= 5:
                current += step
                continue

            change = random.gauss(0, base_price * 0.015)
            open_price = price
            close_price = price + change
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))

            records.append(BarRecord(
                symbol=symbol,
                exchange=exchange,
                datetime=current,
                interval=interval,
                open_price=round(open_price, 4),
                high_price=round(high_price, 4),
                low_price=round(low_price, 4),
                close_price=round(close_price, 4),
                volume=float(random.randint(1000, 100000)),
                turnover=float(random.randint(10000000, 500000000)),
            ))

            price = close_price
            current += step

        return records if records else None

    # ---- Trade calendar ----

    def get_trade_calendar(
        self,
        exchange: str,
        start: str,
        end: str,
    ) -> list[CalendarDay] | None:
        try:
            start_dt = datetime.strptime(start, "%Y%m%d")
            end_dt = datetime.strptime(end, "%Y%m%d")
        except ValueError:
            return None

        result: list[CalendarDay] = []
        current = start_dt

        while current <= end_dt:
            is_open = current.weekday() < 5
            result.append(CalendarDay(
                date=current.strftime("%Y%m%d"),
                exchange=exchange,
                is_open=is_open,
            ))
            current += timedelta(days=1)

        return result if result else None

    # ---- Fundamentals ----

    def get_fundamental(
        self,
        symbol: str,
        exchange: str,
        start: str,
        end: str,
    ) -> list[FundamentalRecord] | None:
        base_price = MOCK_BASE_PRICES.get(symbol, 100.0)

        try:
            start_dt = datetime.strptime(start, "%Y%m%d")
            end_dt = datetime.strptime(end, "%Y%m%d")
        except ValueError:
            return None

        result: list[FundamentalRecord] = []
        current = start_dt

        while current <= end_dt:
            if current.weekday() < 5:
                result.append(FundamentalRecord(
                    symbol=symbol,
                    exchange=exchange,
                    trade_date=current.strftime("%Y%m%d"),
                    pe=round(random.uniform(10, 50), 2),
                    pe_ttm=round(random.uniform(10, 50), 2),
                    pb=round(random.uniform(1, 10), 2),
                    total_mv=round(base_price * random.uniform(1e6, 1e8), 2),
                    circ_mv=round(base_price * random.uniform(5e5, 5e7), 2),
                    turnover_rate=round(random.uniform(0.5, 5.0), 2),
                ))
            current += timedelta(days=1)

        return result if result else None

    # ---- Adjustment factors ----

    def get_adj_factor(
        self,
        symbol: str,
        exchange: str,
        start: str = "",
        end: str = "",
    ) -> list[dict] | None:
        try:
            start_dt = datetime.strptime(start, "%Y%m%d") if start else datetime(2024, 1, 1)
            end_dt = datetime.strptime(end, "%Y%m%d") if end else datetime.now()
        except ValueError:
            return None

        result = []
        current = start_dt
        factor = 1.0

        while current <= end_dt:
            if current.weekday() < 5:
                result.append({
                    "trade_date": current.strftime("%Y%m%d"),
                    "adj_factor": round(factor, 6),
                })
            current += timedelta(days=1)

        return result if result else None
