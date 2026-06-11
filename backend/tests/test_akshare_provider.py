"""AkshareProvider 与数据源选择逻辑单元测试（mock 数据，无需联网）。"""

from __future__ import annotations

from datetime import datetime

import pytest

from aitrade.datasource.akshare_provider import AkshareProvider, AkshareProviderError
from aitrade.datasource.base import BaseProvider
from aitrade.datasource.manager import DataSourceManager
from aitrade.datasource.types import DataCategory, ProviderStatus


class _FakeFrame:
    """最小化模拟 pandas DataFrame，仅实现 Provider 测试所需的方法。"""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def to_dict(self, orient: str = "records") -> list[dict]:
        assert orient == "records"
        return list(self._rows)


class _FakeAk:
    """模拟 akshare 模块：记录调用参数并返回预置行情数据。"""

    def __init__(self, hist_frame=None, min_frame=None, sina_min_frame=None, contract_frame=None, raise_on=None) -> None:
        self._hist_frame = hist_frame
        self._min_frame = min_frame
        self._sina_min_frame = sina_min_frame
        self._contract_frame = contract_frame
        self._raise_on = raise_on or set()
        self.hist_calls: list[dict] = []
        self.min_calls: list[dict] = []
        self.sina_min_calls: list[dict] = []
        self.contract_calls = 0

    def stock_zh_a_hist(self, **kwargs):
        self.hist_calls.append(kwargs)
        if "hist" in self._raise_on:
            raise RuntimeError("boom")
        return self._hist_frame

    def stock_zh_a_hist_min_em(self, **kwargs):
        self.min_calls.append(kwargs)
        if "min" in self._raise_on:
            raise RuntimeError("boom")
        return self._min_frame

    def stock_zh_a_minute(self, **kwargs):
        self.sina_min_calls.append(kwargs)
        if "sina" in self._raise_on:
            raise RuntimeError("boom")
        return self._sina_min_frame

    def stock_info_a_code_name(self):
        self.contract_calls += 1
        if "contract" in self._raise_on:
            raise RuntimeError("boom")
        return self._contract_frame


def _make_provider(fake_ak: _FakeAk) -> AkshareProvider:
    provider = AkshareProvider()
    provider._inited = True
    provider._ak = fake_ak
    return provider


DAILY_ROWS = [
    {"日期": "2024-01-02", "开盘": 10.0, "收盘": 10.5, "最高": 10.8, "最低": 9.9, "成交量": 1000, "成交额": 1_050_000.0},
    {"日期": "2024-01-03", "开盘": 10.5, "收盘": 10.2, "最高": 10.6, "最低": 10.1, "成交量": 1200, "成交额": 1_230_000.0},
]

MIN_ROWS = [
    {"时间": "2024-03-20 09:30:00", "开盘": 10.38, "收盘": 10.40, "最高": 10.41, "最低": 10.38, "成交量": 7174, "成交额": 7_446_612.0},
    {"时间": "2024-03-20 09:31:00", "开盘": 10.40, "收盘": 10.41, "最高": 10.41, "最低": 10.39, "成交量": 1040, "成交额": 1_082_435.0},
]

SINA_MIN_ROWS = [
    {"day": "2024-03-20 09:30:00", "open": 10.38, "high": 10.41, "low": 10.38, "close": 10.40, "volume": 7174, "amount": 7446612.0},
    {"day": "2024-03-20 09:35:00", "open": 10.40, "high": 10.42, "low": 10.39, "close": 10.41, "volume": 1040, "amount": 1082435.0},
]

CONTRACT_ROWS = [
    {"代码": "600519", "名称": "贵州茅台"},
    {"代码": "000001", "名称": "平安银行"},
    {"代码": "830799", "名称": "艾融软件"},
]


def test_daily_bars_mapped_to_records() -> None:
    fake = _FakeAk(hist_frame=_FakeFrame(DAILY_ROWS))
    provider = _make_provider(fake)

    records = provider.get_bar_history(
        "600519", "SSE", "d", datetime(2024, 1, 1), datetime(2024, 2, 1)
    )

    assert records is not None and len(records) == 2
    first = records[0]
    assert first.symbol == "600519"
    assert first.exchange == "SSE"
    assert first.interval == "d"
    assert first.datetime == datetime(2024, 1, 2)
    assert first.open_price == 10.0
    assert first.close_price == 10.5
    assert first.high_price == 10.8
    assert first.low_price == 9.9
    assert first.volume == 1000.0
    assert first.turnover == 1_050_000.0
    assert first.open_interest == 0.0
    # 日线应路由到 stock_zh_a_hist，并使用正确周期与 YYYYMMDD 日期格式。
    assert fake.hist_calls and fake.hist_calls[0]["symbol"] == "600519"
    assert fake.hist_calls[0]["period"] == "daily"
    assert fake.hist_calls[0]["start_date"] == "20240101"
    assert not fake.min_calls


def test_accepts_vt_symbol_in_symbol_field() -> None:
    fake = _FakeAk(hist_frame=_FakeFrame(DAILY_ROWS))
    provider = _make_provider(fake)

    records = provider.get_bar_history(
        "000415.SZSE", "", "d", datetime(2024, 1, 1), datetime(2024, 2, 1)
    )

    assert records is not None and len(records) == 2
    assert records[0].symbol == "000415"
    assert records[0].exchange == "SZSE"
    assert fake.hist_calls[0]["symbol"] == "000415"


def test_accepts_prefixed_symbol_without_exchange() -> None:
    fake = _FakeAk(hist_frame=_FakeFrame(DAILY_ROWS))
    provider = _make_provider(fake)

    records = provider.get_bar_history(
        "sz000415", "", "d", datetime(2024, 1, 1), datetime(2024, 2, 1)
    )

    assert records is not None
    assert fake.hist_calls[0]["symbol"] == "000415"


def test_exchange_mismatch_raises_clear_error() -> None:
    fake = _FakeAk(hist_frame=_FakeFrame(DAILY_ROWS))
    provider = _make_provider(fake)

    with pytest.raises(AkshareProviderError, match="不匹配"):
        provider.get_bar_history("000415", "SSE", "d", datetime(2024, 1, 1))


@pytest.mark.parametrize(
    "interval,period",
    [("d", "daily"), ("w", "weekly"), ("m", "monthly")],
)
def test_daily_period_routing(interval: str, period: str) -> None:
    fake = _FakeAk(hist_frame=_FakeFrame(DAILY_ROWS))
    provider = _make_provider(fake)
    provider.get_bar_history("000001", "SZSE", interval, datetime(2024, 1, 1))
    assert fake.hist_calls[0]["period"] == period


@pytest.mark.parametrize(
    "interval,period",
    [("1m", "1"), ("5m", "5"), ("15m", "15"), ("30m", "30"), ("1h", "60"), ("60m", "60")],
)
def test_minute_period_routing(interval: str, period: str) -> None:
    fake = _FakeAk(min_frame=_FakeFrame(MIN_ROWS))
    provider = _make_provider(fake)
    records = provider.get_bar_history(
        "000001", "SZSE", interval, datetime(2024, 3, 20), datetime(2024, 3, 21)
    )
    assert records is not None and len(records) == 2
    assert records[0].datetime == datetime(2024, 3, 20, 9, 30, 0)
    assert fake.min_calls and fake.min_calls[0]["period"] == period
    assert not fake.hist_calls


def test_minute_fallback_to_sina_when_em_fails(monkeypatch) -> None:
    monkeypatch.setattr("aitrade.datasource.akshare_provider.AKSHARE_MAX_RETRIES", 1)
    fake = _FakeAk(raise_on={"min"}, sina_min_frame=_FakeFrame(SINA_MIN_ROWS))
    provider = _make_provider(fake)

    records = provider.get_bar_history(
        "000415", "SZSE", "5m", datetime(2024, 3, 20), datetime(2024, 3, 21)
    )

    assert records is not None and len(records) == 2
    assert fake.min_calls
    assert fake.sina_min_calls
    assert fake.sina_min_calls[0]["symbol"] == "sz000415"
    assert fake.sina_min_calls[0]["period"] == "5"


def test_invalid_interval_raises() -> None:
    fake = _FakeAk(hist_frame=_FakeFrame(DAILY_ROWS))
    provider = _make_provider(fake)
    with pytest.raises(AkshareProviderError, match="不支持周期"):
        provider.get_bar_history("600519", "SSE", "3m", datetime(2024, 1, 1))
    assert not fake.hist_calls and not fake.min_calls


def test_unsupported_exchange_raises() -> None:
    fake = _FakeAk(hist_frame=_FakeFrame(DAILY_ROWS))
    provider = _make_provider(fake)
    with pytest.raises(AkshareProviderError, match="不支持交易所"):
        provider.get_bar_history("IF2406", "CFFEX", "d", datetime(2024, 1, 1))
    assert not fake.hist_calls


def test_empty_frame_raises() -> None:
    fake = _FakeAk(hist_frame=_FakeFrame([]))
    provider = _make_provider(fake)
    with pytest.raises(AkshareProviderError, match="未返回"):
        provider.get_bar_history("600519", "SSE", "d", datetime(2024, 1, 1))


def test_exception_raises_provider_error(monkeypatch) -> None:
    monkeypatch.setattr("aitrade.datasource.akshare_provider.AKSHARE_MAX_RETRIES", 1)
    fake = _FakeAk(raise_on={"hist"})
    provider = _make_provider(fake)
    with pytest.raises(AkshareProviderError, match="连接失败"):
        provider.get_bar_history("600519", "SSE", "d", datetime(2024, 1, 1))


def test_not_inited_raises() -> None:
    provider = AkshareProvider()  # 未初始化
    with pytest.raises(AkshareProviderError, match="未初始化"):
        provider.get_bar_history("600519", "SSE", "d", datetime(2024, 1, 1))


def test_supported_categories_and_status() -> None:
    fake = _FakeAk()
    provider = _make_provider(fake)
    assert DataCategory.CONTRACT in provider.get_supported_categories()
    assert DataCategory.BAR_HISTORY in provider.get_supported_categories()
    assert provider.get_status() == ProviderStatus.AVAILABLE


def test_contracts_mapped_from_akshare_code_name() -> None:
    fake = _FakeAk(contract_frame=_FakeFrame(CONTRACT_ROWS))
    provider = _make_provider(fake)

    contracts = provider.get_contracts(exchange="SSE")

    assert contracts is not None and len(contracts) == 1
    assert contracts[0].symbol == "600519"
    assert contracts[0].exchange == "SSE"
    assert contracts[0].name == "贵州茅台"
    assert contracts[0].product_type == "股票"
    assert contracts[0].min_volume == 100
    assert fake.contract_calls == 1


def test_get_contract_returns_single_match() -> None:
    fake = _FakeAk(contract_frame=_FakeFrame(CONTRACT_ROWS))
    provider = _make_provider(fake)

    contract = provider.get_contract("000001", "SZSE")

    assert contract is not None
    assert contract.symbol == "000001"
    assert contract.exchange == "SZSE"


def test_manager_routes_to_akshare_by_name() -> None:
    manager = DataSourceManager()
    fake = _FakeAk(hist_frame=_FakeFrame(DAILY_ROWS))
    manager.register(_make_provider(fake), priority=10)

    records = manager.get_bar_history(
        "600519", "SSE", "d", datetime(2024, 1, 1), provider_name="akshare"
    )
    assert records and records[0].symbol == "600519"


def test_manager_reraises_when_explicit_provider_fails(monkeypatch) -> None:
    monkeypatch.setattr("aitrade.datasource.akshare_provider.AKSHARE_MAX_RETRIES", 1)
    manager = DataSourceManager()
    fake = _FakeAk(raise_on={"hist"})
    manager.register(_make_provider(fake), priority=10)

    with pytest.raises(AkshareProviderError, match="连接失败"):
        manager.get_bar_history(
            "600519", "SSE", "d", datetime(2024, 1, 1), provider_name="akshare"
        )


class _UnavailableProvider(BaseProvider):
    name = "tushare"
    display_name = "stub"

    def init(self, output=print) -> bool:
        return False

    def get_status(self) -> ProviderStatus:
        return ProviderStatus.UNAVAILABLE

    def get_supported_categories(self) -> list[DataCategory]:
        return [DataCategory.BAR_HISTORY]


def test_pick_bar_provider_falls_back_to_akshare(monkeypatch) -> None:
    from aitrade.api import alpha_service

    manager = DataSourceManager()
    manager.register(_UnavailableProvider(), priority=0)
    fake = _FakeAk(hist_frame=_FakeFrame(DAILY_ROWS))
    manager.register(_make_provider(fake), priority=10)

    monkeypatch.setattr(alpha_service, "datasource_manager", manager)

    # tushare 不可用时自动选择 akshare。
    assert alpha_service._pick_bar_provider() == "akshare"
    # 显式指定可用数据源时应优先使用。
    assert alpha_service._pick_bar_provider("akshare") == "akshare"
    # 显式指定不可用数据源时回退到自动选择。
    assert alpha_service._pick_bar_provider("tushare") == "akshare"
    # mock 不能作为真实下载数据源。
    assert not alpha_service._is_usable_bar_provider("mock")
