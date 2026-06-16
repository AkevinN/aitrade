"""
AkshareProvider 转债行情分派测试（Task 4.2）。

覆盖：
1. 转债代码正确分派到 bond_zh_hs_cov_daily（不调 stock_zh_a_hist）
2. 列映射正确：date→datetime / open/high/low/close/volume；turnover 置 0
3. symbol 前缀组装：SSE → "sh113050"，SZSE → "sz128093"
4. KeyError('date') 转换为含"无历史行情"字样的 AkshareProviderError
5. 非转债代码（600519.SSE）不受影响，仍走 stock_zh_a_hist
6. 周期校验：转债代码传分钟线周期抛 AkshareProviderError
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aitrade.datasource.akshare_provider import AkshareProvider, AkshareProviderError


# ---------------------------------------------------------------------------
# Fake akshare 模块
# ---------------------------------------------------------------------------


class _FakeFrame:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def to_dict(self, orient: str = "records") -> list[dict]:
        assert orient == "records"
        return list(self._rows)


class _FakeAk:
    """模拟 akshare 模块，同时支持 stock_zh_a_hist 和 bond_zh_hs_cov_daily。"""

    def __init__(
        self,
        hist_frame=None,
        cb_frame=None,
        raise_on_hist: bool = False,
        raise_cb_key_error: bool = False,
    ) -> None:
        self._hist_frame = hist_frame
        self._cb_frame = cb_frame
        self._raise_on_hist = raise_on_hist
        self._raise_cb_key_error = raise_cb_key_error
        # 调用记录
        self.hist_calls: list[dict] = []
        self.cb_calls: list[dict] = []

    def stock_zh_a_hist(self, **kwargs):
        self.hist_calls.append(kwargs)
        if self._raise_on_hist:
            raise RuntimeError("stock_zh_a_hist boom")
        return self._hist_frame

    def bond_zh_hs_cov_daily(self, **kwargs):
        self.cb_calls.append(kwargs)
        if self._raise_cb_key_error:
            raise KeyError("date")  # 未上市新债陷阱
        return self._cb_frame

    # 其他方法 stub（不应被调用）
    def stock_zh_a_hist_min_em(self, **kwargs):
        raise AssertionError("不应调用分钟线接口")

    def stock_info_a_code_name(self):
        raise AssertionError("不应调用合约列表接口")


def _make_provider(fake_ak: _FakeAk) -> AkshareProvider:
    provider = AkshareProvider()
    provider._inited = True
    provider._ak = fake_ak
    return provider


# 转债日线数据
CB_DAILY_ROWS = [
    {"date": "2024-01-02", "open": 108.5, "high": 109.0, "low": 107.8, "close": 108.7, "volume": 5000.0},
    {"date": "2024-01-03", "open": 108.7, "high": 110.2, "low": 108.0, "close": 109.5, "volume": 6200.0},
]

# A 股日线数据（用于非转债测试）
STOCK_DAILY_ROWS = [
    {"日期": "2024-01-02", "开盘": 10.0, "收盘": 10.5, "最高": 10.8, "最低": 9.9, "成交量": 1000, "成交额": 1_050_000.0},
]


# ---------------------------------------------------------------------------
# 测试 1：转债代码分派到 bond_zh_hs_cov_daily
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol,exchange,expected_symbol_prefix",
    [
        ("113050", "SSE", "sh113050"),    # 上交所转债
        ("128093", "SZSE", "sz128093"),   # 深交所转债
        ("110059", "SSE", "sh110059"),    # 110 前缀
        ("123119", "SZSE", "sz123119"),   # 123 前缀
        ("127043", "SZSE", "sz127043"),   # 127 前缀
    ],
)
def test_cb_symbol_routes_to_bond_daily(symbol, exchange, expected_symbol_prefix) -> None:
    """转债代码应路由到 bond_zh_hs_cov_daily，不调 stock_zh_a_hist。"""
    fake = _FakeAk(cb_frame=_FakeFrame(CB_DAILY_ROWS))
    provider = _make_provider(fake)

    records = provider.get_bar_history(
        symbol, exchange, "d", datetime(2024, 1, 1), datetime(2024, 2, 1)
    )

    assert records is not None and len(records) == 2
    # 调用了转债接口
    assert len(fake.cb_calls) == 1
    assert fake.cb_calls[0]["symbol"] == expected_symbol_prefix
    # 未调股票接口
    assert len(fake.hist_calls) == 0


# ---------------------------------------------------------------------------
# 测试 2：列映射正确
# ---------------------------------------------------------------------------


def test_cb_column_mapping() -> None:
    """bond_zh_hs_cov_daily 的列应正确映射：date→datetime，turnover=0。"""
    fake = _FakeAk(cb_frame=_FakeFrame(CB_DAILY_ROWS))
    provider = _make_provider(fake)

    records = provider.get_bar_history(
        "113050", "SSE", "d", datetime(2024, 1, 1), datetime(2024, 2, 1)
    )

    assert records is not None and len(records) == 2

    first = records[0]
    assert first.symbol == "113050"
    assert first.exchange == "SSE"
    assert first.interval == "d"
    assert first.datetime == datetime(2024, 1, 2)
    assert first.open_price == pytest.approx(108.5)
    assert first.high_price == pytest.approx(109.0)
    assert first.low_price == pytest.approx(107.8)
    assert first.close_price == pytest.approx(108.7)
    assert first.volume == pytest.approx(5000.0)
    assert first.turnover == pytest.approx(0.0)   # 无成交额字段，置 0
    assert first.open_interest == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 测试 3：KeyError('date') → AkshareProviderError（含"无历史行情"）
# ---------------------------------------------------------------------------


def test_cb_key_error_date_raises_friendly_error(monkeypatch) -> None:
    """未上市新债 KeyError('date') 应被捕获并转为含"无历史行情"的 AkshareProviderError。"""
    monkeypatch.setattr("aitrade.datasource.akshare_provider.AKSHARE_MAX_RETRIES", 1)
    fake = _FakeAk(raise_cb_key_error=True)
    provider = _make_provider(fake)

    with pytest.raises(AkshareProviderError, match="无历史行情"):
        provider.get_bar_history(
            "113050", "SSE", "d", datetime(2024, 1, 1)
        )

    # 调用了转债接口，未调股票接口
    assert len(fake.cb_calls) >= 1
    assert len(fake.hist_calls) == 0


# ---------------------------------------------------------------------------
# 测试 4：非转债代码不受影响（仍走 stock_zh_a_hist）
# ---------------------------------------------------------------------------


def test_non_cb_symbol_routes_to_stock_hist() -> None:
    """600519.SSE 属于 A 股，不属于转债代码段，应走 stock_zh_a_hist。"""
    fake = _FakeAk(hist_frame=_FakeFrame(STOCK_DAILY_ROWS))
    provider = _make_provider(fake)

    records = provider.get_bar_history(
        "600519", "SSE", "d", datetime(2024, 1, 1), datetime(2024, 2, 1)
    )

    assert records is not None and len(records) == 1
    assert len(fake.hist_calls) == 1
    assert len(fake.cb_calls) == 0


# ---------------------------------------------------------------------------
# 测试 5：转债代码传分钟线周期 → AkshareProviderError
# ---------------------------------------------------------------------------


def test_cb_symbol_minute_interval_raises() -> None:
    """转债代码传分钟线周期（1m/5m）应抛出 AkshareProviderError。"""
    fake = _FakeAk(cb_frame=_FakeFrame(CB_DAILY_ROWS))
    provider = _make_provider(fake)

    with pytest.raises(AkshareProviderError, match="仅支持日线"):
        provider.get_bar_history(
            "113050", "SSE", "5m", datetime(2024, 1, 1)
        )

    # 不应实际调用任何接口
    assert len(fake.cb_calls) == 0
    assert len(fake.hist_calls) == 0


# ---------------------------------------------------------------------------
# 测试 6：vt_symbol 格式（113050.SSE）正确分派
# ---------------------------------------------------------------------------


def test_cb_vt_symbol_format_routes_correctly() -> None:
    """传 '113050.SSE' vt_symbol 格式也能正确分派到转债接口。"""
    fake = _FakeAk(cb_frame=_FakeFrame(CB_DAILY_ROWS))
    provider = _make_provider(fake)

    records = provider.get_bar_history(
        "113050.SSE", "", "d", datetime(2024, 1, 1), datetime(2024, 2, 1)
    )

    assert records is not None and len(records) == 2
    assert fake.cb_calls[0]["symbol"] == "sh113050"
    assert len(fake.hist_calls) == 0


# ---------------------------------------------------------------------------
# 测试 7：空 DataFrame 返回 AkshareProviderError（区间无数据）
# ---------------------------------------------------------------------------


def test_cb_empty_frame_raises(monkeypatch) -> None:
    """转债接口返回空 DataFrame 时应抛 AkshareProviderError。"""
    monkeypatch.setattr("aitrade.datasource.akshare_provider.AKSHARE_MAX_RETRIES", 1)
    fake = _FakeAk(cb_frame=_FakeFrame([]))
    provider = _make_provider(fake)

    with pytest.raises(AkshareProviderError):
        provider.get_bar_history(
            "113050", "SSE", "d", datetime(2024, 1, 1), datetime(2024, 2, 1)
        )
