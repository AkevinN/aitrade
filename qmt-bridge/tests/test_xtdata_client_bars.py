"""xtdata 封装模块的 K 线归一化测试（注入假 xtdata，不连真 QMT）。"""

from datetime import datetime

import pandas as pd
import pytest

from qmt_bridge.xtdata_client import XtdataClient


class FakeXtdata:
    """模拟 xtquant.xtdata：记录调用、返回 get_market_data_ex 形状的数据。"""

    def __init__(self) -> None:
        self.download_calls = []

    def download_history_data2(self, stock_list, period, start_time="", end_time="", callback=None, incrementally=None):
        self.download_calls.append((tuple(stock_list), period, start_time, end_time))
        return None

    def get_market_data_ex(self, field_list, stock_list, period="1d", start_time="", end_time="",
                           count=-1, dividend_type="none", fill_data=True):
        code = stock_list[0]
        df = pd.DataFrame(
            {
                "time": [1704124800000, 1704211200000],
                "open": [10.0, 10.5],
                "high": [11.0, 10.8],
                "low": [9.8, 10.2],
                "close": [10.5, 10.6],
                "volume": [1000.0, 1200.0],
                "amount": [10500.0, 12720.0],
                "openInterest": [0.0, 0.0],
            }
        )
        return {code: df}


def test_get_bars_two_step_and_normalize():
    fake = FakeXtdata()
    client = XtdataClient(xtdata=fake)
    rows = client.get_bars("600000", "SSE", "d", "20240101", "20240131", adjust_type="hfq")

    assert fake.download_calls == [(("600000.SH",), "1d", "20240101", "20240131")]

    assert len(rows) == 2
    first = rows[0]
    assert first["symbol"] == "600000"
    assert first["exchange"] == "SSE"
    assert first["interval"] == "d"
    assert first["open_price"] == 10.0
    assert first["close_price"] == 10.5
    assert first["turnover"] == 10500.0
    assert first["open_interest"] == 0.0
    assert first["adjust_type"] == "hfq"
    assert isinstance(first["datetime"], datetime)
    assert first["datetime"].year == 2024 and first["datetime"].month == 1 and first["datetime"].day == 2


def test_get_bars_passes_back_dividend_for_hfq():
    fake = FakeXtdata()
    captured = {}
    orig = fake.get_market_data_ex

    def spy(field_list, stock_list, **kw):
        captured.update(kw)
        return orig(field_list, stock_list, **kw)

    fake.get_market_data_ex = spy
    XtdataClient(xtdata=fake).get_bars("600000", "SSE", "d", "20240101", "20240131", adjust_type="hfq")
    assert captured["dividend_type"] == "back"


def test_get_bars_empty_returns_empty_list():
    class EmptyXt(FakeXtdata):
        def get_market_data_ex(self, *a, **k):
            return {"600000.SH": pd.DataFrame()}

    rows = XtdataClient(xtdata=EmptyXt()).get_bars("600000", "SSE", "d", "20240101", "20240131", adjust_type="hfq")
    assert rows == []
