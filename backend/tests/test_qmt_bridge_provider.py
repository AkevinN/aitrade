"""QmtBridgeProvider 单元测试（注入假 httpx，不联网）。"""

from datetime import datetime

import polars as pl
import pytest

from aitrade.datasource.qmt_bridge_provider import QmtBridgeProvider
from aitrade.datasource.types import BarRecord, DataCategory, ProviderStatus


BARS_COLUMNS = [
    "symbol", "exchange", "datetime", "interval",
    "open_price", "high_price", "low_price", "close_price",
    "volume", "turnover", "open_interest", "adjust_type",
]


def _arrow_one_bar() -> bytes:
    df = pl.DataFrame([{
        "symbol": "600000", "exchange": "SSE", "datetime": datetime(2024, 1, 2),
        "interval": "d", "open_price": 10.0, "high_price": 11.0, "low_price": 9.8,
        "close_price": 10.5, "volume": 1000.0, "turnover": 10500.0,
        "open_interest": 0.0, "adjust_type": "hfq",
    }]).select(BARS_COLUMNS)
    return df.write_ipc_stream(None, compression="zstd").getvalue()


class _FakeResp:
    def __init__(self, status_code=200, content=b"", json_data=None):
        self.status_code = status_code
        self.content = content
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHttp:
    """模拟 httpx.Client：按 path 返回预置响应。"""

    def __init__(self, responses: dict):
        self._responses = responses
        self.calls = []

    def post(self, path, json=None):
        self.calls.append(("POST", path, json))
        return self._responses[path]

    def get(self, path, params=None):
        self.calls.append(("GET", path, params))
        return self._responses[path]


def _provider(responses, url="http://win:58610", token="t"):
    p = QmtBridgeProvider(url=url, token=token)
    p._http = _FakeHttp(responses)
    p._inited = True
    return p


def test_status_not_configured_without_url():
    p = QmtBridgeProvider(url="", token="")
    assert p.get_status() == ProviderStatus.NOT_CONFIGURED


def test_supported_categories():
    p = QmtBridgeProvider(url="http://win", token="t")
    cats = p.get_supported_categories()
    assert DataCategory.BAR_HISTORY in cats
    assert DataCategory.REFERENCE in cats


def test_get_bar_history_decodes_arrow_to_barrecord():
    p = _provider({"/bars": _FakeResp(content=_arrow_one_bar())})
    bars = p.get_bar_history("600000", "SSE", "d", datetime(2024, 1, 1), datetime(2024, 1, 31))
    assert isinstance(bars, list) and len(bars) == 1
    b = bars[0]
    assert isinstance(b, BarRecord)
    assert b.symbol == "600000" and b.exchange == "SSE"
    assert b.close_price == 10.5
    assert b.adjust_type == "hfq"
    assert p._http.calls[0] == ("POST", "/bars", {
        "symbol": "600000", "exchange": "SSE", "interval": "d",
        "start": "20240101", "end": "20240131", "adjust_type": "hfq",
    })


def test_get_bar_history_empty_returns_none():
    empty = pl.DataFrame(schema={c: (pl.Utf8 if c in ("symbol","exchange","interval","adjust_type")
                                      else pl.Datetime if c == "datetime" else pl.Float64)
                                  for c in BARS_COLUMNS}).write_ipc_stream(None, compression="zstd").getvalue()
    p = _provider({"/bars": _FakeResp(content=empty)})
    assert p.get_bar_history("600000", "SSE", "d", datetime(2024, 1, 1)) is None


def test_get_bar_history_http_error_raises():
    p = _provider({"/bars": _FakeResp(status_code=500)})
    with pytest.raises(RuntimeError):
        p.get_bar_history("600000", "SSE", "d", datetime(2024, 1, 1))
