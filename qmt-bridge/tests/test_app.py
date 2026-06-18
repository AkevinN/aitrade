"""FastAPI 路由集成测试（TestClient + 假 XtdataClient，不连真 QMT）。"""

import polars as pl
from datetime import datetime
from fastapi.testclient import TestClient

from qmt_bridge.app import create_app
from qmt_bridge.contract import BARS_COLUMNS

TOKEN = "t0ken"
H = {"Authorization": f"Bearer {TOKEN}"}


class FakeClient:
    def is_connected(self):
        return True

    def get_bars(self, symbol, exchange, interval, start, end, *, adjust_type="hfq"):
        return [{
            "symbol": symbol, "exchange": exchange, "datetime": datetime(2024, 1, 2),
            "interval": interval, "open_price": 10.0, "high_price": 11.0,
            "low_price": 9.8, "close_price": 10.5, "volume": 1000.0,
            "turnover": 10500.0, "open_interest": 0.0, "adjust_type": adjust_type,
        }]

    def get_contracts(self, *, include_bse=False):
        return [{"symbol": "600000", "exchange": "SSE", "name": "浦发银行",
                 "product_type": "股票", "size": 1.0, "pricetick": 0.01,
                 "list_date": "19991110", "delist_date": "", "extra": {}}]

    def get_trade_calendar(self, exchange, start, end):
        return [{"date": "20240102", "exchange": exchange, "is_open": True}]

    def get_adj_factor(self, symbol, exchange, start="", end=""):
        return [{"trade_date": "20240110", "adj_factor": 1.1}]

    def get_fundamental(self, symbol, exchange, start, end):
        return [{"symbol": symbol, "exchange": exchange, "table": "Balance",
                 "report_period": "20231231", "ann_date": "20240328", "fields": {}}]


def _client():
    app = create_app(client=FakeClient(), token=TOKEN)
    return TestClient(app)


def test_health_no_auth_required():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json()["connected"] is True


def test_bars_returns_arrow():
    r = _client().post("/bars", json={
        "symbol": "600000", "exchange": "SSE", "interval": "d",
        "start": "20240101", "end": "20240131", "adjust_type": "hfq",
    }, headers=H)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/vnd.apache.arrow.stream"
    df = pl.read_ipc_stream(r.content)
    assert df.columns == BARS_COLUMNS
    assert df["close_price"][0] == 10.5


def test_bars_requires_token():
    r = _client().post("/bars", json={
        "symbol": "600000", "exchange": "SSE", "interval": "d",
        "start": "20240101", "end": "20240131", "adjust_type": "hfq",
    })
    assert r.status_code in (401, 403)


def test_contracts_json():
    r = _client().get("/contracts", headers=H)
    assert r.status_code == 200
    assert r.json()[0]["symbol"] == "600000"


def test_calendar_and_adj_and_fundamental():
    c = _client()
    assert c.get("/trading_calendar", params={"exchange": "SSE", "start": "20240101", "end": "20240105"}, headers=H).json()[0]["date"] == "20240102"
    assert c.get("/adj_factor", params={"symbol": "600000", "exchange": "SSE"}, headers=H).json()[0]["adj_factor"] == 1.1
    assert c.get("/fundamental", params={"symbol": "600000", "exchange": "SSE", "start": "20230101", "end": "20240401"}, headers=H).json()[0]["table"] == "Balance"
