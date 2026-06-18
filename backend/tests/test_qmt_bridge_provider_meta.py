"""QmtBridgeProvider 元数据方法测试（注入假 httpx）。"""

from aitrade.datasource.qmt_bridge_provider import QmtBridgeProvider
from aitrade.datasource.types import ContractInfo, CalendarDay


class _Resp:
    def __init__(self, json_data):
        self._json = json_data
        self.status_code = 200

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


class _Http:
    def __init__(self, responses):
        self._responses = responses

    def get(self, path, params=None):
        return self._responses[path]


def _p(responses):
    p = QmtBridgeProvider(url="http://win", token="t")
    p._http = _Http(responses)
    p._inited = True
    return p


def test_get_contracts():
    p = _p({"/contracts": _Resp([{
        "symbol": "600000", "exchange": "SSE", "name": "浦发银行",
        "product_type": "股票", "size": 1.0, "pricetick": 0.01,
        "list_date": "19991110", "delist_date": "", "extra": {},
    }])})
    cs = p.get_contracts()
    assert isinstance(cs[0], ContractInfo)
    assert cs[0].symbol == "600000" and cs[0].name == "浦发银行"


def test_get_trade_calendar():
    p = _p({"/trading_calendar": _Resp([{"date": "20240102", "exchange": "SSE", "is_open": True}])})
    cal = p.get_trade_calendar("SSE", "20240101", "20240105")
    assert isinstance(cal[0], CalendarDay)
    assert cal[0].date == "20240102" and cal[0].is_open is True


def test_get_adj_factor():
    p = _p({"/adj_factor": _Resp([{"trade_date": "20240110", "adj_factor": 1.1}])})
    af = p.get_adj_factor("600000", "SSE")
    assert af == [{"trade_date": "20240110", "adj_factor": 1.1}]


def test_get_contracts_empty_returns_none():
    p = _p({"/contracts": _Resp([])})
    assert p.get_contracts() is None
