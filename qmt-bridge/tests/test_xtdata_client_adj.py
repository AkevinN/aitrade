"""复权因子累乘测试：get_divid_factors 的 dr -> 后复权累积因子。"""

import pandas as pd

from qmt_bridge.xtdata_client import XtdataClient


class FakeXtAdj:
    def get_divid_factors(self, stock_code, start_time="", end_time=""):
        return pd.DataFrame(
            {"dr": [1.10, 1.05]},
            index=["20240110", "20240620"],
        )


def test_adj_factor_cumulative_product():
    rows = XtdataClient(xtdata=FakeXtAdj()).get_adj_factor("600000", "SSE", "20240101", "20241231")
    assert rows == [
        {"trade_date": "20240110", "adj_factor": 1.10},
        {"trade_date": "20240620", "adj_factor": round(1.10 * 1.05, 6)},
    ]


def test_adj_factor_no_dividends_returns_empty():
    class Empty:
        def get_divid_factors(self, *a, **k):
            return pd.DataFrame()

    assert XtdataClient(xtdata=Empty()).get_adj_factor("600000", "SSE", "", "") == []
