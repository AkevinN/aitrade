"""合约/交易日历/基本面归一化测试。"""

import pandas as pd

from qmt_bridge.xtdata_client import XtdataClient


class FakeXtMeta:
    def download_sector_data(self):
        return None

    def get_stock_list_in_sector(self, sector_name, real_timetag=None):
        assert sector_name in ("沪深A股", "沪深京A股")
        return ["600000.SH", "000001.SZ"]

    def get_instrument_detail(self, stock_code, iscomplete=False):
        return {
            "InstrumentName": "浦发银行" if stock_code.startswith("600000") else "平安银行",
            "OpenDate": "19991110",
            "ExpireDate": "",
            "PriceTick": 0.01,
            "VolumeMultiple": 1,
            "InstrumentStatus": 0,
            "IsTrading": True,
        }

    def get_instrument_type(self, stock_code):
        return {"stock": True}

    def download_holiday_data(self):
        return None

    def get_trading_calendar(self, market, start_time="", end_time=""):
        return ["20240102", "20240103"]

    def download_financial_data2(self, stock_list, table_list=None, start_time="", end_time="", callback=None):
        return None

    def get_financial_data(self, stock_list, table_list=None, start_time="", end_time="", report_type="report_time"):
        assert report_type == "announce_time"
        df = pd.DataFrame({"m_timetag": ["20231231"], "m_anntime": ["20240328"], "tot_assets": [1.0e12]})
        return {stock_list[0]: {"Balance": df}}


def test_list_contracts_normalizes_suffix_and_fields():
    rows = XtdataClient(xtdata=FakeXtMeta()).get_contracts(include_bse=False)
    assert {r["symbol"] for r in rows} == {"600000", "000001"}
    pf = next(r for r in rows if r["symbol"] == "600000")
    assert pf["exchange"] == "SSE"
    assert pf["name"] == "浦发银行"
    assert pf["list_date"] == "19991110"
    assert pf["pricetick"] == 0.01


def test_trading_calendar():
    rows = XtdataClient(xtdata=FakeXtMeta()).get_trade_calendar("SSE", "20240101", "20240105")
    assert rows == [
        {"date": "20240102", "exchange": "SSE", "is_open": True},
        {"date": "20240103", "exchange": "SSE", "is_open": True},
    ]


def test_fundamental_uses_announce_time():
    rows = XtdataClient(xtdata=FakeXtMeta()).get_fundamental("600000", "SSE", "20230101", "20240401")
    assert rows[0]["symbol"] == "600000"
    assert rows[0]["report_period"] == "20231231"
    assert rows[0]["ann_date"] == "20240328"
    assert rows[0]["table"] == "Balance"
    assert rows[0]["fields"]["tot_assets"] == 1.0e12
