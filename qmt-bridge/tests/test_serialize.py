"""bars 列式序列化 round-trip 测试。"""

from datetime import datetime

import polars as pl

from qmt_bridge.contract import BARS_COLUMNS
from qmt_bridge.serialize import bars_to_ipc


def test_bars_to_ipc_roundtrip():
    rows = [{
        "symbol": "600000", "exchange": "SSE",
        "datetime": datetime(2024, 1, 2), "interval": "d",
        "open_price": 10.0, "high_price": 11.0, "low_price": 9.8, "close_price": 10.5,
        "volume": 1000.0, "turnover": 10500.0, "open_interest": 0.0, "adjust_type": "hfq",
    }]
    blob = bars_to_ipc(rows)
    assert isinstance(blob, (bytes, bytearray))

    df = pl.read_ipc_stream(blob)
    assert df.columns == BARS_COLUMNS
    assert df.height == 1
    assert df["close_price"][0] == 10.5
    assert df["symbol"][0] == "600000"


def test_bars_to_ipc_empty():
    blob = bars_to_ipc([])
    df = pl.read_ipc_stream(blob)
    assert df.columns == BARS_COLUMNS
    assert df.height == 0
