"""契约常量与代码/周期/复权映射的单元测试。"""

import pytest

from qmt_bridge.contract import (
    BARS_COLUMNS,
    to_qmt_code,
    from_qmt_code,
    to_qmt_period,
    to_dividend_type,
)


def test_bars_columns_match_barrecord_fields():
    assert BARS_COLUMNS == [
        "symbol", "exchange", "datetime", "interval",
        "open_price", "high_price", "low_price", "close_price",
        "volume", "turnover", "open_interest", "adjust_type",
    ]


@pytest.mark.parametrize("symbol,exchange,expected", [
    ("600000", "SSE", "600000.SH"),
    ("000001", "SZSE", "000001.SZ"),
    ("430047", "BSE", "430047.BJ"),
])
def test_to_qmt_code(symbol, exchange, expected):
    assert to_qmt_code(symbol, exchange) == expected


@pytest.mark.parametrize("code,expected", [
    ("600000.SH", ("600000", "SSE")),
    ("000001.SZ", ("000001", "SZSE")),
    ("430047.BJ", ("430047", "BSE")),
])
def test_from_qmt_code(code, expected):
    assert from_qmt_code(code) == expected


@pytest.mark.parametrize("interval,expected", [
    ("d", "1d"), ("1m", "1m"), ("30m", "30m"),
    ("1h", "1h"), ("60m", "1h"), ("w", "1w"),
])
def test_to_qmt_period(interval, expected):
    assert to_qmt_period(interval) == expected


def test_to_qmt_period_rejects_unknown():
    with pytest.raises(ValueError):
        to_qmt_period("3m")


def test_to_dividend_type_default():
    assert to_dividend_type("none") == "none"
    assert to_dividend_type("qfq") == "front"
    assert to_dividend_type("hfq") == "back"


def test_to_dividend_type_ratio_mode():
    assert to_dividend_type("hfq", ratio=True) == "back_ratio"
    assert to_dividend_type("qfq", ratio=True) == "front_ratio"
    assert to_dividend_type("none", ratio=True) == "none"
