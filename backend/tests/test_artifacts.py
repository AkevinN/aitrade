"""回测产物序列化模块单元测试。

覆盖 ``serialize_trades`` 与 ``serialize_equity_curve`` 两个纯数据搬运函数：
- 多种入参形态（dict / list / None / 空）；
- 时间格式（成交 ISO、净值 YYYY-MM-DD）；
- 缺列 / 爆仓等边界返回 ``[]``；
- 输出按时间升序。

仅构造内存对象，不读本地数据、不跑回测引擎。
对应需求：1.3、2.3。
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl

from aitrade.backtest.artifacts import (
    serialize_equity_curve,
    serialize_trades,
    extract_benchmark_prices,
    attach_benchmark_returns,
    summarize_benchmark,
)
from aitrade.backtest.types import Direction, Offset, TradeData


def _make_trade(
    *,
    symbol: str = "AAA",
    exchange: str = "SSE",
    tradeid: str = "1",
    direction: str = Direction.LONG,
    offset: str = Offset.OPEN,
    price: float = 100.0,
    volume: float = 10.0,
    dt: datetime,
) -> TradeData:
    """构造最小 TradeData，便于各用例复用。"""
    return TradeData(
        symbol=symbol,
        exchange=exchange,
        orderid="o1",
        tradeid=tradeid,
        direction=direction,
        offset=offset,
        price=price,
        volume=volume,
        datetime=dt,
    )


# ---------------------------------------------------------------------------
# serialize_trades
# ---------------------------------------------------------------------------


def test_serialize_trades_list_input():
    """list 入参：逐字段裁剪，direction/offset 为小写字符串，价量为 float。"""
    trades = [
        _make_trade(
            tradeid="1",
            direction=Direction.LONG,
            offset=Offset.OPEN,
            price=10.5,
            volume=100,
            dt=datetime(2024, 1, 2, 9, 30, 0),
        )
    ]

    out = serialize_trades(trades)

    assert out == [
        {
            "datetime": "2024-01-02T09:30:00",
            "vt_symbol": "AAA.SSE",
            "direction": "long",
            "offset": "open",
            "price": 10.5,
            "volume": 100.0,
        }
    ]
    # 价量须为 float 类型，保证 JSON 序列化稳定
    assert isinstance(out[0]["price"], float)
    assert isinstance(out[0]["volume"], float)


def test_serialize_trades_dict_input():
    """dict 入参（引擎内部存储形态）：取 values 后序列化，结果与 list 等价。"""
    trades = {
        "t1": _make_trade(tradeid="1", dt=datetime(2024, 1, 2, 9, 30, 0)),
    }

    out = serialize_trades(trades)

    assert len(out) == 1
    assert out[0]["vt_symbol"] == "AAA.SSE"
    assert out[0]["datetime"] == "2024-01-02T09:30:00"


def test_serialize_trades_iso_datetime_format():
    """datetime 输出为 ISO 8601 字符串，可被前端直接解析。"""
    trades = [_make_trade(dt=datetime(2024, 3, 15, 14, 5, 30))]

    out = serialize_trades(trades)

    assert out[0]["datetime"] == "2024-03-15T14:05:30"


def test_serialize_trades_empty_and_none():
    """空 list / 空 dict / None 均返回空列表，不报错。"""
    assert serialize_trades([]) == []
    assert serialize_trades({}) == []
    assert serialize_trades(None) == []


def test_serialize_trades_sorted_ascending():
    """无论入参顺序如何，输出按 datetime 升序排列。"""
    trades = [
        _make_trade(tradeid="3", dt=datetime(2024, 1, 3, 10, 0, 0)),
        _make_trade(tradeid="1", dt=datetime(2024, 1, 1, 10, 0, 0)),
        _make_trade(tradeid="2", dt=datetime(2024, 1, 2, 10, 0, 0)),
    ]

    out = serialize_trades(trades)

    datetimes = [r["datetime"] for r in out]
    assert datetimes == [
        "2024-01-01T10:00:00",
        "2024-01-02T10:00:00",
        "2024-01-03T10:00:00",
    ]


def test_serialize_trades_short_close():
    """direction/offset 透传：short / close 同样输出小写字符串。"""
    trades = [
        _make_trade(
            direction=Direction.SHORT,
            offset=Offset.CLOSE,
            dt=datetime(2024, 1, 2, 9, 30, 0),
        )
    ]

    out = serialize_trades(trades)

    assert out[0]["direction"] == "short"
    assert out[0]["offset"] == "close"


# ---------------------------------------------------------------------------
# serialize_equity_curve
# ---------------------------------------------------------------------------


def test_serialize_equity_curve_normal():
    """正常序列化：逐日字段裁剪，date 为 YYYY-MM-DD，数值为 float。"""
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "balance": [1_000_000.0, 1_010_000.0],
            "drawdown": [0.0, -5_000.0],
            "ddpercent": [0.0, -0.5],
            "net_pnl": [0.0, 10_000.0],
        }
    )

    out = serialize_equity_curve(df)

    assert out == [
        {
            "date": "2024-01-01",
            "balance": 1_000_000.0,
            "drawdown": 0.0,
            "ddpercent": 0.0,
            "net_pnl": 0.0,
        },
        {
            "date": "2024-01-02",
            "balance": 1_010_000.0,
            "drawdown": -5_000.0,
            "ddpercent": -0.5,
            "net_pnl": 10_000.0,
        },
    ]


def test_serialize_equity_curve_date_format():
    """date 列（datetime.date）输出为 YYYY-MM-DD 字符串。"""
    df = pl.DataFrame(
        {
            "date": [date(2024, 12, 31)],
            "balance": [1_000_000.0],
            "drawdown": [0.0],
            "ddpercent": [0.0],
            "net_pnl": [0.0],
        }
    )

    out = serialize_equity_curve(df)

    assert out[0]["date"] == "2024-12-31"


def test_serialize_equity_curve_missing_balance_column():
    """缺 balance 列（爆仓场景）返回空列表，不报错。"""
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)],
            "net_pnl": [0.0],
        }
    )

    assert serialize_equity_curve(df) == []


def test_serialize_equity_curve_none():
    """None 入参返回空列表。"""
    assert serialize_equity_curve(None) == []


def test_serialize_equity_curve_empty_df():
    """空 DataFrame 返回空列表。"""
    empty = pl.DataFrame()
    assert serialize_equity_curve(empty) == []


# ---------------------------------------------------------------------------
# 基准（买入持有标的）与超额收益
# ---------------------------------------------------------------------------


class _FakeDailyResult:
    """模拟 PortfolioDailyResult：仅需 close_prices 字段。"""

    def __init__(self, close_prices: dict[str, float]) -> None:
        self.close_prices = close_prices


def test_extract_benchmark_prices_normal():
    """从逐日盯市结果提取基准标的收盘价，键为 YYYY-MM-DD。"""
    daily_results = {
        date(2024, 1, 2): _FakeDailyResult({"AAA.SSE": 10.0, "BBB.SSE": 5.0}),
        date(2024, 1, 3): _FakeDailyResult({"AAA.SSE": 11.0, "BBB.SSE": 5.5}),
    }

    prices = extract_benchmark_prices(daily_results, "AAA.SSE")

    assert prices == {"2024-01-02": 10.0, "2024-01-03": 11.0}


def test_extract_benchmark_prices_skips_invalid_and_missing():
    """缺基准标的或收盘价无效（None/<=0）的交易日被跳过。"""
    daily_results = {
        date(2024, 1, 2): _FakeDailyResult({"AAA.SSE": 0.0}),       # 无效价
        date(2024, 1, 3): _FakeDailyResult({"BBB.SSE": 5.0}),       # 缺基准
        date(2024, 1, 4): _FakeDailyResult({"AAA.SSE": 12.0}),      # 有效
    }

    prices = extract_benchmark_prices(daily_results, "AAA.SSE")

    assert prices == {"2024-01-04": 12.0}


def test_extract_benchmark_prices_none_inputs():
    """无 benchmark_symbol 或无 daily_results 返回空字典。"""
    assert extract_benchmark_prices(None, "AAA.SSE") == {}
    assert extract_benchmark_prices({}, None) == {}


def test_attach_benchmark_returns_normal():
    """逐行叠加策略/基准/超额累计收益（百分比），首个基准价为基准锚点。"""
    equity = [
        {"date": "2024-01-02", "balance": 1_000_000.0},
        {"date": "2024-01-03", "balance": 1_100_000.0},
    ]
    benchmark_prices = {"2024-01-02": 10.0, "2024-01-03": 10.5}

    out = attach_benchmark_returns(equity, benchmark_prices, capital=1_000_000.0)

    # 首日：策略 0%、基准 0%、超额 0%
    assert out[0]["strategy_return"] == 0.0
    assert out[0]["benchmark_return"] == 0.0
    assert out[0]["excess_return"] == 0.0
    # 次日：策略 +10%、基准 +5%、超额 +5%
    assert abs(out[1]["strategy_return"] - 10.0) < 1e-9
    assert abs(out[1]["benchmark_return"] - 5.0) < 1e-9
    assert abs(out[1]["excess_return"] - 5.0) < 1e-9


def test_attach_benchmark_returns_no_benchmark_prices():
    """无基准价时仅算策略收益，基准/超额置 None。"""
    equity = [{"date": "2024-01-02", "balance": 1_100_000.0}]

    out = attach_benchmark_returns(equity, {}, capital=1_000_000.0)

    assert abs(out[0]["strategy_return"] - 10.0) < 1e-9
    assert out[0]["benchmark_return"] is None
    assert out[0]["excess_return"] is None


def test_attach_benchmark_returns_missing_day_keeps_none():
    """中间交易日缺基准价时该行基准/超额留空，不影响其余行。"""
    equity = [
        {"date": "2024-01-02", "balance": 1_000_000.0},
        {"date": "2024-01-03", "balance": 1_050_000.0},
        {"date": "2024-01-04", "balance": 1_100_000.0},
    ]
    benchmark_prices = {"2024-01-02": 10.0, "2024-01-04": 11.0}

    out = attach_benchmark_returns(equity, benchmark_prices, capital=1_000_000.0)

    assert out[1]["benchmark_return"] is None
    assert out[1]["excess_return"] is None
    assert abs(out[2]["benchmark_return"] - 10.0) < 1e-9
    assert abs(out[2]["excess_return"] - 0.0) < 1e-9


def test_summarize_benchmark_takes_last_valid():
    """汇总取最后一个含有效基准收益的交易日。"""
    equity = [
        {"date": "2024-01-02", "benchmark_return": 0.0, "excess_return": 0.0},
        {"date": "2024-01-03", "benchmark_return": 5.0, "excess_return": 5.0},
        {"date": "2024-01-04", "benchmark_return": None, "excess_return": None},
    ]

    summary = summarize_benchmark(equity, "AAA.SSE")

    assert summary == {
        "benchmark_symbol": "AAA.SSE",
        "benchmark_return": 5.0,
        "excess_return": 5.0,
    }


def test_summarize_benchmark_empty_when_no_benchmark():
    """无任何有效基准收益时返回空字典。"""
    equity = [{"date": "2024-01-02", "benchmark_return": None, "excess_return": None}]

    assert summarize_benchmark(equity, "AAA.SSE") == {}
    assert summarize_benchmark([], "AAA.SSE") == {}
