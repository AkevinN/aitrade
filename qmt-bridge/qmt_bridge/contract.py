"""两端共用的契约：列名、代码/周期/复权口径映射。

这些常量与 aitrade 后端 BarRecord 字段、交易所/周期约定逐字对齐，
是 Mac Provider 与 Windows 服务之间的接口契约。
"""

from __future__ import annotations

# 与 aitrade BarRecord 字段顺序一致，Mac 端可直接 BarRecord(**row)
BARS_COLUMNS = [
    "symbol", "exchange", "datetime", "interval",
    "open_price", "high_price", "low_price", "close_price",
    "volume", "turnover", "open_interest", "adjust_type",
]

# xtdata K 线请求字段
XTDATA_BAR_FIELDS = ["time", "open", "high", "low", "close", "volume", "amount", "openInterest"]

_EXCHANGE_TO_QMT = {"SSE": "SH", "SZSE": "SZ", "BSE": "BJ"}
_QMT_TO_EXCHANGE = {v: k for k, v in _EXCHANGE_TO_QMT.items()}

_PERIOD_TO_QMT = {
    "d": "1d", "1m": "1m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "60m": "1h", "w": "1w",
}

_ADJUST_TO_DIVIDEND = {"none": "none", "qfq": "front", "hfq": "back"}
_ADJUST_TO_DIVIDEND_RATIO = {"none": "none", "qfq": "front_ratio", "hfq": "back_ratio"}


def to_qmt_code(symbol: str, exchange: str) -> str:
    """把 aitrade 的 (symbol, exchange) 拼成 xtdata 合约码。

    Args:
        symbol: 股票代码，如 "600000"。
        exchange: aitrade 交易所标识，支持 "SSE"/"SZSE"/"BSE"。

    Returns:
        xtdata 格式合约码，如 "600000.SH"。

    Raises:
        ValueError: exchange 不在支持列表时抛出。

    Example:
        >>> to_qmt_code("600000", "SSE")
        '600000.SH'
    """
    suffix = _EXCHANGE_TO_QMT.get(exchange)
    if suffix is None:
        raise ValueError(f"不支持的交易所: {exchange}")
    return f"{symbol}.{suffix}"


def from_qmt_code(code: str) -> tuple[str, str]:
    """把 xtdata 合约码拆回 aitrade (symbol, exchange) 形式。

    Args:
        code: xtdata 格式合约码，如 "600000.SH"。

    Returns:
        (symbol, exchange) 二元组，如 ("600000", "SSE")。

    Raises:
        ValueError: 后缀不在支持列表时抛出。

    Example:
        >>> from_qmt_code("600000.SH")
        ('600000', 'SSE')
    """
    sym, _, suffix = code.partition(".")
    exchange = _QMT_TO_EXCHANGE.get(suffix)
    if exchange is None:
        raise ValueError(f"不支持的 QMT 后缀: {code}")
    return sym, exchange


def to_qmt_period(interval: str) -> str:
    """aitrade 内部周期 -> xtdata period 字符串。

    支持 "d"/"1m"/"5m"/"15m"/"30m"/"1h"/"60m"/"w"；
    "60m" 与 "1h" 均映射为 "1h"。

    Args:
        interval: aitrade 周期字符串。

    Returns:
        xtdata period 字符串，如 "1d"、"1h"。

    Raises:
        ValueError: 未知周期时抛出。

    Example:
        >>> to_qmt_period("d")
        '1d'
        >>> to_qmt_period("60m")
        '1h'
    """
    period = _PERIOD_TO_QMT.get(interval)
    if period is None:
        raise ValueError(f"不支持的周期: {interval}")
    return period


def exchange_to_market(exchange: str) -> str:
    """aitrade 交易所 -> xtdata 日历/市场码，如 'SSE'->'SH'。

    Args:
        exchange: aitrade 交易所标识，支持 "SSE"/"SZSE"/"BSE"。

    Returns:
        xtdata 市场码，如 "SH"/"SZ"/"BJ"。

    Raises:
        ValueError: exchange 不在支持列表时抛出。

    Example:
        >>> exchange_to_market("SSE")
        'SH'
    """
    m = _EXCHANGE_TO_QMT.get(exchange)
    if m is None:
        raise ValueError(f"不支持的交易所: {exchange}")
    return m


def to_dividend_type(adjust_type: str, *, ratio: bool = False) -> str:
    """adjust_type(none/qfq/hfq) -> xtdata dividend_type 字符串。

    Args:
        adjust_type: 复权口径，支持 "none"/"qfq"/"hfq"。
        ratio: 为 True 时走等比复权口径（front_ratio/back_ratio），默认 False。

    Returns:
        xtdata dividend_type 字符串，如 "front"、"back_ratio"。

    Raises:
        ValueError: adjust_type 不在支持列表时抛出。

    Example:
        >>> to_dividend_type("qfq")
        'front'
        >>> to_dividend_type("hfq", ratio=True)
        'back_ratio'
    """
    table = _ADJUST_TO_DIVIDEND_RATIO if ratio else _ADJUST_TO_DIVIDEND
    if adjust_type not in table:
        raise ValueError(f"不支持的复权口径: {adjust_type}")
    return table[adjust_type]
