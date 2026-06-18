"""把归一化行情序列化为 Arrow IPC stream（zstd）供远端零转码落盘。"""

from __future__ import annotations

import polars as pl

from .contract import BARS_COLUMNS

# 空数据时也要保证 schema 稳定
_EMPTY_SCHEMA = {
    "symbol": pl.Utf8, "exchange": pl.Utf8, "datetime": pl.Datetime, "interval": pl.Utf8,
    "open_price": pl.Float64, "high_price": pl.Float64, "low_price": pl.Float64,
    "close_price": pl.Float64, "volume": pl.Float64, "turnover": pl.Float64,
    "open_interest": pl.Float64, "adjust_type": pl.Utf8,
}


def bars_to_ipc(rows: list[dict]) -> bytes:
    """将归一化 bar 行列表序列化为 Arrow IPC stream 字节（zstd 压缩）。

    输出可直接由 Mac 客户端用 ``pl.read_ipc_stream`` 解码并零转码写入 Parquet。
    空列表仍产出稳定 schema（``_EMPTY_SCHEMA``），保证接收端可无条件解析。

    Args:
        rows: 每个 dict 必须含 ``BARS_COLUMNS`` 所定义的 12 个字段。
            空列表合法，返回 0 行但 schema 正确的 IPC stream。

    Returns:
        Arrow IPC stream 格式的 ``bytes``，使用 zstd 压缩。

    Example:
        >>> from datetime import datetime
        >>> blob = bars_to_ipc([{
        ...     "symbol": "600000", "exchange": "SSE",
        ...     "datetime": datetime(2024, 1, 2), "interval": "d",
        ...     "open_price": 10.0, "high_price": 11.0, "low_price": 9.8,
        ...     "close_price": 10.5, "volume": 1000.0, "turnover": 10500.0,
        ...     "open_interest": 0.0, "adjust_type": "hfq",
        ... }])
        >>> import polars as pl
        >>> pl.read_ipc_stream(blob).height
        1
    """
    if rows:
        df = pl.DataFrame(rows).select(BARS_COLUMNS)
    else:
        df = pl.DataFrame(schema=_EMPTY_SCHEMA)
    buf = df.write_ipc_stream(None, compression="zstd")
    return buf.getvalue()
