"""技术分析算子（Technical Analysis Operators）——可选 talib 依赖。

封装 TA-Lib 指标为 DataProxy 接口，使其可在字符串表达式中调用。
若未安装 talib，导入时不报错，但调用时会抛出 ImportError。
"""

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

import polars as pl
import pandas as pd

from .utility import DataProxy


def _check_talib() -> None:
    """检查 talib 是否可用，不可用时抛出带安装提示的 ImportError。

    Raises:
        ImportError: talib 未安装时抛出，含安装命令提示。
    """
    if not TALIB_AVAILABLE:
        raise ImportError(
            "talib is not installed. "
            "Install it with: pip install talib  "
            "(may require: brew install ta-lib on macOS)"
        )


def to_pd_series(feature: DataProxy) -> pd.Series:
    """将 DataProxy 转换为以 (datetime, vt_symbol) 为多级索引的 pandas Series。

    talib 函数按标的逐组调用时需要 pandas 格式。

    Args:
        feature: 输入因子 DataProxy。

    Returns:
        以 [datetime, vt_symbol] 为 MultiIndex、"data" 列为值的 pandas Series。
    """
    series: pd.Series = feature.df.to_pandas().set_index(["datetime", "vt_symbol"])["data"]
    return series


def to_pl_dataframe(series: pd.Series) -> pl.DataFrame:
    """将 pandas Series 转回含 datetime、vt_symbol、data 列的 Polars DataFrame。

    Args:
        series: 以 [datetime, vt_symbol] 为 MultiIndex、列名为数值的 pandas Series。

    Returns:
        含 datetime、vt_symbol、data 三列的 Polars DataFrame。
    """
    return pl.from_pandas(series.reset_index().rename(columns={0: "data"}))


def ta_rsi(close: DataProxy, window: int) -> DataProxy:
    """计算 RSI（相对强弱指标）。

    按标的分组，调用 talib.RSI 计算。需安装 talib。

    Args:
        close: 收盘价 DataProxy。
        window: RSI 计算周期（timeperiod）。

    Returns:
        RSI 值 DataProxy，值域 [0, 100]；前 window 个值为 null。

    Raises:
        ImportError: talib 未安装时抛出。
    """
    _check_talib()
    close_: pd.Series = to_pd_series(close)
    result: pd.Series = talib.RSI(close_, timeperiod=window)   # type: ignore
    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_atr(high: DataProxy, low: DataProxy, close: DataProxy, window: int) -> DataProxy:
    """计算 ATR（真实波动幅度均值）。

    按标的分组，调用 talib.ATR 计算。需安装 talib。

    Args:
        high: 最高价 DataProxy。
        low: 最低价 DataProxy。
        close: 收盘价 DataProxy。
        window: ATR 平滑周期（timeperiod）。

    Returns:
        ATR 值 DataProxy（非负浮点数）；前 window 个值为 null。

    Raises:
        ImportError: talib 未安装时抛出。
    """
    _check_talib()
    high_: pd.Series = to_pd_series(high)
    low_: pd.Series = to_pd_series(low)
    close_: pd.Series = to_pd_series(close)
    result: pd.Series = talib.ATR(high_, low_, close_, timeperiod=window)   # type: ignore
    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)
