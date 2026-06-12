"""Alpha 数据集工具——DataProxy 运算代理、表达式计算函数、时间转换与区间枚举。

DataProxy 封装单因子 DataFrame，重载算术/比较运算符使得字符串表达式可以
像普通 Python 表达式一样被 eval，同时保留 datetime/vt_symbol 索引信息。
"""

from datetime import datetime
from enum import Enum
from typing import Union

import polars as pl


class DataProxy:
    """单因子数据代理，将 Polars DataFrame 包装成可参与算术/比较运算的对象。

    内部始终持有含 datetime、vt_symbol、data 三列的 DataFrame（data 为因子值）。
    通过运算符重载，使字符串表达式（如 "close / ts_delay(close, 1) - 1"）能在
    calculate_by_expression 的 eval 上下文中被透明计算。

    Example:
        >>> proxy_a = DataProxy(df_with_close)
        >>> proxy_b = DataProxy(df_with_open)
        >>> result = proxy_a / proxy_b  # 逐行对齐后相除，返回新 DataProxy
    """

    def __init__(self, df: pl.DataFrame) -> None:
        """初始化代理，将最后一列重命名为 "data"。

        Args:
            df: 至少包含 datetime、vt_symbol 以及一列数值列的 Polars DataFrame；
                最后一列将被用作因子值列（重命名为 "data"）。
        """
        self.name: str = df.columns[-1]
        self.df: pl.DataFrame = df.rename({self.name: "data"})

    def result(self, s: pl.Series) -> "DataProxy":
        """将计算结果 Series 包装回 DataProxy。

        保留当前代理的 datetime/vt_symbol 索引，用给定 Series 替换 data 列。

        Args:
            s: 与 self.df 行数一致的 Polars Series，表示新的因子值。

        Returns:
            包含相同索引列与新 data 列的 DataProxy 实例。
        """
        result: pl.DataFrame = self.df[["datetime", "vt_symbol"]]
        result = result.with_columns(other=s)

        return DataProxy(result)

    def __add__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """逐元素加法；other 可为另一个 DataProxy 或标量。"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] + other.df["data"]
        else:
            s = self.df["data"] + other
        return self.result(s)

    def __sub__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """逐元素减法；other 可为另一个 DataProxy 或标量。"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] - other.df["data"]
        else:
            s = self.df["data"] - other
        return self.result(s)

    def __mul__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """逐元素乘法；other 可为另一个 DataProxy 或标量。"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] * other.df["data"]
        else:
            s = self.df["data"] * other
        return self.result(s)

    def __rmul__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """右乘；支持 scalar * DataProxy 写法。"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] * other.df["data"]
        else:
            s = self.df["data"] * other
        return self.result(s)

    def __truediv__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """逐元素除法；other 可为另一个 DataProxy 或标量，不做零除保护。"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] / other.df["data"]
        else:
            s = self.df["data"] / other
        return self.result(s)

    def __abs__(self) -> "DataProxy":
        """返回因子值的绝对值。"""
        s: pl.Series = self.df["data"].abs()
        return self.result(s)

    def __gt__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """逐元素大于比较，结果转为 Int32（1=True, 0=False）。"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] > other.df["data"]
        else:
            s = self.df["data"] > other
        return self.result(s.cast(pl.Int32))

    def __ge__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """逐元素大于等于比较，结果转为 Int32（1=True, 0=False）。"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] >= other.df["data"]
        else:
            s = self.df["data"] >= other
        return self.result(s.cast(pl.Int32))

    def __lt__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """逐元素小于比较，结果转为 Int32（1=True, 0=False）。"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] < other.df["data"]
        else:
            s = self.df["data"] < other
        return self.result(s.cast(pl.Int32))

    def __le__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """逐元素小于等于比较，结果转为 Int32（1=True, 0=False）。"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] <= other.df["data"]
        else:
            s = self.df["data"] <= other
        return self.result(s.cast(pl.Int32))

    def __eq__(self, other: Union["DataProxy", int, float]) -> "DataProxy":    # type: ignore
        """逐元素等于比较，结果转为 Int32（1=True, 0=False）。"""
        if isinstance(other, DataProxy):
            s = self.df["data"] == other.df["data"]
        else:
            s = self.df["data"] == other
        return self.result(s.cast(pl.Int32))


def calculate_by_expression(df: pl.DataFrame, expression: str) -> pl.DataFrame:
    """在 DataProxy 上下文中 eval 字符串表达式，返回含 data 列的 DataFrame。

    将 df 中除 datetime/vt_symbol 外的每一列包装为 DataProxy 注入局部命名空间，
    同时导入所有时序/截面/数学算子函数，然后 eval(expression) 得到结果 DataProxy。

    Args:
        df: 原始行情 DataFrame，列必须包含 datetime、vt_symbol 及所需的因子列
            （如 open、high、low、close、volume、vwap 等）。
        expression: 合法的 Python 字符串表达式，如 "cs_rank(ts_delta(close, 1))"。

    Returns:
        含 datetime、vt_symbol、data 三列的 Polars DataFrame；data 为计算结果。

    Raises:
        NameError: expression 引用了不存在的列名或函数时抛出。

    Example:
        >>> result_df = calculate_by_expression(bar_df, "close / ts_delay(close, 1) - 1")
    """
    from .ts_function import (              # noqa
        ts_delay,
        ts_min, ts_max,
        ts_argmax, ts_argmin,
        ts_rank, ts_sum,
        ts_mean, ts_std,
        ts_slope, ts_quantile,
        ts_rsquare, ts_resi,
        ts_corr,
        ts_less, ts_greater,
        ts_log, ts_abs,
        ts_delta, ts_cov,
        ts_decay_linear,
        ts_product
    )
    from .cs_function import (              # noqa
        cs_rank,
        cs_mean,
        cs_std,
        cs_sum,
        cs_scale
    )
    from .ta_function import (              # noqa
        ta_rsi,
        ta_atr
    )
    from .math_function import (              # noqa
        less, greater, log, abs,
        sign, pow1, pow2,
        quesval, quesval2
    )

    d: dict = locals()

    for column in df.columns:
        if column in {"datetime", "vt_symbol"}:
            continue

        column_df = df[["datetime", "vt_symbol", column]]
        d[column] = DataProxy(column_df)

    other: DataProxy = eval(expression, {}, d)

    return other.df


def calculate_by_polars(df: pl.DataFrame, expression: pl.expr.expr.Expr) -> pl.DataFrame:
    """用 Polars 原生 Expr 计算因子，返回含 data 列的 DataFrame。

    适合直接用 Polars DSL 编写的表达式，避免 eval 开销。

    Args:
        df: 原始行情 DataFrame，需包含表达式所引用的列。
        expression: Polars Expr 对象，alias 会在内部被替换为 "data"。

    Returns:
        含 datetime、vt_symbol、data 三列的 Polars DataFrame。
    """
    return df.select([
        "datetime",
        "vt_symbol",
        expression.alias("data")
    ])


def to_datetime(arg: datetime | str) -> datetime:
    """将字符串或 datetime 统一转换为 datetime 对象。

    支持 "YYYY-MM-DD"（含连字符）与 "YYYYMMDD"（纯数字）两种字符串格式；
    若已是 datetime 则直接返回。

    Args:
        arg: 日期字符串（"2023-01-01" 或 "20230101"）或 datetime 对象。

    Returns:
        对应的 datetime 对象，时间部分为 00:00:00。

    Raises:
        ValueError: 字符串格式不符合上述两种格式时由 strptime 抛出。

    Example:
        >>> to_datetime("2023-06-01")
        datetime.datetime(2023, 6, 1, 0, 0)
        >>> to_datetime("20230601")
        datetime.datetime(2023, 6, 1, 0, 0)
    """
    if isinstance(arg, str):
        if "-" in arg:
            fmt: str = "%Y-%m-%d"
        else:
            fmt = "%Y%m%d"

        return datetime.strptime(arg, fmt)
    else:
        return arg


class Segment(Enum):
    """数据集区间枚举，用于区分训练/验证/测试三段。"""

    TRAIN = 1
    VALID = 2
    TEST = 3
