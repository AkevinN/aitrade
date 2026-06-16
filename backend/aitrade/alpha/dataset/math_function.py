"""数学运算函数（Math Functions）。

提供 DataProxy 级别的数学运算，可在字符串表达式中直接调用：
less/greater（逐元素 min/max）、log、abs、sign、pow1/pow2（安全幂次）、
quesval/quesval2（条件三元运算）。
"""

import polars as pl

from .utility import DataProxy


def less(feature1: DataProxy, feature2: DataProxy | float) -> DataProxy:
    """逐元素取两个因子中的较小值（element-wise min）。

    Args:
        feature1: 第一个输入因子 DataProxy。
        feature2: 第二个输入因子 DataProxy 或标量。

    Returns:
        逐元素最小值 DataProxy。
    """
    if isinstance(feature2, DataProxy):
        df_merged: pl.DataFrame = feature1.df.join(feature2.df, on=["datetime", "vt_symbol"])
    else:
        df_merged = feature1.df.with_columns(pl.lit(feature2).alias("data_right"))

    df: pl.DataFrame = df_merged.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.min_horizontal("data", "data_right").over("vt_symbol").alias("data")
    )

    return DataProxy(df)


def greater(feature1: DataProxy, feature2: DataProxy | float) -> DataProxy:
    """逐元素取两个因子中的较大值（element-wise max）。

    Args:
        feature1: 第一个输入因子 DataProxy。
        feature2: 第二个输入因子 DataProxy 或标量。

    Returns:
        逐元素最大值 DataProxy。
    """
    if isinstance(feature2, DataProxy):
        df_merged: pl.DataFrame = feature1.df.join(feature2.df, on=["datetime", "vt_symbol"])

    else:
        df_merged = feature1.df.with_columns(pl.lit(feature2).alias("data_right"))

    df: pl.DataFrame = df_merged.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.max_horizontal("data", "data_right").over("vt_symbol").alias("data")
    )

    return DataProxy(df)


def log(feature: DataProxy) -> DataProxy:
    """计算因子值的自然对数。

    Args:
        feature: 输入因子 DataProxy，值应大于 0。

    Returns:
        自然对数 DataProxy；值 <= 0 时 Polars 返回 null 或 NaN。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").log().over("vt_symbol")
    )
    return DataProxy(df)


def abs(feature: DataProxy) -> DataProxy:
    """计算因子值的绝对值。

    Args:
        feature: 输入因子 DataProxy。

    Returns:
        绝对值 DataProxy。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").abs().over("vt_symbol")
    )
    return DataProxy(df)


def sign(feature: DataProxy) -> DataProxy:
    """返回因子值的符号：正数→1，负数→-1，零→0。

    Args:
        feature: 输入因子 DataProxy。

    Returns:
        符号值 DataProxy，取值 {-1, 0, 1}（Int32 类型）。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.when(pl.col("data") > 0).then(1).when(pl.col("data") < 0).then(-1).otherwise(0).alias("data")
    )
    return DataProxy(df)


def quesval(threshold: float, feature1: DataProxy, feature2: DataProxy | float | int, feature3: DataProxy | float | int) -> DataProxy:
    """标量阈值三元条件运算：threshold < feature1 时取 feature2，否则取 feature3。

    等价于 Python 的 feature2 if threshold < feature1 else feature3，逐行求值。
    常用于因子表达式中的分段逻辑，如 WorldQuant Alpha 101 的 Alpha9。

    Args:
        threshold: 浮点数阈值，作为条件的左操作数。
        feature1: 条件的右操作数 DataProxy。
        feature2: 条件为真时的取值，DataProxy 或标量。
        feature3: 条件为假时的取值，DataProxy 或标量。

    Returns:
        按条件逐行选择后的 DataProxy。

    Example:
        >>> quesval(0, returns, close, ts_std(returns, 20))
        # returns > 0 时取 close，否则取 ts_std(returns, 20)
    """
    df_merged = feature1.df

    if isinstance(feature2, DataProxy):
        df_merged = df_merged.join(feature2.df, on=["datetime", "vt_symbol"], suffix="_true")
    else:
        df_merged = df_merged.with_columns(pl.lit(feature2).alias("data_true"))

    if isinstance(feature3, DataProxy):
        df_merged = df_merged.join(feature3.df, on=["datetime", "vt_symbol"], suffix="_false")
    else:
        df_merged = df_merged.with_columns(pl.lit(feature3).alias("data_false"))

    df: pl.DataFrame = df_merged.with_columns(
        pl.when(threshold < pl.col("data"))
        .then(pl.col("data_true"))
        .otherwise(pl.col("data_false"))
        .alias("data")
    ).select(["datetime", "vt_symbol", "data"])

    return DataProxy(df)


def quesval2(threshold: DataProxy, feature1: DataProxy, feature2: DataProxy | float | int, feature3: DataProxy | float | int) -> DataProxy:
    """DataProxy 阈值三元条件运算：threshold < feature1 时取 feature2，否则取 feature3。

    与 quesval 的区别在于 threshold 也是 DataProxy（逐行动态阈值）。
    常用于 Alpha7 等需要动态比较两个因子大小的场景。

    Args:
        threshold: 条件左操作数 DataProxy（动态阈值）。
        feature1: 条件右操作数 DataProxy。
        feature2: 条件为真时的取值，DataProxy 或标量。
        feature3: 条件为假时的取值，DataProxy 或标量。

    Returns:
        按逐行条件选择后的 DataProxy。
    """
    df_merged: pl.DataFrame = threshold.df.join(feature1.df, on=["datetime", "vt_symbol"], suffix="_cond")

    if isinstance(feature2, DataProxy):
        df_merged = df_merged.join(feature2.df, on=["datetime", "vt_symbol"], suffix="_true")
    else:
        df_merged = df_merged.with_columns(pl.lit(feature2).alias("data_true"))

    if isinstance(feature3, DataProxy):
        df_merged = df_merged.join(feature3.df, on=["datetime", "vt_symbol"], suffix="_false")
    else:
        df_merged = df_merged.with_columns(pl.lit(feature3).alias("data_false"))

    df: pl.DataFrame = df_merged.with_columns(
        pl.when(pl.col("data_cond") < pl.col("data"))
        .then(pl.col("data_true"))
        .otherwise(pl.col("data_false"))
        .alias("data")
    ).select(["datetime", "vt_symbol", "data"])

    return DataProxy(df)


def pow1(base: DataProxy, exponent: float) -> DataProxy:
    """安全幂次运算（标量指数，处理负底数）。

    base > 0 时计算 base^exponent；
    base < 0 时计算 -1 * |base|^exponent（保留符号）；
    base == 0 时返回 0。

    Args:
        base: 底数 DataProxy。
        exponent: 指数（浮点数标量）。

    Returns:
        幂次结果 DataProxy。
    """
    df: pl.DataFrame = base.df.with_columns(
        pl.when(pl.col("data") > 0)
        .then(pl.col("data").pow(exponent))
        .when(pl.col("data") < 0)
        .then(pl.lit(-1) * pl.col("data").abs().pow(exponent))
        .otherwise(0)
        .alias("data")
    )

    return DataProxy(df)


def pow2(base: DataProxy, exponent: DataProxy) -> DataProxy:
    """安全幂次运算（DataProxy 指数，处理负底数与非整数指数边界）。

    处理逻辑：
    - base > 0：计算 base^exponent。
    - base < 0 且 exponent 为整数：计算 -1 * |base|^exponent（保留符号）。
    - 其他情况（base==0、exponent 为 NaN、负底数与非整数指数）：返回 0。

    Args:
        base: 底数 DataProxy。
        exponent: 指数 DataProxy，需与 base 按 datetime/vt_symbol 对齐。

    Returns:
        幂次结果 DataProxy；边界情况见上述处理逻辑。
    """
    base_renamed = base.df.rename({"data": "base_data"})
    exp_renamed = exponent.df.rename({"data": "exp_data"})

    df_merged: pl.DataFrame = base_renamed.join(exp_renamed, on=["datetime", "vt_symbol"], how="left")

    df: pl.DataFrame = df_merged.with_columns(
        pl.when(pl.col("base_data") > 0)
        .then(pl.col("base_data").pow(pl.col("exp_data")))
        .when(
            (pl.col("base_data") < 0) &
            (~pl.col("exp_data").is_nan()) &
            (pl.col("exp_data").floor() == pl.col("exp_data"))
        )
        .then((-1) * pl.col("base_data").abs().pow(pl.col("exp_data")))
        .otherwise(pl.lit(None))
        .fill_nan(None)
        .fill_null(0)
        .alias("data")
    ).select(["datetime", "vt_symbol", "data"])

    return DataProxy(df)
