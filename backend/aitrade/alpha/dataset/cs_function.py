"""截面算子（Cross-Section Operators）。

在同一时间截面内对多标的因子值做统计运算（排名、均值、标准差、求和、缩放）。
所有函数接收并返回 DataProxy，可在字符串表达式中直接调用。
"""

import polars as pl

from .utility import DataProxy


def cs_rank(feature: DataProxy) -> DataProxy:
    """对每个截面（同一 datetime）内的因子值做升序排名。

    Args:
        feature: 输入因子 DataProxy。

    Returns:
        截面排名后的 DataProxy，排名值从 1 开始。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rank().over("datetime")
    )
    return DataProxy(df)


def cs_mean(feature: DataProxy) -> DataProxy:
    """计算每个截面的均值，结果广播至截面内所有标的。

    Args:
        feature: 输入因子 DataProxy。

    Returns:
        截面均值 DataProxy；每个时间点所有标的值相同。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").mean().over("datetime")
    )
    return DataProxy(df)


def cs_std(feature: DataProxy) -> DataProxy:
    """计算每个截面的标准差，结果广播至截面内所有标的。

    Args:
        feature: 输入因子 DataProxy。

    Returns:
        截面标准差 DataProxy（Bessel 校正，ddof=1）。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").std().over("datetime")
    )
    return DataProxy(df)


def cs_sum(feature: DataProxy) -> DataProxy:
    """计算每个截面的求和，结果广播至截面内所有标的。

    Args:
        feature: 输入因子 DataProxy。

    Returns:
        截面合计值 DataProxy。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").sum().over("datetime")
    )
    return DataProxy(df)


def cs_scale(feature: DataProxy) -> DataProxy:
    """按截面绝对值之和对因子值进行缩放（L1 归一化）。

    公式：scaled = value / sum(|value|)。若截面绝对值之和为零，则输出为 0。
    常用于将多头/空头因子缩放为市值中性的权重。

    Args:
        feature: 输入因子 DataProxy。

    Returns:
        截面 L1 归一化后的 DataProxy。
    """
    abs_feature = abs(feature)
    sum_abs = cs_sum(abs_feature)

    df_merged: pl.DataFrame = feature.df.join(sum_abs.df, on=["datetime", "vt_symbol"], suffix="_sum")

    df: pl.DataFrame = df_merged.with_columns(
        pl.when(pl.col("data_sum") != 0)
        .then(pl.col("data") / pl.col("data_sum"))
        .otherwise(0)
        .alias("data")
    ).select(["datetime", "vt_symbol", "data"])

    return DataProxy(df)
