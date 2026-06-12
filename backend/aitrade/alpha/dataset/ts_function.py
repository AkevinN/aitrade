"""时序算子（Time Series Operators）。

在单标的时间序列上做滚动/滞后/差分等统计运算。
所有函数接收并返回 DataProxy，可在字符串表达式中直接调用。
窗口参数 window 均以 K 线根数为单位。
"""

from typing import cast

from scipy import stats
import polars as pl
import numpy as np

from .utility import DataProxy


def ts_delay(feature: DataProxy, window: int) -> DataProxy:
    """取 window 期前的历史值（滞后/超前）。

    window > 0 取历史值（向过去移动）；window < 0 取未来值（向未来移动，
    常用于标签计算，需注意避免用于特征以防数据泄漏）。

    Args:
        feature: 输入因子 DataProxy。
        window: 滞后期数，按标的分组 shift。

    Returns:
        滞后后的 DataProxy；窗口内不足的位置填 null。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").shift(window).over("vt_symbol")
    )
    return DataProxy(df)


def ts_min(feature: DataProxy, window: int) -> DataProxy:
    """计算滚动窗口内的最小值（min_samples=1，允许不足窗口计算）。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）。

    Returns:
        滚动最小值 DataProxy。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_min(window, min_samples=1).over("vt_symbol")
    )
    return DataProxy(df)


def ts_max(feature: DataProxy, window: int) -> DataProxy:
    """计算滚动窗口内的最大值（min_samples=1，允许不足窗口计算）。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）。

    Returns:
        滚动最大值 DataProxy。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_max(window, min_samples=1).over("vt_symbol")
    )
    return DataProxy(df)


def ts_argmax(feature: DataProxy, window: int) -> DataProxy:
    """返回滚动窗口内最大值所在的位置索引（从 1 开始，越大表示越靠近当前）。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）。

    Returns:
        最大值位置 DataProxy，值域为 [1, window]。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: cast(int, s.arg_max()) + 1, window).over("vt_symbol")
    )
    return DataProxy(df)


def ts_argmin(feature: DataProxy, window: int) -> DataProxy:
    """返回滚动窗口内最小值所在的位置索引（从 1 开始，越大表示越靠近当前）。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）。

    Returns:
        最小值位置 DataProxy，值域为 [1, window]。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: cast(int, s.arg_min()) + 1, window).over("vt_symbol")
    )
    return DataProxy(df)


def ts_rank(feature: DataProxy, window: int) -> DataProxy:
    """计算当前值在滚动窗口内的百分位排名（0~1）。

    使用 scipy.stats.percentileofscore 实现，含当前值在内；
    结果为 0 到 1 之间的浮点数（percentile / 100）。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）。

    Returns:
        百分位排名 DataProxy，值域 [0, 1]。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: stats.percentileofscore(s, s[-1]) / 100, window).over("vt_symbol")
    )
    return DataProxy(df)


def ts_sum(feature: DataProxy, window: int) -> DataProxy:
    """计算滚动窗口内的累加和。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）；窗口内不足 window 根时返回 null。

    Returns:
        滚动求和 DataProxy。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_sum(window).over("vt_symbol")
    )
    return DataProxy(df)


def ts_mean(feature: DataProxy, window: int) -> DataProxy:
    """计算滚动窗口内的均值（忽略 NaN，min_samples=1）。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）。

    Returns:
        滚动均值 DataProxy。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: np.nanmean(s), window, min_samples=1).over("vt_symbol")
    )
    return DataProxy(df)


def ts_std(feature: DataProxy, window: int) -> DataProxy:
    """计算滚动窗口内的总体标准差（ddof=0，忽略 NaN，min_samples=1）。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）。

    Returns:
        滚动标准差 DataProxy。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: np.nanstd(s, ddof=0), window, min_samples=1).over("vt_symbol")
    )
    return DataProxy(df)


def ts_slope(feature: DataProxy, window: int) -> DataProxy:
    """计算滚动窗口内线性回归的斜率（向量化优化实现）。

    以等间距序号 0, 1, …, window-1 为自变量，窗口内因子值为因变量，
    用 OLS 解析式直接计算斜率，避免 rolling_map + scipy 的逐窗口调用开销。
    窗口内数据不足 window 根时结果为 null。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）。

    Returns:
        滚动斜率 DataProxy。
    """
    n = window
    sum_x = n * (n - 1) / 2
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6
    denominator = n * sum_x2 - sum_x * sum_x

    sum_xy_expr: pl.Expr = pl.sum_horizontal([
        (window - 1 - j) * pl.col("data").shift(j)
        for j in range(window)
    ])

    df: pl.DataFrame = feature.df.with_columns([
        pl.col("data").rolling_sum(window, min_samples=window).over("vt_symbol").alias("sum_y"),
        sum_xy_expr.over("vt_symbol").alias("sum_xy")
    ])

    df = df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        ((n * pl.col("sum_xy") - sum_x * pl.col("sum_y")) / denominator).alias("data")
    )
    return DataProxy(df)


def ts_quantile(feature: DataProxy, window: int, quantile: float) -> DataProxy:
    """计算滚动窗口内指定分位数值（线性插值）。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）。
        quantile: 分位数，取值范围 [0.0, 1.0]，如 0.8 表示 80th percentile。

    Returns:
        滚动分位数 DataProxy。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: s.quantile(quantile=quantile, interpolation="linear"), window).over("vt_symbol")
    )
    return DataProxy(df)


def ts_rsquare(feature: DataProxy, window: int) -> DataProxy:
    """计算滚动窗口内线性回归的 R²（判定系数，向量化优化实现）。

    以等间距序号为自变量，窗口内因子值为因变量，
    用解析式 cov(x,y)² / (var_x * var_y) 计算 R²；
    当方差为零（常数序列）时输出 null。
    窗口内数据不足 window 根时结果为 null。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）。

    Returns:
        滚动 R² DataProxy，值域 [0, 1]。
    """
    n = window
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6
    mean_x = (n - 1) / 2
    var_x = sum_x2 / n - mean_x * mean_x

    sum_xy_expr: pl.Expr = pl.sum_horizontal([
        (window - 1 - j) * pl.col("data").shift(j)
        for j in range(window)
    ])

    df: pl.DataFrame = feature.df.with_columns([
        pl.col("data").rolling_sum(window, min_samples=window).over("vt_symbol").alias("sum_y"),
        pl.col("data").rolling_var(window, min_samples=window, ddof=0).over("vt_symbol").alias("var_y"),
        sum_xy_expr.over("vt_symbol").alias("sum_xy")
    ])

    df = df.with_columns([
        (pl.col("sum_y") / n).alias("mean_y"),
    ])

    df = df.with_columns([
        (pl.col("sum_xy") / n - mean_x * pl.col("mean_y")).alias("cov_xy")
    ])

    df = df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (pl.col("cov_xy").pow(2) / (var_x * pl.col("var_y"))).alias("data")
    )

    df = df.with_columns(
        pl.when(pl.col("data").is_infinite() | pl.col("data").is_nan())
        .then(None)
        .otherwise(pl.col("data"))
        .alias("data")
    )

    return DataProxy(df)


def ts_resi(feature: DataProxy, window: int) -> DataProxy:
    """计算当前值相对滚动线性回归预测值的残差（向量化优化实现）。

    以等间距序号为自变量对窗口内数据拟合 OLS，返回当前时点（序号 window-1）
    的实际值减预测值。常用于剥离趋势成分。
    窗口内数据不足 window 根时结果为 null。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）。

    Returns:
        滚动残差 DataProxy。
    """
    n = window
    sum_x = n * (n - 1) / 2
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6
    mean_x = (n - 1) / 2
    denominator = n * sum_x2 - sum_x * sum_x

    sum_xy_expr: pl.Expr = pl.sum_horizontal([
        (window - 1 - j) * pl.col("data").shift(j)
        for j in range(window)
    ])

    df: pl.DataFrame = feature.df.with_columns([
        pl.col("data").rolling_sum(window, min_samples=window).over("vt_symbol").alias("sum_y"),
        sum_xy_expr.over("vt_symbol").alias("sum_xy")
    ])

    df = df.with_columns([
        ((n * pl.col("sum_xy") - sum_x * pl.col("sum_y")) / denominator).alias("slope"),
        (pl.col("sum_y") / n).alias("mean_y"),
    ])

    df = df.with_columns([
        (pl.col("mean_y") - pl.col("slope") * mean_x).alias("intercept")
    ])

    df = df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (pl.col("data") - (pl.col("slope") * (n - 1) + pl.col("intercept"))).alias("data")
    )

    return DataProxy(df)


def ts_corr(feature1: DataProxy, feature2: DataProxy, window: int) -> DataProxy:
    """计算两个因子在滚动窗口内的 Pearson 相关系数。

    对两个 DataProxy 按 datetime/vt_symbol join 后调用 Polars rolling_corr；
    结果为 ±Inf 时替换为 null。

    Args:
        feature1: 第一个输入因子 DataProxy。
        feature2: 第二个输入因子 DataProxy，需与 feature1 索引对齐。
        window: 滚动窗口大小（根数）；min_samples=1 允许不足窗口计算。

    Returns:
        滚动相关系数 DataProxy，值域 [-1, 1]（Inf 已替换为 null）。
    """
    df_merged: pl.DataFrame = feature1.df.join(feature2.df, on=["datetime", "vt_symbol"])

    df: pl.DataFrame = df_merged.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.rolling_corr("data", "data_right", window_size=window, min_samples=1).over("vt_symbol").alias("data")
    )

    df = df.with_columns(
        pl.when(pl.col("data").is_infinite()).then(None).otherwise(pl.col("data")).alias("data")
    )

    return DataProxy(df)


def ts_less(feature1: DataProxy, feature2: DataProxy | float) -> DataProxy:
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


def ts_greater(feature1: DataProxy, feature2: DataProxy | float) -> DataProxy:
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


def ts_log(feature: DataProxy) -> DataProxy:
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


def ts_abs(feature: DataProxy) -> DataProxy:
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


def ts_delta(feature: DataProxy, window: int) -> DataProxy:
    """计算当前值与 window 期前历史值的差分。

    等价于 feature - ts_delay(feature, window)。

    Args:
        feature: 输入因子 DataProxy。
        window: 差分间隔期数（> 0）。

    Returns:
        差分结果 DataProxy。
    """
    return feature - ts_delay(feature, window)


def ts_cov(feature1: DataProxy, feature2: DataProxy, window: int) -> DataProxy:
    """计算两个因子在滚动窗口内的协方差。

    通过 corr * std1 * std2 计算，复用 ts_corr / ts_std。

    Args:
        feature1: 第一个输入因子 DataProxy。
        feature2: 第二个输入因子 DataProxy，需与 feature1 索引对齐。
        window: 滚动窗口大小（根数）。

    Returns:
        滚动协方差 DataProxy。
    """
    return ts_corr(feature1, feature2, window) * ts_std(feature1, window) * ts_std(feature2, window)


def ts_decay_linear(feature: DataProxy, window: int) -> DataProxy:
    """计算线性衰减加权平均（最近权重最大）。

    权重为 [window, window-1, ..., 1]，归一化分母为 window*(window+1)/2。
    常用于使因子更敏感于最近数据。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数），也是最大权重值。

    Returns:
        线性衰减加权均值 DataProxy；窗口内数据不足时返回 null。
    """
    def decay_func(s: pl.Series) -> float:
        """对一个滚动窗口 Series 计算线性衰减加权平均。

        Args:
            s: 长度为 window 的 Polars Series，最后一个元素为最新值。

        Returns:
            加权平均结果（浮点数）。
        """
        weights = pl.Series(range(window, 0, -1))
        return float((s * weights).sum() / (window * (window + 1) / 2))

    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: decay_func(s), window).over("vt_symbol")
    )
    return DataProxy(df)


def ts_product(feature: DataProxy, window: int) -> DataProxy:
    """计算滚动窗口内所有值的连乘积。

    Args:
        feature: 输入因子 DataProxy。
        window: 滚动窗口大小（根数）；窗口内数据不足时返回 null。

    Returns:
        滚动乘积 DataProxy。
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: s.product(), window).over("vt_symbol")
    )
    return DataProxy(df)
