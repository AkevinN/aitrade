"""Alpha 数据集预处理函数库。

提供 drop_na、fill_na、截面标准化（robust/zscore/rank）等常用特征预处理器，
以及可分步拟合/应用的 robust Z-score 统计工具。
所有函数均为纯函数（无副作用），返回新 DataFrame。
"""

from datetime import datetime

import numpy as np
import polars as pl

from .utility import to_datetime


def process_drop_na(df: pl.DataFrame, names: list[str] | None = None) -> pl.DataFrame:
    """删除含缺失值（NaN 或 null）的行。

    先将指定列的 NaN 填充为 null，再执行 drop_nulls；
    不指定 names 时默认作用于第 3 列到倒数第 2 列（即跳过 datetime/vt_symbol 与 label）。

    Args:
        df: 包含 datetime、vt_symbol、特征列及标签列的 Polars DataFrame。
        names: 需要检查的特征列名列表；为 None 时使用 df.columns[2:-1]。

    Returns:
        删除目标列存在缺失值的行后的 DataFrame。
    """
    if names is None:
        names = df.columns[2:-1]

    for name in names:
        df = df.with_columns(
            pl.col(name).fill_nan(None)
        )
    df = df.drop_nulls(subset=names)
    return df


def process_fill_na(df: pl.DataFrame, fill_value: float, fill_label: bool = True) -> pl.DataFrame:
    """将缺失值（NaN/null）填充为指定常数。

    Args:
        df: 包含特征列与标签列的 Polars DataFrame。
        fill_value: 用于填充的数值。
        fill_label: 为 True 时对全部列（含标签）填充；
            为 False 时只填充 df.columns[2:-1] 的特征列，保留标签列原值。

    Returns:
        填充后的 DataFrame，行数不变。
    """
    if fill_label:
        df = df.fill_null(fill_value)
        df = df.fill_nan(fill_value)
    else:
        df = df.with_columns(
            [pl.col(col).fill_null(fill_value).fill_nan(fill_value) for col in df.columns[2:-1]]
        )
    return df


def process_cs_norm(
    df: pl.DataFrame,
    names: list[str],
    method: str         # robust/zscore
) -> pl.DataFrame:
    """截面标准化：逐日对指定特征列做 robust MAD 或 Z-score 归一化。

    "robust" 方法：以截面中位数为中心、MAD（乘以 1.4826）为尺度，结果 clip(-3, 3)。
    "zscore" 方法：以截面均值为中心、截面标准差为尺度，不做裁剪。

    Args:
        df: 包含 datetime 与特征列的 Polars DataFrame。
        names: 需要标准化的列名列表。
        method: "robust" 或 "zscore"（大小写敏感）。

    Returns:
        原地替换目标列后的标准化 DataFrame，行数不变。
    """
    _df: pl.DataFrame = df.fill_nan(None)

    if method == "robust":
        for col in names:
            df = df.with_columns(
                _df.select(
                    (pl.col(col) - pl.col(col).median()).over("datetime").alias(col),
                )
            )

            df = df.with_columns(
                df.select(
                    pl.col(col).abs().median().over("datetime").alias("mad"),
                )
            )

            df = df.with_columns(
                (pl.col(col) / pl.col("mad") / 1.4826).clip(-3, 3).alias(col)
            ).drop(["mad"])
    else:
        for col in names:
            df = df.with_columns(
                _df.select(
                    pl.col(col).mean().over("datetime").alias("mean"),
                    pl.col(col).std().over("datetime").alias("std"),
                )
            )

            df = df.with_columns(
                (pl.col(col) - pl.col("mean")) / pl.col("std").alias(col)
            ).drop(["mean", "std"])

    return df


def process_robust_zscore_norm(
    df: pl.DataFrame,
    fit_start_time: datetime | str | None = None,
    fit_end_time: datetime | str | None = None,
    clip_outlier: bool = True
) -> pl.DataFrame:
    """全局 robust Z-score 标准化（时序维度）。

    在指定拟合区间（或全量数据）上计算各特征列的中位数与 MAD，
    然后对整张表做 (x - median) / (MAD * 1.4826) 转换；
    作用于所有特征列（df.columns[2:-1]，跳过 datetime/vt_symbol 与 label）。

    Args:
        df: 包含 datetime、vt_symbol、特征列、标签列的 Polars DataFrame。
        fit_start_time: 计算统计量的起始时间；为 None 时使用全量数据。
        fit_end_time: 计算统计量的截止时间；为 None 时使用全量数据。
        clip_outlier: 为 True 时将归一化结果裁剪到 [-3, 3]。

    Returns:
        替换特征列为归一化值后的 DataFrame，行数不变。
    """
    _df: pl.DataFrame = df.fill_nan(None)

    if fit_start_time and fit_end_time:
        fit_start_time = to_datetime(fit_start_time)
        fit_end_time = to_datetime(fit_end_time)
        _df = _df.filter((pl.col("datetime") >= fit_start_time) & (pl.col("datetime") <= fit_end_time))

    cols = df.columns[2:-1]
    X = _df.select(cols).to_numpy()

    mean_train = np.nanmedian(X, axis=0)
    std_train = np.nanmedian(np.abs(X - mean_train), axis=0)
    std_train += 1e-12
    std_train *= 1.4826

    for name in cols:
        normalized_col = (
            (pl.col(name) - mean_train[cols.index(name)]) / std_train[cols.index(name)]
        ).cast(pl.Float64)

        if clip_outlier:
            normalized_col = normalized_col.clip(-3, 3)

        df = df.with_columns(normalized_col.alias(name))

    return df


def fit_robust_zscore_stats(
    df: pl.DataFrame,
    names: list[str],
    fit_start_time: datetime | str | None = None,
    fit_end_time: datetime | str | None = None,
) -> dict[str, dict[str, float]]:
    """在指定区间上拟合各特征的 robust Z-score 统计量。

    用于需要将训练集统计量保存后在测试集上复用的场景（防止数据泄漏）。

    Args:
        df: 包含 datetime 及目标特征列的 Polars DataFrame。
        names: 需要拟合的特征列名列表。
        fit_start_time: 拟合区间起始时间；为 None 时不做下界过滤。
        fit_end_time: 拟合区间截止时间；为 None 时不做上界过滤。

    Returns:
        字典 {feature_name: {"median": float, "scale": float}}，
        scale = MAD * 1.4826 + 1e-12（防零除）。

    Raises:
        ValueError: 拟合区间内无可用样本时抛出。
    """
    fit_df: pl.DataFrame = df.fill_nan(None)

    if fit_start_time and fit_end_time:
        fit_start_time = to_datetime(fit_start_time)
        fit_end_time = to_datetime(fit_end_time)
        fit_df = fit_df.filter((pl.col("datetime") >= fit_start_time) & (pl.col("datetime") <= fit_end_time))

    if fit_df.is_empty():
        raise ValueError("训练区间无可用样本，无法拟合标准化参数")

    stats: dict[str, dict[str, float]] = {}
    for name in names:
        values = fit_df.select(name).to_numpy().reshape(-1)
        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - median)))
        stats[name] = {
            "median": median,
            "scale": mad * 1.4826 + 1e-12,
        }

    return stats


def apply_robust_zscore_stats(
    df: pl.DataFrame,
    stats: dict[str, dict[str, float]],
    names: list[str],
    clip_outlier: bool = True,
) -> pl.DataFrame:
    """将预先拟合的 robust Z-score 统计量应用到 DataFrame。

    与 fit_robust_zscore_stats 配合使用：先在训练集 fit，再在验证/测试集 apply，
    避免统计量在推断阶段受未来数据污染。stats 中不存在的列名会被静默跳过。

    Args:
        df: 待标准化的 Polars DataFrame。
        stats: fit_robust_zscore_stats 的返回值：
            {feature_name: {"median": float, "scale": float}}。
        names: 需要应用标准化的列名列表。
        clip_outlier: 为 True 时将结果裁剪到 [-3, 3]。

    Returns:
        替换目标列后的 DataFrame，行数不变。
    """
    for name in names:
        config = stats.get(name)
        if config is None:
            continue

        normalized = (
            (pl.col(name) - pl.lit(config["median"])) / pl.lit(config["scale"])
        ).cast(pl.Float64)
        if clip_outlier:
            normalized = normalized.clip(-3, 3)
        df = df.with_columns(normalized.alias(name))

    return df


def fill_feature_nan(
    df: pl.DataFrame,
    names: list[str],
    fill_value: float = 0.0,
) -> pl.DataFrame:
    """仅对指定特征列填充 NaN/null，标签列保持不变。

    与 process_fill_na 的区别：只操作显式指定的列，不依赖列位置推断。

    Args:
        df: 包含特征列的 Polars DataFrame。
        names: 需要填充的列名列表。
        fill_value: 填充值，默认为 0.0。

    Returns:
        目标列缺失值填充后的 DataFrame，行数不变。
    """
    return df.with_columns(
        [pl.col(name).fill_nan(fill_value).fill_null(fill_value).alias(name) for name in names]
    )


def process_cs_rank_norm(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """截面排名归一化，将因子值映射到 [-1.73, 1.73] 附近。

    对每个截面（同一 datetime）做 average 排名后除以截面股票数，
    再减去 0.5 并乘以 3.46，使得结果分布类似均匀分布的标准化形式。

    Args:
        df: 包含 datetime 与特征列的 Polars DataFrame。
        names: 需要归一化的列名列表。

    Returns:
        目标列替换为截面排名归一化值后的 DataFrame，行数不变。
    """
    _df: pl.DataFrame = df.fill_nan(None)

    _df = _df.with_columns([
        ((pl.col(col).rank("average").over("datetime") / pl.col("datetime").count().over("datetime")) - 0.5) * 3.46
        for col in names
    ])

    df = df.with_columns([
        _df[col].alias(col) for col in names
    ])

    return df
