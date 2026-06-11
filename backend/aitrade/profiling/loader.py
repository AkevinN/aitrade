"""
标的画像（Symbol Profiling）只读数据加载与时间裁剪。

本文件是画像模块唯一的数据入口。除窗口加载器（后续实现）外，这里集中实现
时间裁剪相关的**纯函数**：它们仅对内存中的 polars.DataFrame 做只读变换，
不触碰文件系统、不调用 AlphaLab、不产生任何副作用。

时间窗口隔离是画像模块的第一公民：所有指标计算都必须以 as_of 为右边界，
物理上不读取 datetime > as_of 的行（Requirement 2.2 / 2.3）。把裁剪逻辑抽取为
纯函数，便于以属性测试覆盖"绝不泄露未来数据"这一关键性质。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from aitrade.alpha.lab_utils import normalize_vt_symbol

# 行情 frame 中的时间列名，与 AlphaLab（aitrade/alpha/lab.py）保持一致
_DATETIME_COLUMN = "datetime"


def clip_to_as_of(df: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """裁剪到截止时间：仅保留 datetime <= as_of 的行，并保持时间升序。

    这是时间窗口隔离的核心：物理过滤掉所有 datetime > as_of 的行，确保任何
    下游指标都不可能触及未来数据（Requirement 2.2 / 2.3）。

    纯函数：不修改入参 df，不做任何 I/O。

    边界处理：
    - 空 frame：直接返回（过滤后仍为空）。
    - 缺少 datetime 列：视为无法裁剪，原样返回（由上层保证 frame 合法）。

    :param df: 输入行情 frame（含 datetime 列）
    :param as_of: 截止时间（含义为"站在该时刻回看"），保留 <= as_of 的行
    :return: 仅含 datetime <= as_of 的行、按 datetime 升序排列的新 frame
    """
    # 缺少时间列时无法裁剪，原样返回（防御性处理，正常路径不会触发）
    if _DATETIME_COLUMN not in df.columns:
        return df

    # 物理过滤：只保留 datetime <= as_of 的行，并按时间升序排序
    return df.filter(pl.col(_DATETIME_COLUMN) <= as_of).sort(_DATETIME_COLUMN)


def effective_right_bound(df: pl.DataFrame, as_of: datetime) -> datetime | None:
    """计算有效右边界：先裁剪到 as_of，再返回裁剪后实际最大 datetime。

    当 as_of 晚于本地最新数据时，有效右边界会收窄为本地实际最大时间；
    此返回值用于在 Profile_Artifact 中记录实际参与计算的数据右边界，
    保证画像可复现、可审计未使用未来数据（Requirement 2.4 / 2.5）。

    纯函数：不修改入参 df，不做任何 I/O。

    边界处理：
    - 裁剪后为空（包括空 frame、或 as_of 早于全部数据）：返回 None。
    - 缺少 datetime 列：返回 None。

    :param df: 输入行情 frame（含 datetime 列）
    :param as_of: 截止时间，仅纳入 <= as_of 的行
    :return: 裁剪后最大 datetime（必然 <= as_of）；裁剪后为空时返回 None
    """
    clipped = clip_to_as_of(df, as_of)

    # 裁剪后为空：无有效右边界
    if clipped.is_empty():
        return None

    # 裁剪后的最大时间即有效右边界；clip_to_as_of 已保证其 <= as_of
    return clipped[_DATETIME_COLUMN].max()


def load_local_range(lab, vt_symbol: str, interval: str) -> tuple[datetime | None, datetime | None]:
    """只读获取本地完整数据区间，用于 unavailable_reason 诊断。"""
    normalized = normalize_vt_symbol(vt_symbol)
    df = lab.load_bar_frame_any_range(normalized, interval, include_derived=True)
    if df is None or df.is_empty() or _DATETIME_COLUMN not in df.columns:
        return None, None
    return df[_DATETIME_COLUMN].min(), df[_DATETIME_COLUMN].max()


def _load_window_frame(
    lab,
    vt_symbol: str,
    interval: str,
    as_of: datetime,
    lookback_days: int,
) -> pl.DataFrame | None:
    """只读加载 as_of 左侧窗口行情并再次物理裁剪。

    该函数只调用 AlphaLab 的只读读取接口，不调用任何聚合或写入方法。
    """
    normalized = normalize_vt_symbol(vt_symbol)
    if as_of.tzinfo is not None:
        as_of = as_of.replace(tzinfo=None)
    start = as_of - timedelta(days=max(1, int(lookback_days)))

    df = lab.load_bar_frame(
        normalized,
        interval,
        start,
        as_of,
        include_derived=True,
    )
    if df is None or df.is_empty():
        return None

    clipped = clip_to_as_of(df, as_of)
    if clipped.is_empty():
        return None
    return clipped
