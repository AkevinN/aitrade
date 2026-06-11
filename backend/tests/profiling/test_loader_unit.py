"""
标的画像（Symbol Profiling）时间裁剪纯函数的单元测试（普通 pytest，非 Hypothesis）。

本文件针对 `aitrade.profiling.loader` 的 `clip_to_as_of` 与 `effective_right_bound`
写具体示例，覆盖 as_of 早于 / 晚于 / 居中于数据以及空 frame 的边界情形，
与同目录下基于 Hypothesis 的属性测试互补。

构造 frame 时显式指定 Datetime schema，保证空 frame 的 datetime 列仍为正确类型，
避免下游 filter / 比较因 Null 类型列而行为异常。

Requirements: 2.2, 2.4
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from aitrade.profiling.loader import clip_to_as_of, effective_right_bound

# 显式 schema：保证空 frame 的 datetime 列仍为 Datetime 类型
_DT_SCHEMA = {"datetime": pl.Datetime("us")}

# 一段固定的、升序的样本时间戳（分钟级），用于构造测试 frame
_T0 = datetime(2024, 1, 1, 9, 30)
_T1 = datetime(2024, 1, 1, 9, 31)
_T2 = datetime(2024, 1, 1, 9, 32)
_T3 = datetime(2024, 1, 1, 9, 33)
_T4 = datetime(2024, 1, 1, 9, 34)


def _frame(dts: list[datetime]) -> pl.DataFrame:
    """由时间戳列表构造仅含 datetime 列的 frame（空列表得到带正确类型的空 frame）。"""
    return pl.DataFrame({"datetime": dts}, schema=_DT_SCHEMA)


def _empty_frame() -> pl.DataFrame:
    """构造显式指定 Datetime schema 的空 frame。"""
    return _frame([])


# --- clip_to_as_of ---------------------------------------------------------


def test_clip_as_of_before_all_returns_empty() -> None:
    """as_of 早于全部数据：裁剪后结果为空，且 datetime 列类型保持正常。"""
    df = _frame([_T1, _T2, _T3])
    # as_of 落在最早数据之前
    as_of = datetime(2024, 1, 1, 9, 0)

    result = clip_to_as_of(df, as_of)

    assert result.height == 0
    # 列类型仍为 Datetime（裁剪不应破坏 schema）
    assert result.schema["datetime"] == pl.Datetime("us")


def test_clip_as_of_after_all_returns_all_sorted() -> None:
    """as_of 晚于全部数据：结果等于全部数据，并按 datetime 升序。"""
    # 故意以乱序传入，验证裁剪结果会被排序为升序
    df = _frame([_T2, _T0, _T1])
    as_of = datetime(2024, 1, 1, 10, 0)

    result = clip_to_as_of(df, as_of)

    assert result["datetime"].to_list() == [_T0, _T1, _T2]


def test_clip_as_of_in_middle_keeps_le_rows() -> None:
    """as_of 落在中间：只保留 datetime <= as_of 的行，并按升序排列。"""
    df = _frame([_T0, _T1, _T2, _T3, _T4])
    # as_of 恰好等于 _T2，边界行（== as_of）应被保留
    as_of = _T2

    result = clip_to_as_of(df, as_of)

    assert result["datetime"].to_list() == [_T0, _T1, _T2]


def test_clip_empty_frame_returns_empty() -> None:
    """空 frame：裁剪后仍为空，且 datetime 列类型正常。"""
    df = _empty_frame()
    as_of = _T2

    result = clip_to_as_of(df, as_of)

    assert result.height == 0
    assert result.schema["datetime"] == pl.Datetime("us")


# --- effective_right_bound -------------------------------------------------


def test_right_bound_as_of_before_all_returns_none() -> None:
    """as_of 早于全部数据：裁剪后为空，有效右边界为 None。"""
    df = _frame([_T1, _T2, _T3])
    as_of = datetime(2024, 1, 1, 9, 0)

    assert effective_right_bound(df, as_of) is None


def test_right_bound_as_of_after_all_returns_max() -> None:
    """as_of 晚于全部数据：有效右边界收窄为本地实际最大时间。"""
    df = _frame([_T0, _T1, _T2])
    as_of = datetime(2024, 1, 1, 10, 0)

    assert effective_right_bound(df, as_of) == _T2


def test_right_bound_as_of_in_middle_returns_clipped_max() -> None:
    """as_of 落在中间：有效右边界等于裁剪后最大时间（<= as_of）。"""
    df = _frame([_T0, _T1, _T2, _T3, _T4])
    as_of = _T2

    bound = effective_right_bound(df, as_of)

    assert bound == _T2
    assert bound <= as_of


def test_right_bound_empty_frame_returns_none() -> None:
    """空 frame：有效右边界为 None。"""
    df = _empty_frame()
    as_of = _T2

    assert effective_right_bound(df, as_of) is None
