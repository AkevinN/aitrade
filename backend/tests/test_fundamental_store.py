"""
FundamentalStore 单元测试（Phase 5 Task 5.1）。

覆盖：
1. save / load 往返一致性
2. 增量追加合并（已有数据 + 新数据，行数正确）
3. 按 datetime 去重（新数据优先）
4. start/end 过滤
5. load 文件不存在时返回 None
6. list_symbols 返回正确的 vt_symbol
7. 原子写：无 .tmp.parquet 残留
8. 单位注释存在（docstring 含 "万元"）
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from aitrade.rules.store import FundamentalStore


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------


def _make_df(dates: list[str], total_mv_base: float = 1_000_000.0) -> pl.DataFrame:
    """构造基本面 DataFrame，datetime 列为 YYYYMMDD 字符串。"""
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "datetime": d,
            "pe": 20.0 + i,
            "pe_ttm": 19.5 + i,
            "pb": 2.0 + i * 0.1,
            "total_mv": total_mv_base + i * 1000,
            "circ_mv": total_mv_base * 0.8 + i * 500,
            "turnover_rate": 1.5 + i * 0.01,
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 测试 1：往返一致性
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """save 后 load 应返回内容一致的 DataFrame。"""
    store = FundamentalStore(base_path=tmp_path)
    df_in = _make_df(["20240101", "20240102", "20240103"])

    store.save("600519.SSE", df_in)
    df_out = store.load("600519.SSE")

    assert df_out is not None
    assert df_out.height == 3
    # datetime 已转为 Date 类型
    assert df_out["datetime"].dtype in (pl.Date, pl.Datetime)
    # pe 列数值一致
    assert df_out["pe"].to_list() == pytest.approx([20.0, 21.0, 22.0])
    # total_mv 列存在（单位：万元，原样保存）
    assert "total_mv" in df_out.columns


# ---------------------------------------------------------------------------
# 测试 2：增量追加合并
# ---------------------------------------------------------------------------


def test_incremental_append(tmp_path: Path) -> None:
    """第二次 save 应与已有数据合并，总行数 = 旧 + 新（无重叠）。"""
    store = FundamentalStore(base_path=tmp_path)

    df1 = _make_df(["20240101", "20240102"])
    df2 = _make_df(["20240103", "20240104"])

    store.save("000001.SZSE", df1)
    store.save("000001.SZSE", df2)

    df_out = store.load("000001.SZSE")
    assert df_out is not None
    assert df_out.height == 4, f"期望 4 行，实际 {df_out.height}"


# ---------------------------------------------------------------------------
# 测试 3：按 datetime 去重（新数据优先）
# ---------------------------------------------------------------------------


def test_dedup_keeps_new_data(tmp_path: Path) -> None:
    """相同 datetime 重叠时，新数据（第二次 save）的值应覆盖旧数据。"""
    store = FundamentalStore(base_path=tmp_path)

    # 旧：20240101 pe=10.0
    df_old = pl.DataFrame({
        "datetime": ["20240101"],
        "pe": [10.0],
        "pe_ttm": [9.5],
        "pb": [1.5],
        "total_mv": [100_000.0],
        "circ_mv": [80_000.0],
        "turnover_rate": [1.0],
    })
    # 新：20240101 pe=99.0（覆盖）
    df_new = pl.DataFrame({
        "datetime": ["20240101"],
        "pe": [99.0],
        "pe_ttm": [98.5],
        "pb": [9.9],
        "total_mv": [999_000.0],
        "circ_mv": [888_000.0],
        "turnover_rate": [9.0],
    })

    store.save("510300.SSE", df_old)
    store.save("510300.SSE", df_new)

    df_out = store.load("510300.SSE")
    assert df_out is not None
    assert df_out.height == 1, f"去重后应只有 1 行，实际 {df_out.height}"
    assert df_out["pe"][0] == pytest.approx(99.0), "新数据应覆盖旧数据"


# ---------------------------------------------------------------------------
# 测试 4：start/end 日期过滤
# ---------------------------------------------------------------------------


def test_load_with_date_filter(tmp_path: Path) -> None:
    """load 传入 start/end 时应过滤返回行。"""
    store = FundamentalStore(base_path=tmp_path)
    dates = ["20240101", "20240102", "20240103", "20240104", "20240105"]
    store.save("600036.SSE", _make_df(dates))

    # 只取中间三天
    df_mid = store.load("600036.SSE", start=date(2024, 1, 2), end=date(2024, 1, 4))
    assert df_mid is not None
    assert df_mid.height == 3

    # 只取最后一天
    df_last = store.load("600036.SSE", start=date(2024, 1, 5))
    assert df_last is not None
    assert df_last.height == 1

    # 只取前两天
    df_first = store.load("600036.SSE", end=date(2024, 1, 2))
    assert df_first is not None
    assert df_first.height == 2


# ---------------------------------------------------------------------------
# 测试 5：load 文件不存在时返回 None
# ---------------------------------------------------------------------------


def test_load_returns_none_when_not_exists(tmp_path: Path) -> None:
    """未保存的标的 load 应返回 None。"""
    store = FundamentalStore(base_path=tmp_path)
    assert store.load("999999.SSE") is None


# ---------------------------------------------------------------------------
# 测试 6：list_symbols
# ---------------------------------------------------------------------------


def test_list_symbols(tmp_path: Path) -> None:
    """list_symbols 应返回所有已保存的 vt_symbol，按字母顺序。"""
    store = FundamentalStore(base_path=tmp_path)

    store.save("000001.SZSE", _make_df(["20240101"]))
    store.save("600519.SSE", _make_df(["20240101"]))

    symbols = store.list_symbols()
    assert "000001.SZSE" in symbols
    assert "600519.SSE" in symbols
    assert len(symbols) == 2


# ---------------------------------------------------------------------------
# 测试 7：原子写无临时文件残留
# ---------------------------------------------------------------------------


def test_no_tmp_leftover(tmp_path: Path) -> None:
    """save 成功后不应有 .tmp.parquet 残留。"""
    store = FundamentalStore(base_path=tmp_path)
    store.save("600519.SSE", _make_df(["20240101"]))

    fund_dir = tmp_path / "fundamental"
    tmp_files = list(fund_dir.glob("*.tmp.parquet"))
    assert not tmp_files, f"发现残留临时文件：{tmp_files}"


# ---------------------------------------------------------------------------
# 测试 8：单位注释存在（docstring 含 "万元"）
# ---------------------------------------------------------------------------


def test_unit_annotation_in_docstring() -> None:
    """FundamentalStore 类文档字符串必须含 '万元' 以标明 total_mv/circ_mv 单位。"""
    doc = FundamentalStore.__doc__ or ""
    assert "万元" in doc, (
        "FundamentalStore docstring 应注明 total_mv/circ_mv 单位为万元，"
        f"实际 docstring: {doc!r}"
    )


# ---------------------------------------------------------------------------
# 测试 9：非法 datetime 格式抛 ValueError（fail-fast 守护）
# ---------------------------------------------------------------------------


def test_save_bad_datetime_format_raises(tmp_path: Path) -> None:
    """datetime 列使用 YYYY-MM-DD 格式（非 YYYYMMDD）时，save 应抛 ValueError。"""
    store = FundamentalStore(base_path=tmp_path)
    bad_df = pl.DataFrame({
        "datetime": ["2024-01-01", "2024-01-02"],  # 连字符格式，非法
        "pe": [20.0, 21.0],
        "pe_ttm": [19.5, 20.5],
        "pb": [2.0, 2.1],
        "total_mv": [1_000_000.0, 1_001_000.0],
        "circ_mv": [800_000.0, 800_500.0],
        "turnover_rate": [1.5, 1.51],
    })
    with pytest.raises(ValueError, match="YYYYMMDD"):
        store.save("600519.SSE", bad_df)


def test_save_valid_yyyymmdd_no_regression(tmp_path: Path) -> None:
    """正常 YYYYMMDD 格式不应触发异常（回归保护）。"""
    store = FundamentalStore(base_path=tmp_path)
    good_df = _make_df(["20240101", "20240102"])
    store.save("600519.SSE", good_df)  # 不应抛出
    df_out = store.load("600519.SSE")
    assert df_out is not None
    assert df_out.height == 2
