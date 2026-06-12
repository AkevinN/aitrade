"""
CBTermsStore 单元测试（Task 4.2）。

覆盖：
1. save_snapshot / load_snapshot 往返一致性
2. save_premium_history / load_premium_history 往返一致性
3. 原子写：tmp 文件在 replace 成功后消失
4. 文件不存在时 load 返回 None
5. vt_symbol 中的 "." 被替换为 "_" 保证文件名合法
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from aitrade.rules.store import CBTermsStore


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _snapshot_df() -> pl.DataFrame:
    """构造一个最小化快照 DataFrame。"""
    return pl.DataFrame(
        {
            "债券代码": ["113050", "128093"],
            "债券简称": ["富投转债", "岱勒转债"],
            "转股价": [10.5, 8.3],
            "债现价": [108.7, 95.2],
            "转股溢价率": [12.34, 8.91],
            "发行规模": [6.0, 4.5],
            "信用评级": ["AA", "AA-"],
        }
    )


def _premium_df() -> pl.DataFrame:
    """构造一个最小化溢价率历史 DataFrame。"""
    return pl.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "收盘价": [108.7, 109.2, 110.0],
            "转股溢价率": [12.34, 11.80, 11.20],
        }
    )


# ---------------------------------------------------------------------------
# 测试 1：快照往返
# ---------------------------------------------------------------------------


def test_snapshot_roundtrip(tmp_path: Path) -> None:
    """save_snapshot 后 load_snapshot 应返回内容相同的 DataFrame。"""
    store = CBTermsStore(base_path=tmp_path)
    df_in = _snapshot_df()

    store.save_snapshot(df_in)
    df_out = store.load_snapshot()

    assert df_out is not None
    assert df_out.height == df_in.height
    assert list(df_out.columns) == list(df_in.columns)
    # 检查关键列数值一致
    assert df_out["债券代码"].to_list() == df_in["债券代码"].to_list()
    assert df_out["发行规模"].to_list() == pytest.approx(df_in["发行规模"].to_list())


# ---------------------------------------------------------------------------
# 测试 2：溢价率历史往返
# ---------------------------------------------------------------------------


def test_premium_history_roundtrip(tmp_path: Path) -> None:
    """save_premium_history / load_premium_history 往返正确。"""
    store = CBTermsStore(base_path=tmp_path)
    vt_symbol = "113050.SSE"
    df_in = _premium_df()

    store.save_premium_history(vt_symbol, df_in)
    df_out = store.load_premium_history(vt_symbol)

    assert df_out is not None
    assert df_out.height == df_in.height
    assert df_out["日期"].to_list() == df_in["日期"].to_list()
    assert df_out["转股溢价率"].to_list() == pytest.approx(df_in["转股溢价率"].to_list())


# ---------------------------------------------------------------------------
# 测试 3：文件不存在时返回 None
# ---------------------------------------------------------------------------


def test_load_snapshot_returns_none_when_not_exists(tmp_path: Path) -> None:
    """快照文件不存在时 load_snapshot 应返回 None。"""
    store = CBTermsStore(base_path=tmp_path)
    assert store.load_snapshot() is None


def test_load_premium_history_returns_none_when_not_exists(tmp_path: Path) -> None:
    """溢价率历史文件不存在时 load_premium_history 应返回 None。"""
    store = CBTermsStore(base_path=tmp_path)
    assert store.load_premium_history("999999.SSE") is None


# ---------------------------------------------------------------------------
# 测试 4：vt_symbol 中的 "." 替换为 "_"（文件名合法化）
# ---------------------------------------------------------------------------


def test_vt_symbol_dot_replaced_in_filename(tmp_path: Path) -> None:
    """vt_symbol 中的 "." 应被替换为 "_"，生成合法文件名。"""
    store = CBTermsStore(base_path=tmp_path)
    vt_symbol = "128093.SZSE"
    df_in = _premium_df()

    store.save_premium_history(vt_symbol, df_in)

    # 文件名应为 128093_SZSE.parquet
    expected_path = tmp_path / "cb_premium" / "128093_SZSE.parquet"
    assert expected_path.exists(), f"期望文件 {expected_path} 存在"

    df_out = store.load_premium_history(vt_symbol)
    assert df_out is not None and df_out.height == df_in.height


# ---------------------------------------------------------------------------
# 测试 5：原子写 —— tmp 文件在成功后消失
# ---------------------------------------------------------------------------


def test_atomic_write_no_tmp_leftover(tmp_path: Path) -> None:
    """save_snapshot 成功后，不应有 .tmp.parquet 临时文件残留。"""
    store = CBTermsStore(base_path=tmp_path)
    store.save_snapshot(_snapshot_df())

    tmp_files = list(tmp_path.glob("*.tmp.parquet"))
    assert not tmp_files, f"发现残留临时文件：{tmp_files}"


# ---------------------------------------------------------------------------
# 测试 6：覆盖写（二次 save 应覆盖）
# ---------------------------------------------------------------------------


def test_snapshot_overwrite(tmp_path: Path) -> None:
    """二次 save_snapshot 应覆盖已有文件，load 返回最新数据。"""
    store = CBTermsStore(base_path=tmp_path)

    df1 = pl.DataFrame({"债券代码": ["113050"]})
    df2 = pl.DataFrame({"债券代码": ["113050", "128093", "110059"]})

    store.save_snapshot(df1)
    store.save_snapshot(df2)

    df_out = store.load_snapshot()
    assert df_out is not None
    assert df_out.height == 3
