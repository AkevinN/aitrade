"""数据准备 Parquet 上传特性的正确性属性测试（Hypothesis）。

对应 .kiro/specs/data-prepare-parquet-upload/design.md 的 Correctness Properties 1-8，
每条属性一个独立测试，@settings(max_examples=100)，注释标注属性编号与文本。
存储一律用临时目录；无外部 I/O；时间相关用注入的 now / os.utime 保证确定性。
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.alpha.lab import AlphaLab
from aitrade.alpha.lab_utils import canonical_vt_symbol_from_stem
from aitrade.alpha.parquet_upload import ParquetUploadStaging
from aitrade.api import alpha as alpha_api
from aitrade.api import alpha_service

_BAR_CANON = ["datetime", "open", "high", "low", "close", "volume", "turnover", "open_interest"]

# 6 位数字证券代码（任意 6 位串都能被 normalize 推断出交易所，故均有效）。
_codes6 = st.integers(min_value=0, max_value=999999).map(lambda n: f"{n:06d}")


def _make_bar_frame(n: int, base_price: float = 10.0) -> pl.DataFrame:
    """构造 n 行规范 Bar_Frame schema 的 DataFrame（datetime 递增，避免歧义）。"""
    return pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "open": [base_price] * n,
            "high": [base_price + 0.5] * n,
            "low": [base_price - 0.5] * n,
            "close": [base_price + 0.1] * n,
            "volume": [100.0] * n,
            "turnover": [1000.0] * n,
            "open_interest": [0.0] * n,
        }
    )


def _stage_bar(staging: ParquetUploadStaging, session: str, name: str, n: int) -> None:
    """把 n 行 bar parquet 经暂存层落盘（测试辅助）。"""
    buf = io.BytesIO()
    _make_bar_frame(n).write_parquet(buf)
    buf.seek(0)
    staging.stage_stream(session, name, buf, max_file_bytes=50_000_000)


@contextlib.contextmanager
def _fresh_session_dirs():
    """临时把 alpha_api 的 lab/暂存全局路径切到全新临时目录，退出时还原并清理。

    供 _import_parquet_session（依赖这两个模块级路径）的属性测试在每个 example
    使用隔离的文件系统状态。
    """
    lab_dir = tempfile.mkdtemp()
    stage_dir = tempfile.mkdtemp()
    orig_lab = alpha_api.ALPHA_LAB_PATH
    orig_stage = alpha_api.PARQUET_UPLOAD_STAGING_PATH
    alpha_api.ALPHA_LAB_PATH = Path(lab_dir)
    alpha_api.PARQUET_UPLOAD_STAGING_PATH = Path(stage_dir)
    try:
        yield Path(lab_dir), Path(stage_dir)
    finally:
        alpha_api.ALPHA_LAB_PATH = orig_lab
        alpha_api.PARQUET_UPLOAD_STAGING_PATH = orig_stage
        shutil.rmtree(lab_dir, ignore_errors=True)
        shutil.rmtree(stage_dir, ignore_errors=True)


@settings(max_examples=100, deadline=None)
@given(code=_codes6)
def test_property1_filename_symbol_matches_canonical(code: str) -> None:
    # Feature: data-prepare-parquet-upload, Property 1: 文件名→代码推断与 canonical_vt_symbol_from_stem 一致
    with tempfile.TemporaryDirectory() as d:
        lab = AlphaLab(d)
        path = Path(d) / f"{code}.parquet"
        _make_bar_frame(2).write_parquet(path)

        res = lab.import_parquet_path(path, data_kind="bar", interval="d")

        assert res["success"] is True
        assert res["vt_symbols"] == [canonical_vt_symbol_from_stem(code)]


@settings(max_examples=100, deadline=None)
@given(n=st.integers(min_value=1, max_value=50))
def test_property2_roundtrip_rows_and_idempotent(n: int) -> None:
    # Feature: data-prepare-parquet-upload, Property 2: 暂存→导入往返不变 + 规范化幂等
    with tempfile.TemporaryDirectory() as d:
        lab = AlphaLab(d)
        df = _make_bar_frame(n)

        out1, missing1, _ = lab._normalize_uploaded_frame(df, data_kind="bar")
        assert missing1 == []
        assert out1.columns == _BAR_CANON
        assert out1.height == n

        out2, _missing2, _ = lab._normalize_uploaded_frame(out1, data_kind="bar")
        assert out1.equals(out2)  # 幂等

        path = Path(d) / "600000.parquet"
        df.write_parquet(path)
        res = lab.import_parquet_path(path, data_kind="bar", interval="d")
        assert res["batches"][0]["row_count"] == n  # 往返行数不变


@settings(max_examples=100, deadline=None)
@given(codes=st.lists(_codes6, min_size=1, max_size=5, unique=True))
def test_property3_batch_isolation_only_imports(codes: list[str]) -> None:
    # Feature: data-prepare-parquet-upload, Property 3: 批次隔离——只写 imports 层
    with _fresh_session_dirs() as (lab_dir, stage_dir):
        staging = ParquetUploadStaging(stage_dir)
        for code in codes:
            _stage_bar(staging, "s", f"{code}.parquet", 3)

        res = alpha_service._import_parquet_session("s", data_kind="bar", interval="d")

        resources = AlphaLab(lab_dir).list_data_resources()
        assert resources["raw_bars"] == []  # 正式资源未被触碰
        assert res["success"] == len(codes)
        assert len(resources["raw_bar_batches"]) == len(codes)
        assert staging.list_files("s") == []  # 成功后会话清理（Property 8 的成功删除半边）


@settings(max_examples=100, deadline=None)
@given(flags=st.lists(st.booleans(), min_size=1, max_size=6))
def test_property4_single_file_failure_isolation(flags: list[bool]) -> None:
    # Feature: data-prepare-parquet-upload, Property 4: 单文件失败隔离（success+failed==total）
    with _fresh_session_dirs() as (_lab_dir, stage_dir):
        staging = ParquetUploadStaging(stage_dir)
        good = 0
        for i, ok in enumerate(flags):
            frame = _make_bar_frame(2) if ok else _make_bar_frame(2).drop("close")
            buf = io.BytesIO()
            frame.write_parquet(buf)
            buf.seek(0)
            staging.stage_stream("s", f"{600000 + i:06d}.parquet", buf, max_file_bytes=50_000_000)
            good += int(ok)

        res = alpha_service._import_parquet_session("s", data_kind="bar", interval="d")

        assert res["total"] == len(flags)
        assert res["success"] == good
        assert res["failed"] == len(flags) - good
        assert res["success"] + res["failed"] == res["total"]


@settings(max_examples=100, deadline=None)
@given(drop_col=st.sampled_from(["open", "high", "low", "close"]))
def test_property5_missing_required_blocked(drop_col: str) -> None:
    # Feature: data-prepare-parquet-upload, Property 5: 缺必填列即拦，不入批次
    with tempfile.TemporaryDirectory() as d:
        lab = AlphaLab(d)
        path = Path(d) / "600000.parquet"
        _make_bar_frame(3).drop(drop_col).write_parquet(path)

        preview = lab.preview_parquet_path(path, data_kind="bar")
        assert preview["importable"] is False
        assert drop_col in preview["missing_required"]

        res = lab.import_parquet_path(path, data_kind="bar", interval="d")
        assert res["success"] is False
        assert lab.list_data_resources()["raw_bar_batches"] == []


@settings(max_examples=100, deadline=None)
@given(
    size=st.integers(min_value=0, max_value=20000),
    limit=st.integers(min_value=1, max_value=20000),
    chunk=st.integers(min_value=1, max_value=4096),
)
def test_property6_streaming_cap(size: int, limit: int, chunk: int) -> None:
    # Feature: data-prepare-parquet-upload, Property 6: 流式落盘内存有界 + 超限抛错
    with tempfile.TemporaryDirectory() as d:
        staging = ParquetUploadStaging(d)
        data = b"x" * size
        if size <= limit:
            staged = staging.stage_stream(
                "s", "a.parquet", io.BytesIO(data), max_file_bytes=limit, chunk_bytes=chunk
            )
            assert staged.size_bytes == size
            assert staged.path.read_bytes() == data
        else:
            with pytest.raises(ValueError):
                staging.stage_stream(
                    "s", "a.parquet", io.BytesIO(data), max_file_bytes=limit, chunk_bytes=chunk
                )
            assert staging.list_files("s") == []  # 超限不留半截文件


@settings(max_examples=100, deadline=None)
@given(spec=st.lists(st.tuples(_codes6, st.integers(min_value=1, max_value=5)),
                     min_size=2, max_size=5, unique_by=lambda t: t[0]))
def test_property7_symbol_column_split(spec: list[tuple[str, int]]) -> None:
    # Feature: data-prepare-parquet-upload, Property 7: symbol 列多代码按列拆分
    with tempfile.TemporaryDirectory() as d:
        lab = AlphaLab(d)
        rows: list[dict] = []
        for code, count in spec:
            vt = canonical_vt_symbol_from_stem(code)
            for i in range(count):
                rows.append({
                    "vt_symbol": vt,
                    "datetime": datetime(2024, 1, 1) + timedelta(days=i),
                    "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1,
                })
        path = Path(d) / "mixed.parquet"
        pl.DataFrame(rows).write_parquet(path)

        res = lab.import_parquet_path(path, data_kind="bar", interval="d")

        expected = {canonical_vt_symbol_from_stem(code) for code, _ in spec}
        assert res["success"] is True
        assert set(res["vt_symbols"]) == expected
        assert len(lab.list_data_resources()["raw_bar_batches"]) == len(expected)


@settings(max_examples=100, deadline=None)
@given(age=st.integers(min_value=0, max_value=1000), ttl=st.integers(min_value=0, max_value=1000))
def test_property8_cleanup_expired_ttl(age: int, ttl: int) -> None:
    # Feature: data-prepare-parquet-upload, Property 8: 暂存清理——mtime 超 TTL 即回收
    with tempfile.TemporaryDirectory() as d:
        staging = ParquetUploadStaging(d)
        staging.stage_stream("s", "a.parquet", io.BytesIO(b"a"), max_file_bytes=100)
        base = 1_000_000.0
        session_dir = Path(d) / "s"
        os.utime(session_dir, (base - age, base - age))

        removed = staging.cleanup_expired(ttl_seconds=ttl, now=base)

        if age > ttl:
            assert removed == 1
            assert not session_dir.exists()
        else:
            assert removed == 0
            assert session_dir.exists()
