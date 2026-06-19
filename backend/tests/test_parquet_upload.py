"""数据准备 Parquet 上传特性的 lab 层 + 暂存层单元/属性测试。

覆盖：
- 帧原生批次写入 ``_save_import_batch_frame``（records 路径重构后的共用入口）；
- parquet 帧规范化 / 预览 / 导入（``_normalize_uploaded_frame`` /
  ``preview_parquet_path`` / ``import_parquet_path``）；
- 暂存会话 ``ParquetUploadStaging``（流式落盘 / TTL）。
"""

from __future__ import annotations

import io
import time
from datetime import datetime

import polars as pl
import pytest

from aitrade.alpha.lab import AlphaLab
from aitrade.alpha.parquet_upload import ParquetUploadStaging
from aitrade.api import alpha as alpha_api
from aitrade.api import alpha_service


def _stage_frame(staging: ParquetUploadStaging, session_id: str, name: str, df: pl.DataFrame) -> None:
    """把一张 DataFrame 写成 parquet 字节并经暂存层落盘（测试辅助）。"""
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    staging.stage_stream(session_id, name, buf, max_file_bytes=50_000_000)


def _bar_frame() -> pl.DataFrame:
    """构造一张规范 Bar_Frame schema 的两行 DataFrame。"""
    return pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.8, 9.9],
            "close": [10.1, 10.2],
            "volume": [100.0, 110.0],
            "turnover": [1000.0, 1100.0],
            "open_interest": [0.0, 0.0],
        }
    )


def test_save_import_batch_frame_writes_pending_batch(tmp_path) -> None:
    """帧入口把 DataFrame 写成 pending 批次，不进正式资源。"""
    lab = AlphaLab(tmp_path)

    summary = lab._save_import_batch_frame(
        data_kind="bar",
        vt_symbol="000001.SZSE",
        interval="d",
        df=_bar_frame(),
        file_name="000001.parquet",
        source="upload",
    )

    assert summary["row_count"] == 2
    assert summary["vt_symbol"] == "000001.SZSE"
    assert summary["status"] == "pending"

    resources = lab.list_data_resources()
    assert resources["raw_bars"] == []  # 未写正式资源
    assert len(resources["raw_bar_batches"]) == 1


# --- _normalize_uploaded_frame ------------------------------------------------

_BAR_CANON = ["datetime", "open", "high", "low", "close", "volume", "turnover", "open_interest"]
_TICK_CANON = [
    "datetime", "last_price", "volume", "turnover",
    "bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1",
]


def test_normalize_frame_standard_bar_columns(tmp_path) -> None:
    """标准列 + 多余列 → 规范 Bar_Frame schema，无缺列。"""
    lab = AlphaLab(tmp_path)
    df = _bar_frame().with_columns(pl.lit("noise").alias("extra_col"))

    out, missing, _ = lab._normalize_uploaded_frame(df, data_kind="bar")

    assert missing == []
    assert out.columns == _BAR_CANON
    assert out.height == 2


def test_normalize_frame_chinese_aliases(tmp_path) -> None:
    """中文别名列名经 Field_Mapping 映射到标准字段。"""
    lab = AlphaLab(tmp_path)
    df = pl.DataFrame(
        {
            "交易日期": ["2024-01-02", "2024-01-03"],
            "开盘价": [10.0, 10.1],
            "最高价": [10.2, 10.3],
            "最低价": [9.8, 9.9],
            "收盘价": [10.1, 10.2],
            "成交量": [100.0, 110.0],
        }
    )

    out, missing, _ = lab._normalize_uploaded_frame(df, data_kind="bar")

    assert missing == []
    assert out.columns == _BAR_CANON
    assert out["close"].to_list() == [10.1, 10.2]


def test_normalize_frame_missing_required(tmp_path) -> None:
    """缺必填列（close）→ missing_required 含 close，不规范化。"""
    lab = AlphaLab(tmp_path)
    df = _bar_frame().drop("close")

    _, missing, _ = lab._normalize_uploaded_frame(df, data_kind="bar")

    assert "close" in missing


def test_normalize_frame_string_datetime_parsed(tmp_path) -> None:
    """datetime 列为字符串 → 解析为 Datetime 类型。"""
    lab = AlphaLab(tmp_path)
    df = _bar_frame().with_columns(
        pl.Series("datetime", ["2024-01-02", "2024-01-03"])
    )

    out, missing, _ = lab._normalize_uploaded_frame(df, data_kind="bar")

    assert missing == []
    assert out.schema["datetime"] == pl.Datetime


def test_normalize_frame_tick_schema(tmp_path) -> None:
    """tick 数据规范化为 Tick_Frame schema。"""
    lab = AlphaLab(tmp_path)
    df = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2, 9, 30), datetime(2024, 1, 2, 9, 31)],
            "last_price": [10.0, 10.1],
            "volume": [100.0, 110.0],
        }
    )

    out, missing, _ = lab._normalize_uploaded_frame(df, data_kind="tick")

    assert missing == []
    assert out.columns == _TICK_CANON


# --- preview_parquet_path -----------------------------------------------------


def test_preview_parquet_filename_symbol_szse(tmp_path) -> None:
    """文件名裸码 000001 → 000001.SZSE，含行数/时间范围，可导入。"""
    lab = AlphaLab(tmp_path)
    p = tmp_path / "000001.parquet"
    _bar_frame().write_parquet(p)

    prev = lab.preview_parquet_path(p, data_kind="bar")

    assert prev["vt_symbol"] == "000001.SZSE"
    assert prev["row_count"] == 2
    assert prev["importable"] is True
    assert prev["missing_required"] == []
    assert prev["date_range"] == ("2024-01-02", "2024-01-03")


def test_preview_parquet_filename_symbol_sse(tmp_path) -> None:
    """文件名裸码 600000 → 600000.SSE（沪市）。"""
    lab = AlphaLab(tmp_path)
    p = tmp_path / "600000.parquet"
    _bar_frame().write_parquet(p)

    assert lab.preview_parquet_path(p, data_kind="bar")["vt_symbol"] == "600000.SSE"


def test_preview_parquet_missing_column_not_importable(tmp_path) -> None:
    """缺必填列 → importable=False 且 missing_required 含该列。"""
    lab = AlphaLab(tmp_path)
    p = tmp_path / "000001.parquet"
    _bar_frame().drop("close").write_parquet(p)

    prev = lab.preview_parquet_path(p, data_kind="bar")

    assert prev["importable"] is False
    assert "close" in prev["missing_required"]


def test_preview_parquet_corrupt_file_not_importable(tmp_path) -> None:
    """坏 parquet → importable=False，带原因，不抛错。"""
    lab = AlphaLab(tmp_path)
    p = tmp_path / "bad.parquet"
    p.write_bytes(b"not a parquet file")

    prev = lab.preview_parquet_path(p, data_kind="bar")

    assert prev["importable"] is False
    assert prev["reason"]


# --- import_parquet_path ------------------------------------------------------


def test_import_parquet_filename_creates_batch(tmp_path) -> None:
    """文件名模式：整文件一支股票，落 pending 批次，不进正式资源。"""
    lab = AlphaLab(tmp_path)
    p = tmp_path / "600000.parquet"
    _bar_frame().write_parquet(p)

    res = lab.import_parquet_path(p, data_kind="bar", interval="d")

    assert res["success"] is True
    assert res["vt_symbols"] == ["600000.SSE"]
    resources = lab.list_data_resources()
    assert len(resources["raw_bar_batches"]) == 1
    assert resources["raw_bars"] == []


def test_import_parquet_symbol_column_splits_by_symbol(tmp_path) -> None:
    """列模式：单文件含多代码 → 按代码拆分成多个批次（安全网）。"""
    lab = AlphaLab(tmp_path)
    df = pl.DataFrame(
        {
            "vt_symbol": ["000001.SZSE", "000001.SZSE", "600000.SSE"],
            "datetime": [datetime(2024, 1, 2), datetime(2024, 1, 3), datetime(2024, 1, 2)],
            "open": [1.0, 1.1, 2.0],
            "high": [1.2, 1.3, 2.2],
            "low": [0.9, 1.0, 1.9],
            "close": [1.1, 1.2, 2.1],
        }
    )
    p = tmp_path / "mixed.parquet"
    df.write_parquet(p)

    res = lab.import_parquet_path(p, data_kind="bar", interval="d")

    assert res["success"] is True
    assert set(res["vt_symbols"]) == {"000001.SZSE", "600000.SSE"}
    assert len(lab.list_data_resources()["raw_bar_batches"]) == 2


def test_import_parquet_missing_column_no_batch(tmp_path) -> None:
    """缺必填列 → success=False，不写任何批次。"""
    lab = AlphaLab(tmp_path)
    p = tmp_path / "000001.parquet"
    _bar_frame().drop("close").write_parquet(p)

    res = lab.import_parquet_path(p, data_kind="bar", interval="d")

    assert res["success"] is False
    assert lab.list_data_resources()["raw_bar_batches"] == []


# --- ParquetUploadStaging -----------------------------------------------------


def test_stage_stream_writes_parquet_chunked(tmp_path) -> None:
    """流式分块落盘 .parquet：内容完整、标记 is_parquet、记录大小。"""
    staging = ParquetUploadStaging(tmp_path / "stage")
    data = b"x" * 5000

    sf = staging.stage_stream(
        "sess1", "000001.parquet", io.BytesIO(data), max_file_bytes=10_000, chunk_bytes=1024
    )

    assert sf.is_parquet is True
    assert sf.size_bytes == 5000
    assert sf.path.read_bytes() == data


def test_stage_stream_non_parquet_flagged(tmp_path) -> None:
    """非 .parquet 文件落盘但标记 is_parquet=False。"""
    staging = ParquetUploadStaging(tmp_path / "stage")

    sf = staging.stage_stream("s", "notes.txt", io.BytesIO(b"hi"), max_file_bytes=10_000)

    assert sf.is_parquet is False


def test_stage_stream_exceeds_limit_raises_and_cleans(tmp_path) -> None:
    """超单文件上限 → 抛错且不留半截文件。"""
    staging = ParquetUploadStaging(tmp_path / "stage")

    with pytest.raises(ValueError):
        staging.stage_stream(
            "s", "big.parquet", io.BytesIO(b"x" * 5000), max_file_bytes=1000, chunk_bytes=256
        )

    assert staging.list_files("s") == []


def test_stage_list_and_discard(tmp_path) -> None:
    """list_files 列出会话内文件；discard 删除整会话。"""
    staging = ParquetUploadStaging(tmp_path / "stage")
    staging.stage_stream("s", "a.parquet", io.BytesIO(b"a"), max_file_bytes=100)
    staging.stage_stream("s", "b.parquet", io.BytesIO(b"b"), max_file_bytes=100)

    assert {f.file_name for f in staging.list_files("s")} == {"a.parquet", "b.parquet"}

    staging.discard("s")
    assert staging.list_files("s") == []


def test_stage_cleanup_expired(tmp_path) -> None:
    """cleanup_expired 回收超 TTL 的历史会话（注入 now 保证确定性）。"""
    staging = ParquetUploadStaging(tmp_path / "stage")
    staging.stage_stream("old", "a.parquet", io.BytesIO(b"a"), max_file_bytes=100)

    removed = staging.cleanup_expired(ttl_seconds=10, now=time.time() + 1000)

    assert removed == 1
    assert staging.list_files("old") == []


# --- _import_parquet_session（服务编排）---------------------------------------


def test_import_parquet_session_isolates_failures(tmp_path, monkeypatch) -> None:
    """混合会话：好文件入批次，坏文件计入 failed_files，正式资源不动，会话清理。"""
    lab_dir = tmp_path / "lab"
    stage_dir = tmp_path / "stage"
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", lab_dir)
    monkeypatch.setattr(alpha_api, "PARQUET_UPLOAD_STAGING_PATH", stage_dir)

    staging = ParquetUploadStaging(stage_dir)
    session = "sess-x"
    _stage_frame(staging, session, "600000.parquet", _bar_frame())          # 正常
    _stage_frame(staging, session, "000001.parquet", _bar_frame().drop("close"))  # 缺列

    progress: list[tuple[float, str]] = []
    res = alpha_service._import_parquet_session(
        session, data_kind="bar", interval="d", import_mode="merge",
        on_progress=lambda p, m: progress.append((p, m)),
    )

    assert res["total"] == 2
    assert res["success"] == 1
    assert res["failed"] == 1
    assert len(res["failed_files"]) == 1
    assert res["failed_files"][0]["file"] == "000001.parquet"

    resources = AlphaLab(lab_dir).list_data_resources()
    assert resources["raw_bars"] == []                       # 正式资源未动
    assert len(resources["raw_bar_batches"]) == 1            # 仅好文件入批次

    assert staging.list_files(session) == []                 # 会话已清理
    assert progress[-1][0] == 100


# --- API 端点（stage / import / cancel）---------------------------------------


def _parquet_bytes(df: pl.DataFrame) -> bytes:
    """把 DataFrame 序列化为 parquet 字节（测试辅助）。"""
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


def _poll_task(client, task_id: str, timeout: float = 15.0) -> dict:
    """轮询任务状态直到 completed/failed（仿既有集成测试）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/alpha/tasks/{task_id}")
        task = resp.json()
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.05)
    raise AssertionError(f"任务 {task_id} 未在 {timeout}s 内完成")


def test_parquet_stage_and_import_endpoints(monkeypatch, tmp_path) -> None:
    """stage 多文件→预览；import→异步任务→批次落地；正式资源未动。"""
    from fastapi.testclient import TestClient

    from aitrade.main import create_app

    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path / "lab")
    monkeypatch.setattr(alpha_api, "PARQUET_UPLOAD_STAGING_PATH", tmp_path / "stage")

    app = create_app()
    with TestClient(app) as client:
        stage_resp = client.post(
            "/api/alpha/parquet/stage",
            data={"data_kind": "bar"},
            files=[
                ("files", ("600000.parquet", _parquet_bytes(_bar_frame()), "application/octet-stream")),
                ("files", ("000001.parquet", _parquet_bytes(_bar_frame()), "application/octet-stream")),
            ],
        )
        assert stage_resp.status_code == 200
        stage_payload = stage_resp.json()
        session_id = stage_payload["session_id"]
        assert len(stage_payload["files"]) == 2
        symbols = {f["vt_symbol"] for f in stage_payload["files"]}
        assert symbols == {"600000.SSE", "000001.SZSE"}
        assert all(f["importable"] for f in stage_payload["files"])

        import_resp = client.post(
            "/api/alpha/parquet/import",
            json={"session_id": session_id, "data_kind": "bar", "interval": "d", "import_mode": "merge"},
        )
        assert import_resp.status_code == 200
        task_id = import_resp.json()["task_id"]

        task = _poll_task(client, task_id)
        assert task["status"] == "completed"
        assert task["result"]["success"] == 2
        assert task["result"]["failed"] == 0

        resources = client.get("/api/alpha/data/resources").json()
        assert len(resources["raw_bar_batches"]) == 2
        assert resources["raw_bars"] == []


def test_parquet_import_missing_session_404(monkeypatch, tmp_path) -> None:
    """对不存在的会话发起导入 → 404。"""
    from fastapi.testclient import TestClient

    from aitrade.main import create_app

    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path / "lab")
    monkeypatch.setattr(alpha_api, "PARQUET_UPLOAD_STAGING_PATH", tmp_path / "stage")

    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/alpha/parquet/import",
            json={"session_id": "nope", "data_kind": "bar", "interval": "d"},
        )
        assert resp.status_code == 404
        assert "会话" in resp.json()["detail"]  # 确为"会话不存在"，而非路由未注册


def test_import_parquet_unrecognized_filename_no_batch(tmp_path) -> None:
    """文件名无法识别代码 → success=False，不写批次。"""
    lab = AlphaLab(tmp_path)
    p = tmp_path / "garbage_name.parquet"
    _bar_frame().write_parquet(p)

    res = lab.import_parquet_path(p, data_kind="bar", interval="d")

    assert res["success"] is False
    assert lab.list_data_resources()["raw_bar_batches"] == []
