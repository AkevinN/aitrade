"""数据准备 Parquet 上传特性的 lab 层 + 暂存层单元/属性测试。

覆盖：
- 帧原生批次写入 ``_save_import_batch_frame``（records 路径重构后的共用入口）；
- parquet 帧规范化 / 预览 / 导入（``_normalize_uploaded_frame`` /
  ``preview_parquet_path`` / ``import_parquet_path``）；
- 暂存会话 ``ParquetUploadStaging``（流式落盘 / TTL）。
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from aitrade.alpha.lab import AlphaLab


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
