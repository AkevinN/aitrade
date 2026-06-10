"""下载→批次→合并 流程的 lab 层回归测试。

覆盖：
- 下载数据经 save_bars_as_import_batch 落为 pending 批次，不直接进正式资源；
- 批次可经 merge_import_batches 合并/晋级到正式 K 线；
- 已有正式资源时，重叠且一致的批次可并入并向后扩展。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from aitrade.alpha.lab import AlphaLab, BarData


def _bar(dt: datetime, *, interval: str = "d", price: float = 10.0) -> BarData:
    return BarData(
        symbol="000001",
        exchange="SZSE",
        datetime=dt,
        interval=interval,
        open_price=price,
        high_price=price + 0.2,
        low_price=price - 0.2,
        close_price=price + 0.1,
        volume=100.0,
        turnover=1000.0,
    )


def test_save_bars_as_import_batch_does_not_write_official(tmp_path) -> None:
    lab = AlphaLab(tmp_path)
    bars = [_bar(datetime(2024, 1, 2) + timedelta(days=i)) for i in range(3)]

    batch = lab.save_bars_as_import_batch(bars, adjust_type="none", source="download")

    # 批次已落盘且状态 pending、来源 download。
    resources = lab.list_data_resources()
    assert resources["raw_bars"] == []  # 未写正式资源
    assert len(resources["raw_bar_batches"]) == 1
    assert resources["raw_bar_batches"][0]["status"] == "pending"
    assert batch["key"].startswith("batch__raw_bar__")


def test_download_batch_then_merge_promotes_to_official(tmp_path) -> None:
    lab = AlphaLab(tmp_path)
    bars = [_bar(datetime(2024, 1, 2) + timedelta(days=i)) for i in range(3)]
    batch = lab.save_bars_as_import_batch(bars, adjust_type="none", source="download")

    result = lab.merge_import_batches(kind="raw_bar", keys=[batch["key"]])
    assert result["success"] is True
    assert result["has_official"] is False

    resources = lab.list_data_resources()
    assert len(resources["raw_bars"]) == 1
    assert resources["raw_bars"][0]["row_count"] == 3


def test_overlapping_consistent_batch_merges_into_official(tmp_path) -> None:
    lab = AlphaLab(tmp_path)
    # 第一批：01-02 ~ 01-04 晋级为正式。
    first = [_bar(datetime(2024, 1, 2) + timedelta(days=i), price=10 + i) for i in range(3)]
    b1 = lab.save_bars_as_import_batch(first, source="download")
    lab.merge_import_batches(kind="raw_bar", keys=[b1["key"]])

    # 第二批：01-04（与正式一致）~ 01-06，向后扩展。
    second = [_bar(datetime(2024, 1, 4) + timedelta(days=i), price=12 + i) for i in range(3)]
    b2 = lab.save_bars_as_import_batch(second, source="download")

    preview = lab.preview_merge_import_batches(kind="raw_bar", keys=[b2["key"]])
    assert preview["has_official"] is True
    assert preview["can_merge"] is True

    result = lab.merge_import_batches(kind="raw_bar", keys=[b2["key"]])
    assert result["success"] is True
    # 01-02 ~ 01-06 共 5 行。
    assert result["row_count"] == 5
