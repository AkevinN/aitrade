"""数据可信度回归测试：复权口径校验、时区归一、并发写安全。"""

from __future__ import annotations

import threading
from datetime import datetime, timezone, timedelta

import polars as pl
import pytest

from aitrade.alpha.lab import AlphaLab, BarData, TickData


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


def test_mixed_adjust_type_rejected(tmp_path) -> None:
    """同一原始资源混用不同复权口径应被拒绝。"""
    lab = AlphaLab(tmp_path)
    lab.save_bar_data([_bar(datetime(2024, 1, 2))], adjust_type="none")
    # 同口径可继续写入。
    lab.save_bar_data([_bar(datetime(2024, 1, 3))], adjust_type="none")
    # 不同口径必须报错。
    with pytest.raises(ValueError, match="复权口径不一致"):
        lab.save_bar_data([_bar(datetime(2024, 1, 4))], adjust_type="qfq")


def test_timezone_normalized_to_exchange_local(tmp_path) -> None:
    """带时区的时间写入后应转换为交易所本地裸时间。"""
    lab = AlphaLab(tmp_path)
    # UTC 01:30 == Asia/Shanghai 09:30。
    aware = datetime(2024, 1, 2, 1, 30, tzinfo=timezone.utc)
    lab.save_bar_data([_bar(aware)], adjust_type="none")

    df = lab.load_bar_frame("000001.SZSE", "d", datetime(2024, 1, 1), datetime(2024, 1, 3))
    stored = df["datetime"].to_list()[0]
    assert stored == datetime(2024, 1, 2, 9, 30)
    assert stored.tzinfo is None


def test_concurrent_bar_writes_do_not_lose_data(tmp_path) -> None:
    """同一合约并发写入不同日期，结果应包含全部行（无 read-modify-write 丢失）。"""
    lab = AlphaLab(tmp_path)
    base = datetime(2024, 1, 1)
    days = 60

    def writer(offset: int) -> None:
        lab.save_bar_data([_bar(base + timedelta(days=offset))], adjust_type="none")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(days)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    df = lab.load_bar_frame("000001.SZSE", "d", base, base + timedelta(days=days + 1))
    assert df.height == days
    assert df["datetime"].n_unique() == days
