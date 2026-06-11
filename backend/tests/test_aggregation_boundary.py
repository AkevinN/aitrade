"""聚合时间轴正确性回归测试。

覆盖三类问题：
- #1 输出统一为"区间结束时刻"约定（end-labeled）；
- #2 午盘/收盘整点（11:30 / 15:00）数据不被丢弃；
- #3 末尾不完整桶（bar 来源缺少收盘分钟）被剔除，tick 来源不误删。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from aitrade.alpha.lab import AlphaLab, BarData, TickData


def _make_bar(dt: datetime, interval: str = "1m", price: float = 10.0) -> BarData:
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


def test_bar_aggregation_is_end_labeled(tmp_path) -> None:
    """09:31~09:35 的 1m 应聚合为标注 09:35 的 5m（结束时刻）。"""
    lab = AlphaLab(tmp_path)
    bars = [_make_bar(datetime(2024, 1, 2, 9, 31) + timedelta(minutes=i)) for i in range(5)]
    lab.save_bar_data(bars)

    out = lab.aggregate_bar_frame(
        lab.load_bar_frame("000001.SZSE", "1m", datetime(2024, 1, 2), datetime(2024, 1, 2, 23, 59)),
        source_interval="1m",
        target_interval="5m",
    )
    assert out["datetime"].to_list() == [datetime(2024, 1, 2, 9, 35)]


def test_morning_close_1130_not_dropped(tmp_path) -> None:
    """11:30 的源分钟线必须并入 11:30 收盘桶，不能被丢弃（#2）。"""
    lab = AlphaLab(tmp_path)
    # 11:26~11:30 共 5 根，组成结束于 11:30 的完整 5m。
    bars = [_make_bar(datetime(2024, 1, 2, 11, 26) + timedelta(minutes=i)) for i in range(5)]
    lab.save_bar_data(bars)

    out = lab.aggregate_bar_frame(
        lab.load_bar_frame("000001.SZSE", "1m", datetime(2024, 1, 2), datetime(2024, 1, 2, 23, 59)),
        source_interval="1m",
        target_interval="5m",
    )
    assert datetime(2024, 1, 2, 11, 30) in out["datetime"].to_list()


def test_closing_auction_1500_tick_not_dropped(tmp_path) -> None:
    """15:00:00 的收盘 tick 必须计入 15:00 收盘桶（#2）。"""
    lab = AlphaLab(tmp_path)
    ticks = [
        TickData("000001", "SZSE", datetime(2024, 1, 2, 14, 56, 0), 10.0, 10, 100),
        TickData("000001", "SZSE", datetime(2024, 1, 2, 14, 59, 30), 10.5, 10, 105),
        TickData("000001", "SZSE", datetime(2024, 1, 2, 15, 0, 0), 11.0, 20, 220),  # 收盘集合竞价
    ]
    lab.save_tick_data(ticks)

    out = lab.aggregate_tick_frame_to_bars(
        lab.load_tick_frame("000001.SZSE", datetime(2024, 1, 2), datetime(2024, 1, 2, 23, 59)),
        target_interval="5m",
    )
    closing = out.filter(out["datetime"] == datetime(2024, 1, 2, 15, 0))
    assert closing.height == 1
    # 收盘价应来自 15:00:00 的集合竞价 tick。
    assert closing["close"].to_list()[0] == 11.0


def test_trailing_incomplete_bar_bucket_dropped(tmp_path) -> None:
    """bar 来源缺少收盘分钟时，末尾半根 K 线应被剔除（#3）。"""
    lab = AlphaLab(tmp_path)
    # 完整桶 09:31~09:35（=> 09:35），随后仅有 09:36~09:38（缺 09:39/09:40）。
    bars = [_make_bar(datetime(2024, 1, 2, 9, 31) + timedelta(minutes=i)) for i in range(5)]
    bars += [_make_bar(datetime(2024, 1, 2, 9, 36) + timedelta(minutes=i)) for i in range(3)]
    lab.save_bar_data(bars)

    stats: dict = {}
    out = lab.aggregate_bar_frame(
        lab.load_bar_frame("000001.SZSE", "1m", datetime(2024, 1, 2), datetime(2024, 1, 2, 23, 59)),
        source_interval="1m",
        target_interval="5m",
        stats=stats,
    )
    assert out["datetime"].to_list() == [datetime(2024, 1, 2, 9, 35)]
    assert stats["dropped_incomplete"] == 1


def test_trailing_tick_bucket_kept(tmp_path) -> None:
    """tick 来源末尾桶不因未达收盘而被误删（保留合法数据）。"""
    lab = AlphaLab(tmp_path)
    ticks = [
        TickData("000001", "SZSE", datetime(2024, 1, 2, 13, 0, 10), 10.3, 12, 124),
        TickData("000001", "SZSE", datetime(2024, 1, 2, 13, 4, 30), 10.6, 8, 84),
    ]
    lab.save_tick_data(ticks)

    stats: dict = {}
    out = lab.aggregate_tick_frame_to_bars(
        lab.load_tick_frame("000001.SZSE", datetime(2024, 1, 2), datetime(2024, 1, 2, 23, 59)),
        target_interval="5m",
        stats=stats,
    )
    assert out["datetime"].to_list() == [datetime(2024, 1, 2, 13, 5)]
    assert stats["dropped_incomplete"] == 0
