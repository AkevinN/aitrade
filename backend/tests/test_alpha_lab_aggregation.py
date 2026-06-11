from __future__ import annotations

from datetime import date, datetime, timedelta

from aitrade.alpha.lab import AlphaLab, BarData, TickData


def test_tick_aggregation_respects_session_boundaries(tmp_path) -> None:
    lab = AlphaLab(tmp_path)
    ticks = [
        TickData("000001", "SZSE", datetime(2024, 1, 2, 11, 29, 0), 10.0, 10, 100),
        TickData("000001", "SZSE", datetime(2024, 1, 2, 11, 29, 40), 10.2, 5, 51),
        TickData("000001", "SZSE", datetime(2024, 1, 2, 13, 0, 10), 10.3, 12, 124),
        TickData("000001", "SZSE", datetime(2024, 1, 2, 13, 4, 30), 10.6, 8, 84),
    ]
    lab.save_tick_data(ticks)

    result = lab.aggregate_market_data(
        ["000001.SZSE"],
        source_kind="tick",
        source_interval=None,
        target_interval="5m",
        start=datetime(2024, 1, 2),
        end=datetime(2024, 1, 2, 23, 59, 59),
    )

    assert result["success"] == 1

    bars = lab.load_bar_data(
        "000001.SZSE",
        "5m",
        datetime(2024, 1, 2),
        datetime(2024, 1, 2, 23, 59, 59),
    )
    # 输出按"区间结束时刻"标注：盘中 11:29 的 tick 归入 11:30 收盘桶，
    # 13:00:10 / 13:04:30 归入 (13:00, 13:05] 桶并标注 13:05。
    assert len(bars) == 2
    assert bars[0].datetime == datetime(2024, 1, 2, 11, 30)
    assert bars[1].datetime == datetime(2024, 1, 2, 13, 5)


def test_tick_aggregation_accepts_same_day_date_bounds(tmp_path) -> None:
    lab = AlphaLab(tmp_path)
    ticks = [
        TickData("000001", "SZSE", datetime(2024, 1, 2, 9, 30, 1), 10.0, 10, 100),
        TickData("000001", "SZSE", datetime(2024, 1, 2, 9, 31, 1), 10.2, 11, 112),
    ]
    lab.save_tick_data(ticks)

    result = lab.aggregate_market_data(
        ["000001.SZSE"],
        source_kind="tick",
        source_interval=None,
        target_interval="5m",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
    )

    assert result["success"] == 1


def test_bar_aggregation_from_one_minute_to_ten_minute(tmp_path) -> None:
    lab = AlphaLab(tmp_path)
    bars = []
    # 源分钟线按"区间结束时刻"标注（下载源约定）：09:31 ~ 09:40 共 10 根，
    # 恰好组成结束于 09:40 的一根 10 分钟 K 线。
    start = datetime(2024, 1, 2, 9, 31)
    for offset in range(10):
        current = start + timedelta(minutes=offset)
        bars.append(
            BarData(
                symbol="000001",
                exchange="SZSE",
                datetime=current,
                interval="1m",
                open_price=10 + offset * 0.1,
                high_price=10.2 + offset * 0.1,
                low_price=9.9 + offset * 0.1,
                close_price=10.1 + offset * 0.1,
                volume=100 + offset,
                turnover=1000 + offset * 10,
            )
        )
    lab.save_bar_data(bars)

    result = lab.aggregate_market_data(
        ["000001.SZSE"],
        source_kind="bar",
        source_interval="1m",
        target_interval="10m",
        start=datetime(2024, 1, 2),
        end=datetime(2024, 1, 2, 23, 59, 59),
    )

    assert result["success"] == 1

    aggregated = lab.load_bar_data(
        "000001.SZSE",
        "10m",
        datetime(2024, 1, 2),
        datetime(2024, 1, 2, 23, 59, 59),
    )
    assert len(aggregated) == 1
    assert aggregated[0].datetime == datetime(2024, 1, 2, 9, 40)
    assert aggregated[0].open_price == 10.0
    assert round(aggregated[0].close_price, 4) == 11.0
