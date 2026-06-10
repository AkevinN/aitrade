from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from aitrade.cnn import model as cnn_model
from aitrade.cnn import dataset as cnn_dataset


def _make_frame(start: datetime, count: int, offset: int = 0) -> pl.DataFrame:
    rows: list[dict] = []
    for index in range(offset, offset + count):
        current = start + timedelta(days=index)
        close = 100 + index
        rows.append(
            {
                "datetime": current,
                "open": close - 1,
                "high": close + 1,
                "low": close - 2,
                "close": close,
                "volume": 1_000 + index,
                "turnover": (1_000 + index) * close,
                "open_interest": float(index),
            }
        )
    return pl.DataFrame(rows)


def test_normalize_observation_groups_converts_prefixed_symbols() -> None:
    target_symbol, groups = cnn_model.normalize_observation_groups(
        target_symbol="sz000415",
        observation_groups=[
            {"role": "market", "name": "大盘", "symbols": ["sh000001"]},
        ],
        vt_symbols=["sz000415", "sh000001"],
    )

    assert target_symbol == "000415.SZSE"
    assert groups[0]["symbols"] == ["000415.SZSE"]
    assert groups[1]["symbols"] == ["000001.SSE"]


def test_build_dataset_with_semantic_groups(monkeypatch) -> None:
    start_dt = datetime(2024, 1, 1)

    def fake_loader(
        vt_symbol: str,
        start: date,
        end: date,
        *,
        input_data_kind: str,
        input_interval: str,
    ) -> pl.DataFrame:
        offsets = {
            "AAA.SSE": 0,
            "INDEX.SSE": 0,
            "BBB.SSE": 0,
            "CCC.SSE": 0,
        }
        return _make_frame(start_dt, 50, offset=offsets[vt_symbol])

    monkeypatch.setattr(cnn_dataset, "_load_market_frame", fake_loader)

    X, y, mask, info = cnn_model.build_dataset(
        vt_symbols=["AAA.SSE", "INDEX.SSE", "BBB.SSE", "CCC.SSE"],
        start=date(2024, 1, 1),
        end=date(2024, 3, 31),
        lookback=10,
        target_symbol="AAA.SSE",
        input_data_kind="bar",
        input_interval="d",
        observation_groups=[
            {"role": "market", "name": "大盘", "symbols": ["INDEX.SSE"]},
            {"role": "leaders", "name": "龙头", "symbols": ["BBB.SSE", "CCC.SSE"]},
        ],
        label_spec={"mode": "next_bar"},
    )

    assert info["target_symbol"] == "AAA.SSE"
    assert info["group_count"] == 3
    assert info["max_group_width"] == 2
    assert X.shape[0] == len(y)
    assert X.shape[1:] == (6, 10, 2, 3)
    assert mask.shape == (1, 1, 1, 2, 3)
    assert mask[0, 0, 0, 1, 0] == 0.0
    assert mask[0, 0, 0, 1, 2] == 1.0


def test_build_dataset_requires_local_data_for_all_symbols(monkeypatch) -> None:
    def fake_loader(
        vt_symbol: str,
        start: date,
        end: date,
        *,
        input_data_kind: str,
        input_interval: str,
    ) -> pl.DataFrame:
        if vt_symbol == "AAA.SSE":
            return _make_frame(datetime(2024, 1, 1), 40)
        raise ValueError("missing local bars")

    monkeypatch.setattr(cnn_dataset, "_load_market_frame", fake_loader)

    with pytest.raises(ValueError, match="不会使用 mock 数据"):
        cnn_model.build_dataset(
            vt_symbols=["AAA.SSE", "BBB.SSE"],
            start=date(2024, 1, 1),
            end=date(2024, 2, 29),
            lookback=10,
        )


def test_session_close_requires_intraday_input(monkeypatch) -> None:
    def fake_loader(
        vt_symbol: str,
        start: date,
        end: date,
        *,
        input_data_kind: str,
        input_interval: str,
    ) -> pl.DataFrame:
        return _make_frame(datetime(2024, 1, 1), 50)

    monkeypatch.setattr(cnn_dataset, "_load_market_frame", fake_loader)

    with pytest.raises(ValueError, match="session_close"):
        cnn_model.build_dataset(
            vt_symbols=["AAA.SSE"],
            start=date(2024, 1, 1),
            end=date(2024, 3, 31),
            lookback=10,
            target_symbol="AAA.SSE",
            input_data_kind="bar",
            input_interval="d",
            label_spec={"mode": "session_close"},
        )
