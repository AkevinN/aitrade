from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest
from fastapi import HTTPException

from aitrade.api import alpha as alpha_api
from aitrade.profiling import Profiler, ProfileStore


class FakeLab:
    def __init__(self, frames: dict[tuple[str, str], pl.DataFrame]) -> None:
        self.frames = frames
        self.aggregate_called = False

    def load_bar_frame(self, vt_symbol: str, interval: str, start, end, *, include_derived: bool = True):
        return self.frames.get((vt_symbol, interval))

    def load_bar_frame_any_range(self, vt_symbol: str, interval: str, *, include_derived: bool = True):
        return self.frames.get((vt_symbol, interval))

    def load_or_aggregate_bar_frame(self, *args, **kwargs):  # pragma: no cover - must never be called
        self.aggregate_called = True
        raise AssertionError("profiling must stay read-only")


def _bars(symbol: str = "600030.SSE") -> pl.DataFrame:
    base = datetime(2024, 1, 1, 9, 30)
    rows = []
    for i in range(180):
        price = 10 + i * 0.01
        rows.append(
            {
                "datetime": base + timedelta(minutes=30 * i),
                "open": price,
                "high": price + 0.05,
                "low": price - 0.05,
                "close": price + 0.01,
                "volume": 1000 + i,
                "turnover": (1000 + i) * price,
                "open_interest": 0.0,
            }
        )
    return pl.DataFrame(rows)


def test_profiler_clips_future_data_and_persists_only_profile_path(tmp_path) -> None:
    df = _bars()
    as_of = df["datetime"][120]
    lab = FakeLab({("600030.SSE", "30m"): df})
    store = ProfileStore(tmp_path)

    profile = Profiler(lab, store=store).profile(
        vt_symbol="600030.SSE",
        interval="30m",
        as_of=as_of,
        lookback_days=10,
        with_suggestion=True,
        persist=True,
    )

    assert profile.available
    assert profile.input.effective_right_bound == as_of
    assert profile.input.effective_bar_count == 121
    assert profile.artifact_id == "600030.SSE__30m__20240103T213000"
    assert not lab.aggregate_called
    saved_ids = store.list_ids()
    assert len(saved_ids) == 1
    loaded = store.load(saved_ids[0])
    assert loaded.input.effective_bar_count == 121
    assert loaded.artifact_id == profile.artifact_id


def test_profiler_returns_structured_unavailable(tmp_path) -> None:
    lab = FakeLab({})
    profile = Profiler(lab, store=ProfileStore(tmp_path)).profile(
        vt_symbol="600030.SSE",
        interval="30m",
        as_of=datetime(2024, 1, 1),
        lookback_days=5,
    )

    assert not profile.available
    assert profile.overall_confidence == "insufficient"
    assert profile.unavailable_reason


@pytest.mark.asyncio
async def test_profiling_api_validates_to_400(monkeypatch) -> None:
    monkeypatch.setattr(alpha_api, "_check_alpha_installed", lambda: True)

    with pytest.raises(HTTPException) as exc:
        await alpha_api.create_symbol_profile(
            {"vt_symbol": "600030.SSE", "interval": "30m", "lookback_days": 5}
        )

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_profiling_api_returns_profile(monkeypatch, tmp_path) -> None:
    lab = FakeLab({("600030.SSE", "30m"): _bars()})
    monkeypatch.setattr(alpha_api, "_check_alpha_installed", lambda: True)
    monkeypatch.setattr(alpha_api, "_get_alpha_lab", lambda: lab)
    monkeypatch.setattr(alpha_api, "ProfileStore", lambda: ProfileStore(tmp_path))

    result = await alpha_api.create_symbol_profile(
        {
            "vt_symbol": "600030.SSE",
            "interval": "30m",
            "as_of": "2024-01-03T21:30:00",
            "lookback_days": 10,
            "with_suggestion": False,
            "persist": False,
        }
    )

    assert result.available
    assert result.suggestion is None


@pytest.mark.asyncio
async def test_profiling_api_lists_artifacts(monkeypatch, tmp_path) -> None:
    store = ProfileStore(tmp_path)
    lab = FakeLab({("600030.SSE", "30m"): _bars()})
    monkeypatch.setattr(alpha_api, "_check_alpha_installed", lambda: True)
    monkeypatch.setattr(alpha_api, "ProfileStore", lambda: ProfileStore(tmp_path))

    Profiler(lab, store=store).profile(
        vt_symbol="600030.SSE",
        interval="30m",
        as_of=datetime(2024, 1, 3, 21, 30),
        lookback_days=10,
        persist=True,
    )

    assert await alpha_api.list_symbol_profile_artifacts() == [
        "600030.SSE__30m__20240103T213000"
    ]
