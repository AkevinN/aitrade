"""
AlphaLab — factor research and market data persistence layer.

Storage layout (`AITRADE_HOME/alpha_lab/`，默认 `./.aitrade/alpha_lab/`):
    bars/<interval>/    — raw bar parquet files, such as d / 1m / 5m
    ticks/              — raw historical tick parquet files
    derived/<interval>/ — locally aggregated bar parquet files
    component/          — index constituent shelve files
    dataset/            — pickled AlphaDataset objects
    model/              — pickled AlphaModel objects
    signal/             — signal parquet files
    contract.json       — contract trading settings
"""

from __future__ import annotations

import io
import json
import os
import pickle
import shelve
import threading
import uuid
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import polars as pl
from dateutil import parser as dateutil_parser

from ..config import CSV_FIELD_MAPPING, CSV_REQUIRED_FIELDS
from .dataset import AlphaDataset, to_datetime
from .lab_utils import (  # noqa: F401  re-export 保持向后兼容
    _canonical_bar_interval,
    _display_bar_interval,
    _is_minute_interval,
    _interval_minutes,
    _parse_vt_symbol,
    _EXCHANGE_ALIASES,
    _PREFIX_EXCHANGES,
    normalize_vt_symbol,
    _append_lookup_key,
    _vt_symbol_lookup_keys,
    canonical_vt_symbol_from_stem,
    _datetime_preview_values,
    _normalize_bound,
)
from .logger import logger
from .model import AlphaModel

TICK_FIELD_MAPPING: dict[str, list[str]] = {
    "datetime": ["datetime", "date", "time", "ts", "timestamp", "成交时间", "时间"],
    "symbol": ["symbol", "code", "stock_code", "股票代码", "代码"],
    "exchange": ["exchange", "market", "board", "交易所"],
    "vt_symbol": ["vt_symbol", "vtsymbol"],
    "last_price": ["last_price", "price", "last", "最新价", "成交价"],
    "volume": ["volume", "vol", "成交量", "成交手"],
    "turnover": ["turnover", "amount", "成交额", "成交金额"],
    "bid_price_1": ["bid_price_1", "bid1", "买一价"],
    "ask_price_1": ["ask_price_1", "ask1", "卖一价"],
    "bid_volume_1": ["bid_volume_1", "bid_vol_1", "买一量"],
    "ask_volume_1": ["ask_volume_1", "ask_vol_1", "卖一量"],
}

BAR_PREVIEW_FIELDS: list[str] = [
    "datetime",
    "symbol",
    "vt_symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "open_interest",
]
TICK_PREVIEW_FIELDS: list[str] = [
    "datetime",
    "symbol",
    "vt_symbol",
    "last_price",
    "volume",
    "turnover",
    "bid_price_1",
    "ask_price_1",
    "bid_volume_1",
    "ask_volume_1",
]


# A 股交易所时区。所有本地存储统一为交易所本地裸时间（naive），
# 避免带时区与不带时区的时间混存导致 8 小时错位。
_EXCHANGE_TZ = ZoneInfo("Asia/Shanghai")


def _to_exchange_naive(dt: datetime | None) -> datetime | None:
    """将时间统一为交易所本地裸时间：带时区者先转 Asia/Shanghai 再去时区。"""
    if dt is not None and dt.tzinfo is not None:
        return dt.astimezone(_EXCHANGE_TZ).replace(tzinfo=None)
    return dt


# 模块级纯函数已迁移至 lab_utils.py（interval / vt_symbol / 日期边界辅助），
# 上方通过 re-export 保持 `from aitrade.alpha.lab import normalize_vt_symbol` 等向后兼容。


class BarData:
    """Standalone BarData dataclass (no vnpy dependency)."""

    def __init__(
        self,
        symbol: str,
        exchange: str,
        datetime,
        interval: str,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float = 0.0,
        turnover: float = 0.0,
        open_interest: float = 0.0,
    ) -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.datetime = datetime
        self.interval = _canonical_bar_interval(interval)
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        self.turnover = turnover
        self.open_interest = open_interest

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange}" if self.exchange else self.symbol


class TickData:
    """Standalone historical tick record."""

    def __init__(
        self,
        symbol: str,
        exchange: str,
        datetime,
        last_price: float,
        volume: float = 0.0,
        turnover: float = 0.0,
        bid_price_1: float = 0.0,
        ask_price_1: float = 0.0,
        bid_volume_1: float = 0.0,
        ask_volume_1: float = 0.0,
    ) -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.datetime = datetime
        self.last_price = last_price
        self.volume = volume
        self.turnover = turnover
        self.bid_price_1 = bid_price_1
        self.ask_price_1 = ask_price_1
        self.bid_volume_1 = bid_volume_1
        self.ask_volume_1 = ask_volume_1

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange}" if self.exchange else self.symbol


class AlphaLab:
    """Alpha Research Laboratory — manages research artifacts and local market data."""

    def __init__(self, lab_path: Path | str) -> None:
        self.lab_path: Path = Path(lab_path)
        self.bars_path: Path = self.lab_path / "bars"
        self.daily_path: Path = self.bars_path / "d"
        self.minute_path: Path = self.bars_path / "1m"
        self.ticks_path: Path = self.lab_path / "ticks"
        self.imports_path: Path = self.lab_path / "imports"
        self.derived_path: Path = self.lab_path / "derived"
        self.component_path: Path = self.lab_path / "component"
        self.dataset_path: Path = self.lab_path / "dataset"
        self.model_path: Path = self.lab_path / "model"
        self.signal_path: Path = self.lab_path / "signal"
        self.contract_path: Path = self.lab_path / "contract.json"

        # Legacy folders retained for backwards-compatible reads.
        self.legacy_daily_path: Path = self.lab_path / "daily"
        self.legacy_minute_path: Path = self.lab_path / "minute"

        for path in [
            self.lab_path,
            self.bars_path,
            self.daily_path,
            self.minute_path,
            self.ticks_path,
            self.imports_path,
            self.derived_path,
            self.component_path,
            self.dataset_path,
            self.model_path,
            self.signal_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # 进程内按文件路径加锁，避免同一资源的并发读改写竞态（download/import/aggregate）。
        self._file_locks: dict[str, threading.Lock] = {}
        self._file_locks_guard = threading.Lock()

    def _lock_for(self, path: Path) -> threading.Lock:
        key = str(path)
        with self._file_locks_guard:
            lock = self._file_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._file_locks[key] = lock
            return lock

    @staticmethod
    def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
        """先写临时文件再原子替换，避免写入中途崩溃留下半截文件。"""
        tmp_path = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
        try:
            df.write_parquet(tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    # =========================================================================
    # Path and metadata helpers
    # =========================================================================

    def _bar_interval_path(self, interval: str, *, derived: bool = False) -> Path:
        canonical = _canonical_bar_interval(interval)
        base = self.derived_path if derived else self.bars_path
        return base / canonical

    def _bar_file_path(self, vt_symbol: str, interval: str, *, derived: bool = False) -> Path:
        folder = self._bar_interval_path(interval, derived=derived)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{vt_symbol}.parquet"

    def _bar_metadata_path(self, vt_symbol: str, interval: str, *, derived: bool = True) -> Path:
        folder = self._bar_interval_path(interval, derived=derived)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{vt_symbol}.meta.json"

    def _write_raw_bar_metadata(self, vt_symbol: str, interval: str, adjust_type: str) -> None:
        """记录原始 K 线的复权口径，供后续写入时校验一致性。"""
        payload = {
            "vt_symbol": normalize_vt_symbol(vt_symbol),
            "interval": _canonical_bar_interval(interval),
            "adjust_type": adjust_type,
            "updated_at": datetime.now().isoformat(),
        }
        with open(self._bar_metadata_path(vt_symbol, interval, derived=False), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_raw_bar_metadata(self, vt_symbol: str, interval: str) -> dict[str, Any]:
        metadata_path = self._bar_metadata_path(vt_symbol, interval, derived=False)
        if not metadata_path.exists():
            return {}
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)

    def _tick_file_path(self, vt_symbol: str) -> Path:
        return self.ticks_path / f"{vt_symbol}.parquet"

    def _legacy_bar_candidates(self, vt_symbol: str, interval: str) -> list[Path]:
        canonical = _canonical_bar_interval(interval)
        if canonical == "d":
            return [self.legacy_daily_path / f"{vt_symbol}.parquet"]
        if canonical == "1m":
            return [self.legacy_minute_path / f"{vt_symbol}.parquet"]
        return []

    def _iter_bar_candidates(
        self,
        vt_symbol: str,
        interval: str,
        *,
        include_derived: bool = True,
    ) -> list[Path]:
        canonical = _canonical_bar_interval(interval)
        candidates: list[Path] = []
        if include_derived:
            candidates.append(self._bar_file_path(vt_symbol, canonical, derived=True))
        candidates.append(self._bar_file_path(vt_symbol, canonical, derived=False))
        candidates.extend(self._legacy_bar_candidates(vt_symbol, canonical))

        ordered: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(candidate)
        return ordered

    def _scan_bar_files(
        self,
        vt_symbol: str,
        interval: str,
        *,
        include_derived: bool = True,
    ) -> list[Path]:
        """按规范证券代码扫描目录，匹配历史遗留文件名。"""
        canonical = _canonical_bar_interval(interval)
        target = normalize_vt_symbol(vt_symbol)
        folders: list[Path] = [self.bars_path / canonical]
        if include_derived:
            folders.append(self.derived_path / canonical)
        if canonical == "d":
            folders.append(self.legacy_daily_path)
        if canonical == "1m":
            folders.append(self.legacy_minute_path)

        matched: list[Path] = []
        seen: set[str] = set()
        for folder in folders:
            if not folder.exists():
                continue
            for file_path in sorted(folder.glob("*.parquet")):
                if canonical_vt_symbol_from_stem(file_path.stem) != target:
                    continue
                key = str(file_path)
                if key in seen:
                    continue
                seen.add(key)
                matched.append(file_path)
        return matched

    def _collect_bar_file_paths(
        self,
        vt_symbol: str,
        interval: str,
        *,
        include_derived: bool = True,
    ) -> list[Path]:
        """汇总精确路径与目录扫描得到的候选 parquet 文件。"""
        ordered: list[Path] = []
        seen: set[str] = set()
        for lookup_key in _vt_symbol_lookup_keys(vt_symbol):
            for candidate in self._iter_bar_candidates(
                lookup_key,
                interval,
                include_derived=include_derived,
            ):
                key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(candidate)
        for candidate in self._scan_bar_files(
            vt_symbol,
            interval,
            include_derived=include_derived,
        ):
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(candidate)
        return ordered

    def _load_frame_from_path(
        self,
        file_path: Path,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Optional[pl.DataFrame]:
        if not file_path.exists():
            return None

        df = pl.read_parquet(file_path)
        if "datetime" in df.columns:
            if start is not None:
                df = df.filter(pl.col("datetime") >= start)
            if end is not None:
                df = df.filter(pl.col("datetime") <= end)
        return df.sort("datetime")

    def _write_derived_metadata(
        self,
        vt_symbol: str,
        interval: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "vt_symbol": vt_symbol,
            "target_interval": _canonical_bar_interval(interval),
            "created_at": datetime.now().isoformat(),
        }
        if metadata:
            payload.update(metadata)
        with open(self._bar_metadata_path(vt_symbol, interval), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_derived_metadata(self, vt_symbol: str, interval: str) -> dict[str, Any]:
        metadata_path = self._bar_metadata_path(vt_symbol, interval)
        if not metadata_path.exists():
            return {}
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)

    # =========================================================================
    # Bar and tick persistence
    # =========================================================================

    def save_bar_frame(
        self,
        vt_symbol: str,
        interval: str,
        df: pl.DataFrame,
        *,
        derived: bool = False,
        metadata: dict[str, Any] | None = None,
        adjust_type: str | None = None,
    ) -> None:
        """Save normalized bar dataframe to parquet."""
        if df.is_empty():
            return

        vt_symbol = normalize_vt_symbol(vt_symbol)
        canonical = _canonical_bar_interval(interval)
        required_columns = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "open_interest",
        ]
        normalized = df
        for column in required_columns:
            if column not in normalized.columns:
                normalized = normalized.with_columns(
                    pl.lit(0.0 if column != "datetime" else None).alias(column)
                )
        normalized = normalized.select(required_columns).sort("datetime")

        file_path = self._bar_file_path(vt_symbol, canonical, derived=derived)
        with self._lock_for(file_path):
            if file_path.exists():
                # 拒绝把不同复权口径的数据静默拼进同一原始资源（除权日价格跳变会污染回测）。
                if not derived and adjust_type is not None:
                    existing_adjust = self._load_raw_bar_metadata(vt_symbol, canonical).get("adjust_type")
                    if existing_adjust and existing_adjust != adjust_type:
                        raise ValueError(
                            f"复权口径不一致：{vt_symbol}/{canonical} 已存在 {existing_adjust} 数据，"
                            f"本次为 {adjust_type}。请先删除该资源后再以统一口径重新下载。"
                        )
                old_df = pl.read_parquet(file_path)
                normalized = pl.concat([old_df, normalized]).unique(
                    subset=["datetime"],
                    keep="last",
                    maintain_order=True,
                )
                normalized = normalized.sort("datetime")

            self._atomic_write_parquet(normalized, file_path)
            if derived:
                self._write_derived_metadata(vt_symbol, canonical, metadata)
            elif adjust_type is not None:
                self._write_raw_bar_metadata(vt_symbol, canonical, adjust_type)

    def save_bar_data(
        self,
        bars: list[BarData],
        *,
        derived: bool = False,
        metadata: dict[str, Any] | None = None,
        adjust_type: str | None = None,
    ) -> None:
        """Save bar data to parquet."""
        if not bars:
            return

        data: list[dict[str, Any]] = []
        for bar in bars:
            data.append(
                {
                    "datetime": _to_exchange_naive(bar.datetime),
                    "open": bar.open_price,
                    "high": bar.high_price,
                    "low": bar.low_price,
                    "close": bar.close_price,
                    "volume": bar.volume,
                    "turnover": bar.turnover,
                    "open_interest": bar.open_interest,
                }
            )

        self.save_bar_frame(
            vt_symbol=bars[0].vt_symbol,
            interval=bars[0].interval,
            df=pl.DataFrame(data),
            derived=derived,
            metadata=metadata,
            adjust_type=adjust_type,
        )

    def save_bars_as_import_batch(
        self,
        bars: list[BarData],
        *,
        adjust_type: str = "none",
        source: str = "download",
        file_name: str | None = None,
    ) -> dict[str, Any]:
        """将一组 K 线写入「待合并批次」（imports 层），不直接进正式资源。

        供下载流程复用：下载到的数据先入批次，由用户在合并环节做连续性/一致性
        校验后再并入正式 K 线，避免脏数据或断档直接污染正式资源。
        """
        if not bars:
            raise ValueError("没有可写入批次的 K 线数据")
        return self._save_import_batch(
            data_kind="bar",
            vt_symbol=bars[0].vt_symbol,
            interval=bars[0].interval,
            records=bars,
            file_name=file_name,
            adjust_type=adjust_type,
            source=source,
        )

    def save_tick_frame(self, vt_symbol: str, df: pl.DataFrame) -> None:
        """Save normalized historical tick dataframe."""
        if df.is_empty():
            return

        vt_symbol = normalize_vt_symbol(vt_symbol)
        required_columns = [
            "datetime",
            "last_price",
            "volume",
            "turnover",
            "bid_price_1",
            "ask_price_1",
            "bid_volume_1",
            "ask_volume_1",
        ]
        normalized = df
        for column in required_columns:
            if column not in normalized.columns:
                normalized = normalized.with_columns(
                    pl.lit(0.0 if column != "datetime" else None).alias(column)
                )
        normalized = normalized.select(required_columns).sort("datetime")

        file_path = self._tick_file_path(vt_symbol)
        with self._lock_for(file_path):
            if file_path.exists():
                old_df = pl.read_parquet(file_path)
                normalized = pl.concat([old_df, normalized]).unique(
                    subset=["datetime"],
                    keep="last",
                    maintain_order=True,
                )
                normalized = normalized.sort("datetime")

            self._atomic_write_parquet(normalized, file_path)

    def save_tick_data(self, ticks: list[TickData]) -> None:
        """Save historical tick records."""
        if not ticks:
            return

        rows: list[dict[str, Any]] = []
        for tick in ticks:
            rows.append(
                {
                    "datetime": _to_exchange_naive(tick.datetime),
                    "last_price": tick.last_price,
                    "volume": tick.volume,
                    "turnover": tick.turnover,
                    "bid_price_1": tick.bid_price_1,
                    "ask_price_1": tick.ask_price_1,
                    "bid_volume_1": tick.bid_volume_1,
                    "ask_volume_1": tick.ask_volume_1,
                }
            )

        self.save_tick_frame(ticks[0].vt_symbol, pl.DataFrame(rows))

    def load_bar_frame(
        self,
        vt_symbol: str,
        interval: str,
        start,
        end,
        *,
        include_derived: bool = True,
    ) -> Optional[pl.DataFrame]:
        """Load stored bar dataframe by interval."""
        start_dt = _normalize_bound(start, is_end=False)
        end_dt = _normalize_bound(end, is_end=True)
        for candidate in self._collect_bar_file_paths(
            vt_symbol,
            interval,
            include_derived=include_derived,
        ):
            df = self._load_frame_from_path(candidate, start=start_dt, end=end_dt)
            if df is not None and not df.is_empty():
                return df
        return None

    def load_bar_frame_any_range(
        self,
        vt_symbol: str,
        interval: str,
        *,
        include_derived: bool = True,
    ) -> Optional[pl.DataFrame]:
        """Load stored bar dataframe without date filtering."""
        for candidate in self._collect_bar_file_paths(
            vt_symbol,
            interval,
            include_derived=include_derived,
        ):
            df = self._load_frame_from_path(candidate)
            if df is not None and not df.is_empty():
                return df
        return None

    def load_bar_data(
        self,
        vt_symbol: str,
        interval: str,
        start,
        end,
    ) -> list[BarData]:
        """Load bar data from parquet files."""
        df = self.load_bar_frame(vt_symbol, interval, start, end, include_derived=True)
        if df is None or df.is_empty():
            logger.error(f"Bar data {vt_symbol}/{interval} does not exist")
            return []

        canonical_vt_symbol = normalize_vt_symbol(vt_symbol)
        symbol, exchange = _parse_vt_symbol(canonical_vt_symbol)
        bars: list[BarData] = []
        for row in df.iter_rows(named=True):
            bars.append(
                BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=row["datetime"],
                    interval=interval,
                    open_price=row["open"],
                    high_price=row["high"],
                    low_price=row["low"],
                    close_price=row["close"],
                    volume=row.get("volume", 0.0),
                    turnover=row.get("turnover", 0.0),
                    open_interest=row.get("open_interest", 0.0),
                )
            )
        return bars

    def load_tick_frame(self, vt_symbol: str, start, end) -> Optional[pl.DataFrame]:
        """Load stored historical tick dataframe."""
        start_dt = _normalize_bound(start, is_end=False)
        end_dt = _normalize_bound(end, is_end=True)
        for lookup_key in _vt_symbol_lookup_keys(vt_symbol):
            df = self._load_frame_from_path(
                self._tick_file_path(lookup_key),
                start=start_dt,
                end=end_dt,
            )
            if df is not None and not df.is_empty():
                return df
        return None

    def load_tick_data(self, vt_symbol: str, start, end) -> list[TickData]:
        """Load historical tick data from parquet."""
        df = self.load_tick_frame(vt_symbol, start, end)
        if df is None or df.is_empty():
            return []

        canonical_vt_symbol = normalize_vt_symbol(vt_symbol)
        symbol, exchange = _parse_vt_symbol(canonical_vt_symbol)
        ticks: list[TickData] = []
        for row in df.iter_rows(named=True):
            ticks.append(
                TickData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=row["datetime"],
                    last_price=row["last_price"],
                    volume=row.get("volume", 0.0),
                    turnover=row.get("turnover", 0.0),
                    bid_price_1=row.get("bid_price_1", 0.0),
                    ask_price_1=row.get("ask_price_1", 0.0),
                    bid_volume_1=row.get("bid_volume_1", 0.0),
                    ask_volume_1=row.get("ask_volume_1", 0.0),
                )
            )
        return ticks

    def available_bar_intervals(self, vt_symbol: str, *, include_derived: bool = True) -> list[str]:
        """List stored intervals for one symbol."""
        intervals: set[str] = set()
        lookup_keys = _vt_symbol_lookup_keys(vt_symbol)
        for base in [self.bars_path, self.derived_path] if include_derived else [self.bars_path]:
            if not base.exists():
                continue
            for folder in base.iterdir():
                if not folder.is_dir():
                    continue
                for lookup_key in lookup_keys:
                    if (folder / f"{lookup_key}.parquet").exists():
                        intervals.add(folder.name)
                        break

        for lookup_key in lookup_keys:
            if (self.legacy_daily_path / f"{lookup_key}.parquet").exists():
                intervals.add("d")
            if (self.legacy_minute_path / f"{lookup_key}.parquet").exists():
                intervals.add("1m")
        return sorted(intervals, key=lambda item: (0 if item == "d" else 1 if item == "1m" else 2, item))

    def load_or_aggregate_bar_frame(
        self,
        vt_symbol: str,
        interval: str,
        start,
        end,
        *,
        input_data_kind: str = "bar",
        session_profile: str = "cn_equity",
    ) -> Optional[pl.DataFrame]:
        """
        Resolve bar input for training/preview.

        Priority:
        1. exact raw/derived bar file
        2. finer stored bars that can be aggregated locally
        3. historical ticks aggregated locally
        """
        canonical = _canonical_bar_interval(interval)
        direct = self.load_bar_frame(vt_symbol, canonical, start, end, include_derived=True)
        if direct is not None and not direct.is_empty():
            return direct

        if not _is_minute_interval(canonical):
            return direct

        if input_data_kind == "tick":
            tick_df = self.load_tick_frame(vt_symbol, start, end)
            if tick_df is not None and not tick_df.is_empty():
                aggregated = self.aggregate_tick_frame_to_bars(
                    tick_df,
                    target_interval=canonical,
                    session_profile=session_profile,
                )
                if not aggregated.is_empty():
                    self.save_bar_frame(
                        vt_symbol,
                        canonical,
                        aggregated,
                        derived=True,
                        metadata={
                            "source_kind": "tick",
                            "source_interval": "tick",
                            "target_interval": canonical,
                            "session_profile": session_profile,
                            "ts_convention": "end",
                        },
                    )
                    return aggregated
            return None

        target_minutes = _interval_minutes(canonical)
        finer_candidates: list[str] = []
        for available in self.available_bar_intervals(vt_symbol, include_derived=True):
            if not _is_minute_interval(available):
                continue
            available_minutes = _interval_minutes(available)
            if available_minutes < target_minutes and target_minutes % available_minutes == 0:
                finer_candidates.append(available)

        finer_candidates.sort(key=_interval_minutes)
        for source_interval in finer_candidates:
            source_df = self.load_bar_frame(vt_symbol, source_interval, start, end, include_derived=True)
            if source_df is None or source_df.is_empty():
                continue
            aggregated = self.aggregate_bar_frame(
                source_df,
                source_interval=source_interval,
                target_interval=canonical,
                session_profile=session_profile,
            )
            if not aggregated.is_empty():
                self.save_bar_frame(
                    vt_symbol,
                    canonical,
                    aggregated,
                    derived=True,
                    metadata={
                        "source_kind": "bar",
                        "source_interval": _canonical_bar_interval(source_interval),
                        "target_interval": canonical,
                        "session_profile": session_profile,
                        "ts_convention": "end",
                    },
                )
                return aggregated

        tick_df = self.load_tick_frame(vt_symbol, start, end)
        if tick_df is None or tick_df.is_empty():
            return None

        aggregated = self.aggregate_tick_frame_to_bars(
            tick_df,
            target_interval=canonical,
            session_profile=session_profile,
        )
        if aggregated.is_empty():
            return None

        self.save_bar_frame(
            vt_symbol,
            canonical,
            aggregated,
            derived=True,
            metadata={
                "source_kind": "tick",
                "source_interval": "tick",
                "target_interval": canonical,
                "session_profile": session_profile,
                "ts_convention": "end",
            },
        )
        return aggregated

    def load_bar_df(
        self,
        vt_symbols: list[str],
        interval: str,
        start,
        end,
        extended_days: int = 0,
    ) -> Optional[pl.DataFrame]:
        """Load bar data as a normalized polars DataFrame."""
        if not vt_symbols:
            return None

        start_dt = _normalize_bound(start, is_end=False) - timedelta(days=extended_days)
        end_dt = _normalize_bound(end, is_end=True) + timedelta(days=extended_days // 10)

        dfs: list[pl.DataFrame] = []
        for vt_symbol in vt_symbols:
            canonical_vt_symbol = normalize_vt_symbol(vt_symbol)
            df = self.load_bar_frame(canonical_vt_symbol, interval, start_dt, end_dt, include_derived=True)
            if df is None or df.is_empty():
                continue

            if "turnover" not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias("turnover"))
            if "open_interest" not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias("open_interest"))

            df = df.with_columns(
                pl.when(pl.col("volume") == 0)
                .then(None)
                .otherwise(pl.col("turnover").truediv(pl.col("volume")))
                .fill_null(pl.col("close"))
                .alias("vwap")
            )

            close_0: float = df.select(pl.col("close")).item(0, 0)
            close_base = max(float(close_0), 1e-12)
            df = df.with_columns(
                (pl.col("open") / close_base).alias("open"),
                (pl.col("high") / close_base).alias("high"),
                (pl.col("low") / close_base).alias("low"),
                (pl.col("close") / close_base).alias("close"),
            )

            numeric_columns: list[str] = [c for c in df.columns if c not in ["datetime", "vt_symbol"]]
            mask: pl.Series = df[numeric_columns].sum_horizontal() == 0
            df = df.with_columns(
                *[
                    pl.when(mask).then(float("nan")).otherwise(pl.col(c)).alias(c)
                    for c in numeric_columns
                ]
            )
            df = df.with_columns(pl.lit(canonical_vt_symbol).alias("vt_symbol"))
            dfs.append(df)

        if not dfs:
            return None
        return pl.concat(dfs).sort(["datetime", "vt_symbol"])

    # =========================================================================
    # Local aggregation
    # =========================================================================

    # A 股连续竞价时段（秒），按"区间结束时刻"约定聚合。
    _CN_EQUITY_SESSIONS: tuple[tuple[int, int], ...] = (
        (9 * 3600 + 30 * 60, 11 * 3600 + 30 * 60),  # 09:30:00 ~ 11:30:00
        (13 * 3600, 15 * 3600),  # 13:00:00 ~ 15:00:00
    )

    @staticmethod
    def _seconds_to_datetime(reference: datetime, seconds_of_day: int) -> datetime:
        """以 reference 的日期为基准，构造当日 seconds_of_day 对应的裸时间。"""
        hour, remainder = divmod(seconds_of_day, 3600)
        minute, second = divmod(remainder, 60)
        return reference.replace(hour=hour, minute=minute, second=second, microsecond=0)

    def _session_bucket_end(
        self,
        dt: datetime,
        interval_minutes: int,
        *,
        source_kind: str,
        session_profile: str = "cn_equity",
    ) -> datetime | None:
        """将时间戳映射到其所属聚合桶的「结束时刻」。

        输出 K 线时间统一为区间结束时刻（A 股惯例，与下载源一致）：
        - bar 来源：源 K 线本身按结束时刻标注，用 ceil 向上取整到目标边界
          （开盘快照 09:30 与首根分钟线一并落入首桶）；
        - tick 来源：事件时刻按 floor 落入 [start, end) 桶后以桶结束时刻标注；
        - 收盘/午盘整点（11:30 / 15:00）并入该 session 最后一个桶（右闭）。

        非交易时段返回 None（由调用方丢弃）。
        """
        normalized = _to_exchange_naive(dt).replace(microsecond=0)
        interval_sec = interval_minutes * 60
        t_sec = normalized.hour * 3600 + normalized.minute * 60 + normalized.second

        if session_profile != "cn_equity":
            if source_kind == "bar":
                steps = max(1, -(-t_sec // interval_sec))
            else:
                steps = t_sec // interval_sec + 1
            return self._seconds_to_datetime(normalized, steps * interval_sec)

        for open_sec, close_sec in self._CN_EQUITY_SESSIONS:
            if not (open_sec <= t_sec <= close_sec):
                continue
            if t_sec == close_sec:
                return self._seconds_to_datetime(normalized, close_sec)
            offset = t_sec - open_sec
            if source_kind == "bar":
                steps = max(1, -(-offset // interval_sec))  # ceil
            else:
                steps = offset // interval_sec + 1
            end_sec = min(open_sec + steps * interval_sec, close_sec)
            return self._seconds_to_datetime(normalized, end_sec)
        return None

    def _is_trailing_bucket_complete(
        self,
        bucket_end: datetime,
        bucket_rows: list[dict[str, Any]],
        *,
        source_kind: str,
        session_profile: str,
    ) -> bool:
        """判断末尾桶是否完整，防止把尚在形成的「半根 K 线」写入派生数据。

        - tick 来源：事件数据无固定成分根数，无法可靠判定完整性，一律保留，
          避免在数据正常结束于盘中时静默丢弃合法 K 线；
        - bar 来源：收于 session 收盘时刻（11:30 / 15:00）视为完整；否则要求
          桶内含有标注为该结束时刻的源 K 线（末分钟存在）才视为完整。
        """
        if source_kind != "bar":
            return True
        if session_profile == "cn_equity":
            t_sec = bucket_end.hour * 3600 + bucket_end.minute * 60 + bucket_end.second
            if any(t_sec == close_sec for _, close_sec in self._CN_EQUITY_SESSIONS):
                return True
        return any(item["datetime"] == bucket_end for item in bucket_rows)

    def _aggregate_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        target_interval: str,
        source_kind: str,
        session_profile: str = "cn_equity",
        stats: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame([])

        target_minutes = _interval_minutes(target_interval)

        def build_bucket(bucket_end: datetime, bucket_rows: list[dict[str, Any]]) -> dict[str, Any]:
            if source_kind == "tick":
                prices = [float(item["last_price"]) for item in bucket_rows]
                return {
                    "datetime": bucket_end,
                    "open": prices[0],
                    "high": max(prices),
                    "low": min(prices),
                    "close": prices[-1],
                    "volume": sum(float(item.get("volume", 0.0) or 0.0) for item in bucket_rows),
                    "turnover": sum(float(item.get("turnover", 0.0) or 0.0) for item in bucket_rows),
                    "open_interest": 0.0,
                }
            return {
                "datetime": bucket_end,
                "open": float(bucket_rows[0]["open"]),
                "high": max(float(item["high"]) for item in bucket_rows),
                "low": min(float(item["low"]) for item in bucket_rows),
                "close": float(bucket_rows[-1]["close"]),
                "volume": sum(float(item.get("volume", 0.0) or 0.0) for item in bucket_rows),
                "turnover": sum(float(item.get("turnover", 0.0) or 0.0) for item in bucket_rows),
                "open_interest": float(bucket_rows[-1].get("open_interest", 0.0) or 0.0),
            }

        # 先按桶结束时刻分组（输入已按 datetime 升序）。
        grouped: list[tuple[datetime, list[dict[str, Any]]]] = []
        current_bucket: datetime | None = None
        current_rows: list[dict[str, Any]] = []
        for row in rows:
            bucket_end = self._session_bucket_end(
                row["datetime"],
                target_minutes,
                source_kind=source_kind,
                session_profile=session_profile,
            )
            if bucket_end is None:
                continue
            if current_bucket is None:
                current_bucket = bucket_end
                current_rows = [row]
            elif bucket_end != current_bucket:
                grouped.append((current_bucket, current_rows))
                current_bucket = bucket_end
                current_rows = [row]
            else:
                current_rows.append(row)
        if current_bucket is not None:
            grouped.append((current_bucket, current_rows))

        # 仅丢弃全局末尾的不完整桶（最可能仍在形成），不影响盘中正常聚合。
        dropped_incomplete = 0
        if grouped and not self._is_trailing_bucket_complete(
            grouped[-1][0],
            grouped[-1][1],
            source_kind=source_kind,
            session_profile=session_profile,
        ):
            grouped.pop()
            dropped_incomplete = 1

        if stats is not None:
            stats["dropped_incomplete"] = dropped_incomplete
            stats["bucket_count"] = len(grouped)

        if not grouped:
            return pl.DataFrame([])
        records = [build_bucket(bucket_end, bucket_rows) for bucket_end, bucket_rows in grouped]
        return pl.DataFrame(records).sort("datetime")

    def aggregate_tick_frame_to_bars(
        self,
        tick_df: pl.DataFrame,
        *,
        target_interval: str,
        session_profile: str = "cn_equity",
        stats: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Aggregate historical ticks into minute bars."""
        rows = tick_df.sort("datetime").iter_rows(named=True)
        return self._aggregate_rows(
            list(rows),
            target_interval=_canonical_bar_interval(target_interval),
            source_kind="tick",
            session_profile=session_profile,
            stats=stats,
        )

    def aggregate_bar_frame(
        self,
        bar_df: pl.DataFrame,
        *,
        source_interval: str,
        target_interval: str,
        session_profile: str = "cn_equity",
        stats: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Aggregate stored minute bars into a coarser minute interval."""
        source_minutes = _interval_minutes(source_interval)
        target_minutes = _interval_minutes(target_interval)
        if target_minutes % source_minutes != 0:
            raise ValueError(
                f"目标周期 {target_interval} 不是来源周期 {source_interval} 的整数倍，无法本地聚合"
            )
        rows = bar_df.sort("datetime").iter_rows(named=True)
        return self._aggregate_rows(
            list(rows),
            target_interval=_canonical_bar_interval(target_interval),
            source_kind="bar",
            session_profile=session_profile,
            stats=stats,
        )

    def aggregate_market_data(
        self,
        vt_symbols: list[str],
        *,
        source_kind: str,
        source_interval: str | None,
        target_interval: str,
        start,
        end,
        session_profile: str = "cn_equity",
    ) -> dict[str, Any]:
        """Aggregate local raw data into derived bar resources."""
        start_dt = _normalize_bound(start, is_end=False)
        end_dt = _normalize_bound(end, is_end=True)
        canonical_target = _canonical_bar_interval(target_interval)
        if not _is_minute_interval(canonical_target):
            raise ValueError("当前仅支持聚合到分钟周期，例如 5m / 10m / 30m")

        total = len(vt_symbols)
        success = 0
        failed: list[str] = []

        for vt_symbol in vt_symbols:
            try:
                agg_stats: dict[str, Any] = {}
                if source_kind == "tick":
                    tick_df = self.load_tick_frame(vt_symbol, start_dt, end_dt)
                    if tick_df is None or tick_df.is_empty():
                        raise ValueError("缺少历史 Tick 数据")
                    aggregated = self.aggregate_tick_frame_to_bars(
                        tick_df,
                        target_interval=canonical_target,
                        session_profile=session_profile,
                        stats=agg_stats,
                    )
                else:
                    canonical_source = _canonical_bar_interval(source_interval or "1m")
                    if not _is_minute_interval(canonical_source):
                        raise ValueError("Bar 聚合来源必须是分钟周期，例如 1m")
                    source_df = self.load_bar_frame(
                        vt_symbol,
                        canonical_source,
                        start_dt,
                        end_dt,
                        include_derived=True,
                    )
                    if source_df is None or source_df.is_empty():
                        raise ValueError(f"缺少 {canonical_source} 原始/派生K线数据")
                    aggregated = self.aggregate_bar_frame(
                        source_df,
                        source_interval=canonical_source,
                        target_interval=canonical_target,
                        session_profile=session_profile,
                        stats=agg_stats,
                    )

                if aggregated.is_empty():
                    raise ValueError("聚合后结果为空")

                self.save_bar_frame(
                    vt_symbol,
                    canonical_target,
                    aggregated,
                    derived=True,
                    metadata={
                        "source_kind": source_kind,
                        "source_interval": "tick" if source_kind == "tick" else _canonical_bar_interval(source_interval or "1m"),
                        "target_interval": canonical_target,
                        "ts_convention": "end",
                        "dropped_incomplete": agg_stats.get("dropped_incomplete", 0),
                        "session_profile": session_profile,
                    },
                )
                success += 1
            except Exception as exc:
                failed.append(f"{vt_symbol}: {exc}")

        return {
            "total": total,
            "success": success,
            "failed": len(failed),
            "failed_symbols": failed,
            "target_interval": canonical_target,
        }

    # =========================================================================
    # Resource discovery and previews
    # =========================================================================

    def _resource_key(self, kind: str, vt_symbol: str, interval: str = "") -> str:
        vt_symbol = normalize_vt_symbol(vt_symbol)
        if kind in {"raw_tick"}:
            return vt_symbol
        return f"{interval}__{vt_symbol}"

    def _batch_resource_key(
        self,
        *,
        data_kind: str,
        vt_symbol: str,
        interval: str,
        batch_id: str,
    ) -> str:
        vt_symbol = normalize_vt_symbol(vt_symbol)
        resource_kind = "raw_tick" if data_kind == "tick" else "raw_bar"
        canonical = "tick" if data_kind == "tick" else _canonical_bar_interval(interval)
        # 批次 key 必须带 batch_id；同合约同周期多次上传也要在前端 rowKey 下稳定区分。
        return f"batch__{resource_kind}__{canonical}__{vt_symbol}__{batch_id}"

    def _parse_batch_key(self, key: str) -> dict[str, str]:
        parts = key.split("__")
        if len(parts) != 5 or parts[0] != "batch":
            raise ValueError("批次 key 格式错误")
        resource_kind, interval, vt_symbol, batch_id = parts[1], parts[2], parts[3], parts[4]
        if resource_kind not in {"raw_bar", "raw_tick"}:
            raise ValueError("批次资源类型错误")
        data_kind = "tick" if resource_kind == "raw_tick" else "bar"
        canonical = "tick" if data_kind == "tick" else _canonical_bar_interval(interval)
        vt_symbol = normalize_vt_symbol(vt_symbol)
        return {
            "data_kind": data_kind,
            "resource_kind": resource_kind,
            "interval": canonical,
            "vt_symbol": vt_symbol,
            "batch_id": batch_id,
        }

    def _batch_dir(self, data_kind: str, interval: str, vt_symbol: str) -> Path:
        vt_symbol = normalize_vt_symbol(vt_symbol)
        canonical = "tick" if data_kind == "tick" else _canonical_bar_interval(interval)
        return self.imports_path / data_kind / canonical / vt_symbol

    def _batch_file_path(self, data_kind: str, interval: str, vt_symbol: str, batch_id: str) -> Path:
        return self._batch_dir(data_kind, interval, vt_symbol) / f"{batch_id}.parquet"

    def _batch_metadata_path(self, data_kind: str, interval: str, vt_symbol: str, batch_id: str) -> Path:
        return self._batch_dir(data_kind, interval, vt_symbol) / f"{batch_id}.meta.json"

    def _write_batch_metadata(
        self,
        *,
        data_kind: str,
        vt_symbol: str,
        interval: str,
        batch_id: str,
        file_name: str | None,
        df: pl.DataFrame,
        status: str = "pending",
        adjust_type: str = "none",
        source: str = "upload",
    ) -> dict[str, Any]:
        vt_symbol = normalize_vt_symbol(vt_symbol)
        canonical = "tick" if data_kind == "tick" else _canonical_bar_interval(interval)
        metadata = {
            "batch_id": batch_id,
            "data_kind": data_kind,
            "kind": "raw_tick_batch" if data_kind == "tick" else "raw_bar_batch",
            "vt_symbol": vt_symbol,
            "interval": canonical,
            "file_name": file_name or "",
            "created_at": datetime.now().isoformat(),
            "row_count": len(df),
            "start": str(df["datetime"].min()) if "datetime" in df.columns and len(df) else "",
            "end": str(df["datetime"].max()) if "datetime" in df.columns and len(df) else "",
            "status": status,
            # 复权口径用于合并时校验不混用；source 区分上传/下载来源便于审计。
            "adjust_type": adjust_type,
            "source": source,
        }
        metadata_path = self._batch_metadata_path(data_kind, canonical, vt_symbol, batch_id)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return metadata

    def _load_batch_metadata(
        self,
        data_kind: str,
        interval: str,
        vt_symbol: str,
        batch_id: str,
    ) -> dict[str, Any]:
        metadata_path = self._batch_metadata_path(data_kind, interval, vt_symbol, batch_id)
        if not metadata_path.exists():
            return {}
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)

    def _batch_summary_from_file(
        self,
        *,
        data_kind: str,
        vt_symbol: str,
        interval: str,
        batch_id: str,
        file_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        vt_symbol = normalize_vt_symbol(vt_symbol)
        canonical = "tick" if data_kind == "tick" else _canonical_bar_interval(interval)
        resource_kind = "raw_tick_batch" if data_kind == "tick" else "raw_bar_batch"
        raw_kind = "raw_tick" if data_kind == "tick" else "raw_bar"
        meta = metadata or {}
        try:
            # 列表只需要区间与行数，按 datetime 单列读取可避免完整 parquet 扫描。
            df = pl.read_parquet(file_path, columns=["datetime"])
            row_count = len(df)
            start_value = df["datetime"].min()
            end_value = df["datetime"].max()
        except Exception:
            # 文件损坏或列缺失时仍尽量展示 meta，避免一个坏批次拖垮整个资源列表。
            row_count = int(meta.get("row_count", 0) or 0)
            start_value = meta.get("start", "")
            end_value = meta.get("end", "")

        return {
            "key": self._batch_resource_key(
                data_kind=data_kind,
                vt_symbol=vt_symbol,
                interval=canonical,
                batch_id=batch_id,
            ),
            "kind": resource_kind,
            "vt_symbol": vt_symbol,
            "interval": canonical,
            "row_count": row_count,
            "start": str(start_value),
            "end": str(end_value),
            "file_size_kb": round(file_path.stat().st_size / 1024, 1),
            "source_kind": "tick" if data_kind == "tick" else "bar",
            "source_interval": canonical,
            "target_interval": canonical,
            "created_at": meta.get("created_at"),
            "status": meta.get("status", "pending"),
            "batch_id": batch_id,
            "file_name": meta.get("file_name", ""),
            "batch_resource_kind": raw_kind,
        }

    def _resource_summary_from_file(
        self,
        *,
        kind: str,
        vt_symbol: str,
        interval: str,
        file_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        vt_symbol = normalize_vt_symbol(vt_symbol)
        canonical = _canonical_bar_interval(interval) if kind != "raw_tick" else "tick"
        try:
            df = pl.read_parquet(file_path, columns=["datetime"])
            row_count = len(df)
            start_value = df["datetime"].min()
            end_value = df["datetime"].max()
        except Exception:
            row_count = 0
            start_value = ""
            end_value = ""

        payload = {
            "key": self._resource_key(kind, vt_symbol, canonical),
            "kind": kind,
            "vt_symbol": vt_symbol,
            "interval": canonical,
            "row_count": row_count,
            "start": str(start_value),
            "end": str(end_value),
            "file_size_kb": round(file_path.stat().st_size / 1024, 1),
            "source_kind": metadata.get("source_kind", "bar") if metadata else ("tick" if kind == "raw_tick" else "bar"),
            "source_interval": _canonical_bar_interval(metadata.get("source_interval", canonical)) if metadata else canonical,
            "target_interval": _canonical_bar_interval(metadata.get("target_interval", canonical)) if metadata else canonical,
            "created_at": metadata.get("created_at") if metadata else None,
        }
        if metadata:
            payload.update(
                {
                    "session_profile": metadata.get("session_profile", ""),
                }
            )
        return payload

    def list_data_resources(self) -> dict[str, Any]:
        """List raw bar, raw tick, and derived bar resources."""
        raw_bars: list[dict[str, Any]] = []
        raw_ticks: list[dict[str, Any]] = []
        raw_bar_batches: list[dict[str, Any]] = []
        raw_tick_batches: list[dict[str, Any]] = []
        derived_bars: list[dict[str, Any]] = []

        seen_raw: set[tuple[str, str]] = set()
        if self.bars_path.exists():
            for interval_dir in sorted(self.bars_path.iterdir()):
                if not interval_dir.is_dir():
                    continue
                for file_path in sorted(interval_dir.glob("*.parquet")):
                    canonical_vt = canonical_vt_symbol_from_stem(file_path.stem)
                    dedupe_key = (interval_dir.name, canonical_vt)
                    if dedupe_key in seen_raw:
                        continue
                    seen_raw.add(dedupe_key)
                    raw_bars.append(
                        self._resource_summary_from_file(
                            kind="raw_bar",
                            vt_symbol=canonical_vt,
                            interval=interval_dir.name,
                            file_path=file_path,
                        )
                    )

        for legacy_interval, folder in [("d", self.legacy_daily_path), ("1m", self.legacy_minute_path)]:
            if not folder.exists():
                continue
            for file_path in sorted(folder.glob("*.parquet")):
                canonical_vt = canonical_vt_symbol_from_stem(file_path.stem)
                dedupe_key = (legacy_interval, canonical_vt)
                if dedupe_key in seen_raw:
                    continue
                seen_raw.add(dedupe_key)
                raw_bars.append(
                    self._resource_summary_from_file(
                        kind="raw_bar",
                        vt_symbol=canonical_vt,
                        interval=legacy_interval,
                        file_path=file_path,
                    )
                )

        seen_ticks: set[str] = set()
        for file_path in sorted(self.ticks_path.glob("*.parquet")):
            canonical_vt = canonical_vt_symbol_from_stem(file_path.stem)
            if canonical_vt in seen_ticks:
                continue
            seen_ticks.add(canonical_vt)
            raw_ticks.append(
                self._resource_summary_from_file(
                    kind="raw_tick",
                    vt_symbol=canonical_vt,
                    interval="tick",
                    file_path=file_path,
                    metadata={"source_kind": "tick", "source_interval": "tick", "target_interval": "tick"},
                )
            )

        for data_kind, target in [("bar", raw_bar_batches), ("tick", raw_tick_batches)]:
            base = self.imports_path / data_kind
            if not base.exists():
                continue
            for file_path in sorted(base.glob("*/*/*.parquet")):
                batch_id = file_path.stem
                vt_symbol = file_path.parent.name
                interval = file_path.parent.parent.name
                metadata = self._load_batch_metadata(data_kind, interval, vt_symbol, batch_id)
                target.append(
                    self._batch_summary_from_file(
                        data_kind=data_kind,
                        vt_symbol=metadata.get("vt_symbol", vt_symbol),
                        interval=metadata.get("interval", interval),
                        batch_id=metadata.get("batch_id", batch_id),
                        file_path=file_path,
                        metadata=metadata,
                    )
                )

        if self.derived_path.exists():
            for interval_dir in sorted(self.derived_path.iterdir()):
                if not interval_dir.is_dir():
                    continue
                for file_path in sorted(interval_dir.glob("*.parquet")):
                    metadata = self._load_derived_metadata(file_path.stem, interval_dir.name)
                    derived_bars.append(
                        self._resource_summary_from_file(
                            kind="derived_bar",
                            vt_symbol=file_path.stem,
                            interval=interval_dir.name,
                            file_path=file_path,
                            metadata=metadata,
                        )
                    )

        return {
            "raw_bars": raw_bars,
            "raw_ticks": raw_ticks,
            "raw_bar_batches": raw_bar_batches,
            "raw_tick_batches": raw_tick_batches,
            "derived_bars": derived_bars,
            "raw_bar_intervals": sorted({item["interval"] for item in raw_bars}),
            "derived_intervals": sorted({item["interval"] for item in derived_bars}),
        }

    def _resolve_resource_file(self, kind: str, key: str) -> tuple[Path, str, str]:
        if kind == "raw_tick":
            vt_symbol = normalize_vt_symbol(key)
            seen: set[str] = set()
            for lookup_key in _vt_symbol_lookup_keys(vt_symbol):
                candidate = self._tick_file_path(lookup_key)
                path_key = str(candidate)
                if path_key in seen:
                    continue
                seen.add(path_key)
                if candidate.exists():
                    return candidate, "tick", vt_symbol
            for candidate in sorted(self.ticks_path.glob("*.parquet")):
                if canonical_vt_symbol_from_stem(candidate.stem) == vt_symbol:
                    return candidate, "tick", vt_symbol
            return self._tick_file_path(vt_symbol), "tick", vt_symbol

        if kind in {"raw_bar_batch", "raw_tick_batch"}:
            parsed = self._parse_batch_key(key)
            expected_kind = "tick" if kind == "raw_tick_batch" else "bar"
            if parsed["data_kind"] != expected_kind:
                raise ValueError("资源类型与批次 key 不匹配")
            return (
                self._batch_file_path(
                    parsed["data_kind"],
                    parsed["interval"],
                    parsed["vt_symbol"],
                    parsed["batch_id"],
                ),
                parsed["interval"],
                parsed["vt_symbol"],
            )

        if "__" not in key:
            raise ValueError("资源 key 格式错误")
        interval, vt_symbol = key.split("__", 1)
        canonical = _canonical_bar_interval(interval)
        vt_symbol = normalize_vt_symbol(vt_symbol)

        if kind == "derived_bar":
            return self._bar_file_path(vt_symbol, canonical, derived=True), canonical, vt_symbol
        if kind == "raw_bar":
            candidates = self._collect_bar_file_paths(
                vt_symbol,
                canonical,
                include_derived=False,
            )
            for candidate in candidates:
                if candidate.exists():
                    return candidate, canonical, vt_symbol
            fallback = self._bar_file_path(vt_symbol, canonical, derived=False)
            return fallback, canonical, vt_symbol

        raise ValueError(f"不支持的资源类型: {kind}")

    def get_data_resource_detail(
        self,
        kind: str,
        key: str,
        *,
        limit: int = 100,
        before: str | None = None,
    ) -> dict[str, Any]:
        """Preview a single stored resource."""
        file_path, interval, vt_symbol = self._resolve_resource_file(kind, key)
        if not file_path.exists():
            raise FileNotFoundError(f"资源不存在: {kind}/{key}")

        df = pl.read_parquet(file_path)

        before_dt: datetime | None = None
        if before:
            before_dt = _to_exchange_naive(datetime.fromisoformat(before.replace("Z", "+00:00")))

        filtered_df = df
        if before_dt is not None:
            filtered_df = filtered_df.filter(pl.col("datetime") < before_dt)
        filtered_df = filtered_df.sort("datetime")

        if limit > 0:
            has_more = len(filtered_df) > limit
            data_df = filtered_df.tail(limit)
        else:
            has_more = False
            data_df = filtered_df

        next_before: str | None = None
        if has_more and len(data_df) > 0:
            first_dt = data_df["datetime"][0]
            next_before = first_dt.isoformat() if hasattr(first_dt, "isoformat") else str(first_dt)

        preview: list[dict[str, Any]] = []
        for row in data_df.iter_rows(named=True):
            item: dict[str, Any] = {}
            for field, value in row.items():
                if hasattr(value, "isoformat"):
                    item[field] = value.isoformat()
                elif isinstance(value, (int, float)):
                    item[field] = float(value)
                else:
                    item[field] = value
            preview.append(item)

        metadata = self._load_derived_metadata(vt_symbol, interval) if kind == "derived_bar" else {}
        batch_meta: dict[str, Any] = {}
        if kind in {"raw_bar_batch", "raw_tick_batch"}:
            parsed = self._parse_batch_key(key)
            batch_meta = self._load_batch_metadata(
                parsed["data_kind"],
                parsed["interval"],
                parsed["vt_symbol"],
                parsed["batch_id"],
            )

        detail_source_kind = metadata.get("source_kind", "tick" if kind in {"raw_tick", "raw_tick_batch"} else "bar")
        detail_source_interval = metadata.get("source_interval", interval)
        detail_target_interval = metadata.get("target_interval", interval)
        if batch_meta:
            detail_source_kind = "tick" if kind == "raw_tick_batch" else "bar"
            detail_source_interval = batch_meta.get("interval", interval)
            detail_target_interval = batch_meta.get("interval", interval)

        return {
            "key": key,
            "kind": kind,
            "vt_symbol": vt_symbol,
            "interval": interval,
            "row_count": len(df),
            "start": str(df["datetime"].min()) if "datetime" in df.columns and len(df) else "",
            "end": str(df["datetime"].max()) if "datetime" in df.columns and len(df) else "",
            "columns": list(df.columns),
            "preview": preview,
            "loaded_count": len(data_df),
            "has_more": has_more,
            "next_before": next_before,
            "source_kind": detail_source_kind,
            "source_interval": detail_source_interval,
            "target_interval": detail_target_interval,
            "file_size_kb": round(file_path.stat().st_size / 1024, 1),
            "status": batch_meta.get("status"),
            "batch_id": batch_meta.get("batch_id"),
            "file_name": batch_meta.get("file_name"),
        }

    def relocate_raw_bar_interval(self, key: str, new_interval: str) -> dict[str, Any]:
        """将原始 K 线资源移动到正确的周期目录，用于更正错误标签。"""
        canonical_new = _canonical_bar_interval(new_interval)
        supported = {"d", "1m", "5m", "15m", "30m", "60m"}
        if canonical_new not in supported:
            raise ValueError(f"不支持的周期: {new_interval}")

        file_path, old_interval, vt_symbol = self._resolve_resource_file("raw_bar", key)
        if not file_path.exists():
            raise FileNotFoundError(f"资源不存在: raw_bar/{key}")

        canonical_old = _canonical_bar_interval(old_interval)
        if canonical_new == canonical_old:
            return {
                "success": True,
                "message": "周期未变化",
                "key": key,
                "interval": canonical_new,
                "vt_symbol": vt_symbol,
            }

        new_path = self._bar_file_path(vt_symbol, canonical_new, derived=False)
        ordered_paths = sorted([file_path, new_path], key=str)
        with self._lock_for(ordered_paths[0]), self._lock_for(ordered_paths[1]):
            current_resolved = file_path.resolve()
            for candidate in [new_path, *self._legacy_bar_candidates(vt_symbol, canonical_new)]:
                if candidate.exists() and candidate.resolve() != current_resolved:
                    raise ValueError(
                        f"目标周期 {canonical_new} 下已存在 {vt_symbol}，请先删除或合并后再更正"
                    )

            new_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.rename(new_path)

            # 同步迁移复权口径 sidecar，并记录本次仅改标签的来源周期。
            old_meta = self._load_raw_bar_metadata(vt_symbol, canonical_old)
            old_meta_path = self._bar_metadata_path(vt_symbol, canonical_old, derived=False)
            if old_meta_path.exists():
                old_meta_path.unlink(missing_ok=True)
            payload = {
                "vt_symbol": vt_symbol,
                "interval": canonical_new,
                "adjust_type": old_meta.get("adjust_type", "none"),
                "relocated_from": canonical_old,
                "updated_at": datetime.now().isoformat(),
            }
            with open(self._bar_metadata_path(vt_symbol, canonical_new, derived=False), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        new_key = self._resource_key("raw_bar", vt_symbol, canonical_new)
        return {
            "success": True,
            "message": f"已将 {vt_symbol} 从 {canonical_old} 更正为 {canonical_new}",
            "key": new_key,
            "interval": canonical_new,
            "vt_symbol": vt_symbol,
        }

    def delete_data_resource(self, kind: str, key: str) -> bool:
        """Delete a stored raw or derived data resource."""
        file_path, interval, vt_symbol = self._resolve_resource_file(kind, key)
        if not file_path.exists():
            return False
        file_path.unlink()
        if kind == "derived_bar":
            metadata_path = self._bar_metadata_path(vt_symbol, interval)
            if metadata_path.exists():
                metadata_path.unlink()
        if kind == "raw_bar":
            raw_meta_path = self._bar_metadata_path(vt_symbol, interval, derived=False)
            if raw_meta_path.exists():
                raw_meta_path.unlink()
        if kind in {"raw_bar_batch", "raw_tick_batch"}:
            parsed = self._parse_batch_key(key)
            metadata_path = self._batch_metadata_path(
                parsed["data_kind"],
                parsed["interval"],
                parsed["vt_symbol"],
                parsed["batch_id"],
            )
            if metadata_path.exists():
                metadata_path.unlink()
        return True

    def _batch_minutes_expected(self, prev_dt: datetime, curr_dt: datetime, interval: str) -> bool:
        """按 A 股日内时段校验分钟线相邻点；午休和跨日天然允许断开。"""
        minutes = _interval_minutes(interval)
        if curr_dt.date() != prev_dt.date():
            return True
        if curr_dt <= prev_dt:
            return False

        def session_id(dt: datetime) -> str | None:
            t = dt.time()
            if time(9, 30) <= t < time(11, 30):
                return "morning"
            if time(13, 0) <= t < time(15, 0):
                return "afternoon"
            return None

        prev_session = session_id(prev_dt)
        curr_session = session_id(curr_dt)
        if prev_session is None or curr_session is None:
            return True
        if prev_session != curr_session:
            return True
        return curr_dt - prev_dt == timedelta(minutes=minutes)

    def _validate_batch_frame(self, df: pl.DataFrame, *, data_kind: str, interval: str) -> list[str]:
        errors: list[str] = []
        if df.is_empty():
            return ["批次数据为空"]
        if "datetime" not in df.columns:
            return ["批次缺少 datetime 字段"]

        datetimes = df["datetime"].to_list()
        if any(value is None for value in datetimes):
            errors.append("批次存在空 datetime")
        if datetimes != sorted(datetimes):
            errors.append("批次 datetime 未按升序排列")
        unique_count = df.select(pl.col("datetime").n_unique()).item()
        if unique_count != len(df):
            errors.append("批次内部存在重复 datetime")

        # Tick 不强制固定频率；分钟 K 线只在同一交易小节内要求周期连续。
        if data_kind == "bar" and _is_minute_interval(interval):
            sorted_datetimes = sorted(datetimes)
            for prev_dt, curr_dt in zip(sorted_datetimes, sorted_datetimes[1:]):
                if not self._batch_minutes_expected(prev_dt, curr_dt, interval):
                    errors.append(
                        f"批次分钟线存在断档: {prev_dt.isoformat(sep=' ')} -> {curr_dt.isoformat(sep=' ')}"
                    )
                    break
        return errors

    def _load_official_raw_frame(
        self, kind: str, vt_symbol: str, interval: str
    ) -> Optional[pl.DataFrame]:
        """加载现有正式（raw）资源，作为合并基底；不含派生数据。无则返回 None。"""
        if not vt_symbol or not interval:
            return None
        if kind == "raw_tick":
            for lookup_key in _vt_symbol_lookup_keys(vt_symbol):
                df = self._load_frame_from_path(self._tick_file_path(lookup_key))
                if df is not None and not df.is_empty():
                    return df
            return None
        df = self.load_bar_frame_any_range(vt_symbol, interval, include_derived=False)
        return df if df is not None and not df.is_empty() else None

    def _build_merge_plan(self, kind: str, keys: list[str]) -> dict[str, Any]:
        """构建合并计划：现有正式 K 线作基底 + 选中批次，统一做重叠/一致/连续校验。

        规则（与产品约定一致）：
        - 参与方 = 现有正式资源（若存在）+ 选中批次；
        - 参与方 ≥ 2 时要求存在公共重叠区间（日线同样要求重叠），否则视为拼接而拒绝；
        - 重叠时间点的数据必须完全一致（OHLC/量额），不一致则拒绝（防止复权口径/源混用）；
        - 分钟线还要求合并结果在交易小节内无断档；
        - 仅 1 个批次且无正式资源时，允许直接晋级为正式（无需重叠）。
        """
        result: dict[str, Any] = {
            "can_merge": False,
            "reason": "",
            "errors": [],
            "kind": kind,
            "keys": keys,
            "vt_symbol": "",
            "interval": "",
            "intersection_start": "",
            "intersection_end": "",
            "conflict_count": 0,
            "estimated_rows": 0,
            "batch_count": 0,
            "has_official": False,
        }
        if kind not in {"raw_bar", "raw_tick"}:
            raise ValueError("kind 仅支持 raw_bar 或 raw_tick")
        if len(keys) < 1:
            result["reason"] = "至少需要选择一个待合并批次"
            result["errors"] = [result["reason"]]
            return result

        batches, errors = self._load_merge_batches(kind, keys)
        if len(batches) != len(keys):
            errors.append("部分批次无法读取")
        if not batches:
            result["reason"] = errors[0] if errors else "批次不可用"
            result["errors"] = errors
            return result

        vt_symbols = {item["vt_symbol"] for item in batches}
        intervals = {item["interval"] for item in batches}
        if len(vt_symbols) != 1:
            errors.append("只能合并同一合约的批次")
        if len(intervals) != 1:
            errors.append("只能合并同一周期的批次")
        vt_symbol = next(iter(vt_symbols)) if len(vt_symbols) == 1 else ""
        interval = next(iter(intervals)) if len(intervals) == 1 else ""
        result["vt_symbol"] = vt_symbol
        result["interval"] = interval
        result["batch_count"] = len(batches)

        # 复权口径一致性：批次之间 + 批次与现有正式资源都不能混用。
        adjust_types = {item["metadata"].get("adjust_type", "none") for item in batches}
        official_df = self._load_official_raw_frame(kind, vt_symbol, interval) if not errors else None
        has_official = official_df is not None
        result["has_official"] = has_official
        if kind == "raw_bar" and has_official:
            official_adjust = self._load_raw_bar_metadata(vt_symbol, interval).get("adjust_type")
            if official_adjust:
                adjust_types.add(official_adjust)
        if len(adjust_types) > 1:
            errors.append(f"复权口径不一致（{', '.join(sorted(adjust_types))}），拒绝合并")

        # 参与方：现有正式资源（order=0）+ 批次（按上传时间 order=1..N）。
        participants: list[pl.DataFrame] = []
        if has_official:
            participants.append(official_df)
        participants.extend(item["df"] for item in batches)

        ranges = [(df["datetime"].min(), df["datetime"].max()) for df in participants]
        intersection_start = max(start for start, _ in ranges)
        intersection_end = min(end for _, end in ranges)
        if len(participants) >= 2 and intersection_start > intersection_end:
            errors.append("数据无重叠，不能合并（合并要求与现有数据或批次之间存在重叠区间）")

        combined = pl.concat(
            [df.with_columns(pl.lit(order).alias("_batch_order")) for order, df in enumerate(participants)],
            how="vertical_relaxed",
        )
        value_columns = [col for col in combined.columns if col not in {"_batch_order", "datetime"}]
        conflict_count = 0
        for _, group in combined.group_by("datetime", maintain_order=True):
            if len(group) <= 1:
                continue
            if group.select(value_columns).unique().height > 1:
                conflict_count += 1
        if conflict_count > 0:
            errors.append(
                f"重叠区存在 {conflict_count} 个时间点数据不一致，拒绝合并"
                "（请确认复权口径/数据源一致后重试）"
            )

        merged_df = (
            combined
            # 参与方按 order 升序；keep="last" 仅在数据一致时生效（不一致已拦截）。
            .sort(["datetime", "_batch_order"])
            .unique(subset=["datetime"], keep="last", maintain_order=True)
            .drop("_batch_order")
            .sort("datetime")
        )

        # 分钟线：校验合并结果在交易小节内连续（满足「合并要校验是否连续」）。
        if kind == "raw_bar" and interval and _is_minute_interval(interval):
            merged_dts = merged_df["datetime"].to_list()
            for prev_dt, curr_dt in zip(merged_dts, merged_dts[1:]):
                if not self._batch_minutes_expected(prev_dt, curr_dt, interval):
                    errors.append(
                        f"合并后分钟线存在断档: {prev_dt.isoformat(sep=' ')} -> {curr_dt.isoformat(sep=' ')}"
                    )
                    break

        result.update(
            {
                "errors": errors,
                "can_merge": not errors,
                "reason": errors[0] if errors else "",
                "intersection_start": intersection_start.isoformat() if hasattr(intersection_start, "isoformat") else str(intersection_start),
                "intersection_end": intersection_end.isoformat() if hasattr(intersection_end, "isoformat") else str(intersection_end),
                "conflict_count": conflict_count,
                "estimated_rows": len(merged_df),
                "adjust_type": next(iter(adjust_types)) if len(adjust_types) == 1 else "none",
                "_batches": batches,
                "_merged_df": merged_df,
            }
        )
        return result

    def _load_merge_batches(self, kind: str, keys: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
        errors: list[str] = []
        batches: list[dict[str, Any]] = []
        expected_data_kind = "tick" if kind == "raw_tick" else "bar"
        for key in keys:
            try:
                parsed = self._parse_batch_key(key)
                if parsed["data_kind"] != expected_data_kind:
                    errors.append(f"{key}: 批次类型与请求 kind 不一致")
                    continue
                file_path = self._batch_file_path(
                    parsed["data_kind"],
                    parsed["interval"],
                    parsed["vt_symbol"],
                    parsed["batch_id"],
                )
                if not file_path.exists():
                    errors.append(f"{key}: 批次文件不存在")
                    continue
                metadata = self._load_batch_metadata(
                    parsed["data_kind"],
                    parsed["interval"],
                    parsed["vt_symbol"],
                    parsed["batch_id"],
                )
                if metadata.get("status") == "merged":
                    errors.append(f"{key}: 批次已合并")
                    continue
                df = pl.read_parquet(file_path)
                batch_errors = self._validate_batch_frame(
                    df,
                    data_kind=parsed["data_kind"],
                    interval=parsed["interval"],
                )
                errors.extend(f"{key}: {error}" for error in batch_errors)
                batches.append({**parsed, "key": key, "file_path": file_path, "metadata": metadata, "df": df})
            except Exception as exc:
                errors.append(f"{key}: {exc}")
        # 按上传时间升序排列，使后续合并的「后上传批次覆盖先上传批次」名副其实
        # （覆盖优先级取决于上传时间，而非前端勾选/传参顺序）。batch_id 以时间戳前缀，
        # 在 created_at 缺失时作为稳定回退键。
        batches.sort(key=lambda item: (item["metadata"].get("created_at", ""), item.get("batch_id", "")))
        return batches, errors

    def preview_merge_import_batches(self, *, kind: str, keys: list[str]) -> dict[str, Any]:
        """Validate batches against the official base and summarize without writing."""
        plan = self._build_merge_plan(kind, keys)
        # 预览结果不向外暴露内部缓存（df / 批次对象）。
        return {key: value for key, value in plan.items() if not key.startswith("_")}

    def merge_import_batches(self, *, kind: str, keys: list[str]) -> dict[str, Any]:
        """Merge validated batches (with existing official base) into the official raw resource."""
        plan = self._build_merge_plan(kind, keys)
        public = {key: value for key, value in plan.items() if not key.startswith("_")}
        if not plan.get("can_merge"):
            return {**public, "success": False}

        batches = plan["_batches"]
        merged_df = plan["_merged_df"]
        vt_symbol = plan["vt_symbol"]
        interval = plan["interval"]
        adjust_type = plan.get("adjust_type", "none")
        if kind == "raw_tick":
            self.save_tick_frame(vt_symbol, merged_df)
        else:
            self.save_bar_frame(
                vt_symbol, interval, merged_df, derived=False, adjust_type=adjust_type
            )

        merged_at = datetime.now().isoformat()
        for item in batches:
            metadata = dict(item["metadata"])
            # 合并后仅标记状态，不删除原始批次，方便审计和必要时追溯来源文件。
            metadata.update({"status": "merged", "merged_at": merged_at})
            metadata_path = self._batch_metadata_path(
                item["data_kind"],
                item["interval"],
                item["vt_symbol"],
                item["batch_id"],
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        base_hint = "（含现有正式数据作为基底）" if plan.get("has_official") else "（新建正式资源）"
        return {
            **public,
            "success": True,
            "message": f"已合并 {len(batches)} 个批次到 {vt_symbol}/{interval}{base_hint}",
            "row_count": len(merged_df),
            "start": str(merged_df["datetime"].min()) if len(merged_df) else "",
            "end": str(merged_df["datetime"].max()) if len(merged_df) else "",
        }

    # =========================================================================
    # Dataset / model / signal / component persistence
    # =========================================================================

    def save_component_data(
        self,
        index_symbol: str,
        index_components: dict[str, list[str]],
    ) -> None:
        """Save index component data to shelve."""
        file_path: Path = self.component_path / index_symbol
        with shelve.open(str(file_path)) as db:
            db.update(index_components)

    @lru_cache
    def load_component_data(
        self,
        index_symbol: str,
        start,
        end,
    ) -> dict:
        """Load index component data from shelve."""
        file_path: Path = self.component_path / index_symbol
        start = to_datetime(start)
        end = to_datetime(end)

        with shelve.open(str(file_path)) as db:
            keys: list[str] = sorted(db.keys())
            index_components: dict = {}
            for key in keys:
                dt = datetime.strptime(key, "%Y-%m-%d")
                if start <= dt <= end:
                    index_components[dt] = db[key]
            return index_components

    def load_component_symbols(
        self,
        index_symbol: str,
        start,
        end,
    ) -> list[str]:
        """Collect all component symbols for an index."""
        components: dict = self.load_component_data(index_symbol, start, end)
        symbols: set[str] = set()
        for syms in components.values():
            symbols.update(syms)
        return list(symbols)

    def save_dataset(self, name: str, dataset: AlphaDataset) -> None:
        """Save dataset to pickle."""
        file_path: Path = self.dataset_path / f"{name}.pkl"
        with open(file_path, mode="wb") as f:
            pickle.dump(dataset, f)

    def load_dataset(self, name: str) -> Optional[AlphaDataset]:
        """Load dataset from pickle."""
        file_path: Path = self.dataset_path / f"{name}.pkl"
        if not file_path.exists():
            logger.error(f"Dataset file {name} does not exist")
            return None
        with open(file_path, mode="rb") as f:
            return pickle.load(f)

    def remove_dataset(self, name: str) -> bool:
        """Delete a dataset."""
        file_path: Path = self.dataset_path / f"{name}.pkl"
        if not file_path.exists():
            return False
        file_path.unlink()
        return True

    def list_all_datasets(self) -> list[str]:
        """List all dataset names."""
        return [f.stem for f in self.dataset_path.glob("*.pkl")]

    def save_model(self, name: str, model: AlphaModel) -> None:
        """Save model to pickle."""
        file_path: Path = self.model_path / f"{name}.pkl"
        with open(file_path, mode="wb") as f:
            pickle.dump(model, f)

    def load_model(self, name: str):
        """Load model from pickle."""
        file_path: Path = self.model_path / f"{name}.pkl"
        if not file_path.exists():
            logger.error(f"Model file {name} does not exist")
            return None
        with open(file_path, mode="rb") as f:
            return pickle.load(f)

    def remove_model(self, name: str) -> bool:
        """Delete a model."""
        file_path: Path = self.model_path / f"{name}.pkl"
        if not file_path.exists():
            return False
        file_path.unlink()
        return True

    def list_all_models(self) -> list[str]:
        """List all model names."""
        return [f.stem for f in self.model_path.glob("*.pkl")]

    def save_signal(self, name: str, signal: pl.DataFrame) -> None:
        """Save signal to parquet."""
        file_path: Path = self.signal_path / f"{name}.parquet"
        signal.write_parquet(file_path)

    def load_signal(self, name: str) -> Optional[pl.DataFrame]:
        """Load signal from parquet."""
        file_path: Path = self.signal_path / f"{name}.parquet"
        if not file_path.exists():
            logger.error(f"Signal file {name} does not exist")
            return None
        return pl.read_parquet(file_path)

    def remove_signal(self, name: str) -> bool:
        """Delete a signal."""
        file_path: Path = self.signal_path / f"{name}.parquet"
        if not file_path.exists():
            return False
        file_path.unlink()
        return True

    def list_all_signals(self) -> list[str]:
        """List all signal names."""
        return [f.stem for f in self.signal_path.glob("*.parquet")]

    def add_contract_setting(
        self,
        vt_symbol: str,
        long_rate: float,
        short_rate: float,
        size: float,
        pricetick: float,
    ) -> None:
        """Add contract trading configuration."""
        vt_symbol = normalize_vt_symbol(vt_symbol)
        contracts: dict[str, Any] = {}
        if self.contract_path.exists():
            with open(self.contract_path, encoding="utf-8") as f:
                contracts = json.load(f)

        contracts[vt_symbol] = {
            "long_rate": long_rate,
            "short_rate": short_rate,
            "size": size,
            "pricetick": pricetick,
        }

        with open(self.contract_path, mode="w+", encoding="utf-8") as f:
            json.dump(contracts, f, indent=4, ensure_ascii=False)

    def load_contract_settings(self) -> dict:
        """Load all contract settings."""
        if not self.contract_path.exists():
            return {}
        with open(self.contract_path, encoding="utf-8") as f:
            return json.load(f)

    # =========================================================================
    # CSV import / preview
    # =========================================================================

    def parse_csv_mapping(
        self,
        columns: list[str],
        custom_mapping: dict[str, str] | None = None,
        *,
        data_kind: str = "bar",
    ) -> dict[str, str]:
        """Auto-match CSV columns to standard fields."""
        aliases = dict(CSV_FIELD_MAPPING)
        if data_kind == "tick":
            aliases.update(TICK_FIELD_MAPPING)

        matched: dict[str, str] = {}
        column_lower: dict[str, str] = {c.lower(): c for c in columns}
        for std_field, alias_list in aliases.items():
            for alias in alias_list:
                if alias.lower() in column_lower:
                    matched[std_field] = column_lower[alias.lower()]
                    break

        if custom_mapping:
            matched.update(custom_mapping)
        return matched

    def preview_csv(
        self,
        csv_content: bytes,
        custom_mapping: dict[str, str] | None = None,
        *,
        data_kind: str = "bar",
    ) -> dict[str, Any]:
        """
        Parse CSV and return preview information without saving.

        Returns:
            dict with columns, sample_rows, matched_fields, unmapped_columns,
            missing_required, total_rows, date_range, symbols
        """
        df: pl.DataFrame = pl.read_csv(io.BytesIO(csv_content), infer_schema_length=0)
        columns: list[str] = df.columns
        matched: dict[str, str] = self.parse_csv_mapping(
            columns,
            custom_mapping,
            data_kind=data_kind,
        )
        unmapped: list[str] = [c for c in columns if c not in matched.values()]

        required = set(CSV_REQUIRED_FIELDS if data_kind == "bar" else ["datetime", "last_price"])
        missing = [field for field in required if field not in matched]

        preview_fields = BAR_PREVIEW_FIELDS if data_kind == "bar" else TICK_PREVIEW_FIELDS
        sample_rows: list[dict[str, Any]] = []
        for row in df.head(5).iter_rows(named=True):
            mapped_row: dict[str, Any] = {}
            for field in preview_fields:
                csv_column = matched.get(field, "")
                if csv_column and csv_column in row:
                    value = row.get(csv_column)
                    if isinstance(value, (int, float)):
                        mapped_row[field] = round(float(value), 4)
                    else:
                        mapped_row[field] = value
                else:
                    mapped_row[field] = None
            sample_rows.append(mapped_row)

        datetime_column = matched.get("datetime")
        date_range = ("", "")
        if datetime_column and datetime_column in df.columns:
            date_range = _datetime_preview_values(df[datetime_column].to_list())

        symbols: list[str] = []
        vt_symbol_col = matched.get("vt_symbol")
        symbol_col = matched.get("symbol")
        exchange_col = matched.get("exchange")
        if vt_symbol_col and vt_symbol_col in df.columns:
            symbols = sorted(
                {
                    normalize_vt_symbol(str(value))
                    for value in df[vt_symbol_col].drop_nulls().unique().to_list()
                    if str(value)
                }
            )
        elif symbol_col and symbol_col in df.columns:
            symbol_values: set[str] = set()
            for row in df.select([c for c in [symbol_col, exchange_col] if c]).iter_rows(named=True):
                raw_symbol = row.get(symbol_col, "")
                symbol = str(raw_symbol).zfill(6) if isinstance(raw_symbol, int) else str(raw_symbol)
                exchange = str(row.get(exchange_col, "")) if exchange_col else ""
                if symbol:
                    symbol_values.add(normalize_vt_symbol(f"{symbol}.{exchange}" if exchange else symbol))
            symbols = sorted(symbol_values)

        return {
            "data_kind": data_kind,
            "columns": preview_fields,
            "sample_rows": sample_rows,
            "matched_fields": matched,
            "unmapped_columns": unmapped,
            "missing_required": missing,
            "total_rows": len(df),
            "date_range": date_range,
            "symbols": symbols,
        }

    def _bar_records_frame(self, bars: list[BarData]) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for bar in bars:
            rows.append(
                {
                    "datetime": _to_exchange_naive(bar.datetime),
                    "open": bar.open_price,
                    "high": bar.high_price,
                    "low": bar.low_price,
                    "close": bar.close_price,
                    "volume": bar.volume,
                    "turnover": bar.turnover,
                    "open_interest": bar.open_interest,
                }
            )
        return pl.DataFrame(rows)

    def _tick_records_frame(self, ticks: list[TickData]) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for tick in ticks:
            rows.append(
                {
                    "datetime": _to_exchange_naive(tick.datetime),
                    "last_price": tick.last_price,
                    "volume": tick.volume,
                    "turnover": tick.turnover,
                    "bid_price_1": tick.bid_price_1,
                    "ask_price_1": tick.ask_price_1,
                    "bid_volume_1": tick.bid_volume_1,
                    "ask_volume_1": tick.ask_volume_1,
                }
            )
        return pl.DataFrame(rows)

    def _save_import_batch(
        self,
        *,
        data_kind: str,
        vt_symbol: str,
        interval: str,
        records: list[BarData] | list[TickData],
        file_name: str | None = None,
        adjust_type: str = "none",
        source: str = "upload",
    ) -> dict[str, Any]:
        vt_symbol = normalize_vt_symbol(vt_symbol)
        batch_id = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4().hex[:8]}"
        canonical = "tick" if data_kind == "tick" else _canonical_bar_interval(interval)
        df = self._tick_records_frame(records) if data_kind == "tick" else self._bar_records_frame(records)
        # 上传/下载批次先落在 imports 层；只有用户手动合并后才进入 bars/ticks 正式数据层。
        file_path = self._batch_file_path(data_kind, canonical, vt_symbol, batch_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(file_path)
        metadata = self._write_batch_metadata(
            data_kind=data_kind,
            vt_symbol=vt_symbol,
            interval=canonical,
            batch_id=batch_id,
            file_name=file_name,
            df=df,
            status="pending",
            adjust_type=adjust_type,
            source=source,
        )
        return self._batch_summary_from_file(
            data_kind=data_kind,
            vt_symbol=vt_symbol,
            interval=canonical,
            batch_id=batch_id,
            file_path=file_path,
            metadata=metadata,
        )

    def import_csv(
        self,
        csv_content: bytes,
        *,
        data_kind: str = "bar",
        interval: str = "d",
        import_mode: str = "merge",
        save_mode: str = "batch",
        file_name: str | None = None,
        custom_mapping: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Import CSV data and save to parquet files."""
        df: pl.DataFrame = pl.read_csv(io.BytesIO(csv_content), infer_schema_length=0)
        matched = self.parse_csv_mapping(df.columns, custom_mapping, data_kind=data_kind)
        required = set(CSV_REQUIRED_FIELDS if data_kind == "bar" else ["datetime", "last_price"])
        missing = [field for field in required if field not in matched]
        if missing:
            return {
                "success": False,
                "message": f"Missing required fields: {missing}",
                "imported_count": 0,
                "skipped_count": 0,
                "errors": [f"Required fields not found: {', '.join(missing)}"],
            }

        vt_symbol_col = matched.get("vt_symbol")
        symbol_col = None if vt_symbol_col else matched.get("symbol")
        exchange_col = None if vt_symbol_col else matched.get("exchange")

        bar_groups: dict[str, list[BarData]] = {}
        tick_groups: dict[str, list[TickData]] = {}

        for row in df.iter_rows(named=True):
            try:
                dt = dateutil_parser.parse(str(row.get(matched["datetime"], "")))
            except Exception:
                continue

            if vt_symbol_col:
                vt_symbol = str(row.get(vt_symbol_col, "") or "")
            else:
                symbol_value = row.get(symbol_col, "") if symbol_col else ""
                symbol = str(symbol_value).zfill(6) if isinstance(symbol_value, int) else str(symbol_value)
                exchange = str(row.get(exchange_col, "")) if exchange_col else ""
                vt_symbol = f"{symbol}.{exchange}" if exchange else symbol

            vt_symbol = normalize_vt_symbol(vt_symbol)
            symbol, exchange = _parse_vt_symbol(vt_symbol)
            if not symbol:
                continue

            try:
                if data_kind == "tick":
                    tick = TickData(
                        symbol=symbol,
                        exchange=exchange,
                        datetime=dt,
                        last_price=float(row.get(matched["last_price"], 0) or 0),
                        volume=float(row.get(matched.get("volume", ""), 0) or 0),
                        turnover=float(row.get(matched.get("turnover", ""), 0) or 0),
                        bid_price_1=float(row.get(matched.get("bid_price_1", ""), 0) or 0),
                        ask_price_1=float(row.get(matched.get("ask_price_1", ""), 0) or 0),
                        bid_volume_1=float(row.get(matched.get("bid_volume_1", ""), 0) or 0),
                        ask_volume_1=float(row.get(matched.get("ask_volume_1", ""), 0) or 0),
                    )
                    tick_groups.setdefault(vt_symbol, []).append(tick)
                else:
                    bar = BarData(
                        symbol=symbol,
                        exchange=exchange,
                        datetime=dt,
                        interval=interval,
                        open_price=float(row.get(matched["open"], 0) or 0),
                        high_price=float(row.get(matched["high"], 0) or 0),
                        low_price=float(row.get(matched["low"], 0) or 0),
                        close_price=float(row.get(matched["close"], 0) or 0),
                        volume=float(row.get(matched.get("volume", ""), 0) or 0),
                        turnover=float(row.get(matched.get("turnover", ""), 0) or 0),
                        open_interest=float(row.get(matched.get("open_interest", ""), 0) or 0),
                    )
                    bar_groups.setdefault(vt_symbol, []).append(bar)
            except Exception:
                continue

        imported_count = 0
        skipped_count = 0
        errors: list[str] = []
        batches: list[dict[str, Any]] = []

        if data_kind == "tick":
            groups = tick_groups
        else:
            groups = bar_groups

        for vt_symbol, records in groups.items():
            try:
                if save_mode == "batch":
                    # 页面默认走 batch 模式，避免同名 CSV 导入时自动覆盖/去重导致批次丢失。
                    batch = self._save_import_batch(
                        data_kind=data_kind,
                        vt_symbol=vt_symbol,
                        interval="tick" if data_kind == "tick" else interval,
                        records=records,
                        file_name=file_name,
                        source="upload",
                    )
                    batches.append(batch)
                    imported_count += len(records)
                    continue

                if data_kind == "tick":
                    file_path = self._tick_file_path(vt_symbol)
                else:
                    file_path = self._bar_file_path(vt_symbol, interval, derived=False)

                if import_mode == "replace" and file_path.exists():
                    file_path.unlink()

                if data_kind == "tick":
                    self.save_tick_data(records)
                else:
                    self.save_bar_data(records)
                imported_count += len(records)
            except Exception as exc:
                errors.append(f"Failed to import {vt_symbol}: {exc}")
                skipped_count += len(records)

        record_type = "ticks" if data_kind == "tick" else "bars"
        return {
            "success": not errors or imported_count > 0,
            "message": (
                f"Saved {len(batches)} import batches with {imported_count} {record_type}"
                if save_mode == "batch"
                else f"Imported {imported_count} {record_type} for {len(groups)} symbols"
            ),
            "imported_count": imported_count,
            "skipped_count": skipped_count,
            "errors": errors,
            "batches": batches,
        }
