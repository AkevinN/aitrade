"""
AlphaLab — data persistence layer for factor research.

Stores bar data, datasets, models, and signals to disk.
Storage layout (~/.aitrade/alpha_lab/):
    daily/         — daily bar parquet files
    minute/       — minute bar parquet files
    component/    — index constituent shelve files
    dataset/      — pickled AlphaDataset objects
    model/         — pickled AlphaModel objects
    signal/       — signal parquet files
    contract.json  — contract trading settings
"""

import io
import json
import pickle
import re
import shelve
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import polars as pl
from dateutil import parser as dateutil_parser

from ..config import CSV_FIELD_MAPPING, CSV_REQUIRED_FIELDS
from .logger import logger
from .dataset import AlphaDataset, to_datetime
from .model import AlphaModel


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
        self.interval = interval
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        self.turnover = turnover
        self.open_interest = open_interest

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange}"


class AlphaLab:
    """Alpha Research Laboratory — manages all research artifacts."""

    def __init__(self, lab_path: Path | str) -> None:
        self.lab_path: Path = Path(lab_path)
        self.daily_path: Path = self.lab_path / "daily"
        self.minute_path: Path = self.lab_path / "minute"
        self.component_path: Path = self.lab_path / "component"
        self.dataset_path: Path = self.lab_path / "dataset"
        self.model_path: Path = self.lab_path / "model"
        self.signal_path: Path = self.lab_path / "signal"
        self.contract_path: Path = self.lab_path / "contract.json"

        for path in [
            self.lab_path,
            self.daily_path,
            self.minute_path,
            self.component_path,
            self.dataset_path,
            self.model_path,
            self.signal_path,
        ]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

    def save_bar_data(self, bars: list[BarData]) -> None:
        """Save bar data to parquet."""
        if not bars:
            return

        bar: BarData = bars[0]
        if bar.interval == "d":
            file_path: Path = self.daily_path / f"{bar.vt_symbol}.parquet"
        elif bar.interval == "m":
            file_path = self.minute_path / f"{bar.vt_symbol}.parquet"
        else:
            logger.error(f"Unsupported interval {bar.interval}")
            return

        data: list[dict] = []
        for bar in bars:
            data.append({
                "datetime": bar.datetime.replace(tzinfo=None),
                "open": bar.open_price,
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                "volume": bar.volume,
                "turnover": bar.turnover,
                "open_interest": bar.open_interest,
            })

        new_df: pl.DataFrame = pl.DataFrame(data)

        if file_path.exists():
            old_df: pl.DataFrame = pl.read_parquet(file_path)
            new_df = pl.concat([old_df, new_df])
            new_df = new_df.unique(subset=["datetime"])
            new_df = new_df.sort("datetime")

        new_df.write_parquet(file_path)

    def load_bar_data(
        self,
        vt_symbol: str,
        interval: str,
        start,
        end,
    ) -> list[BarData]:
        """Load bar data from parquet files."""
        start = to_datetime(start)
        end = to_datetime(end)

        if interval == "d":
            folder_path: Path = self.daily_path
        elif interval == "m":
            folder_path = self.minute_path
        else:
            logger.error(f"Unsupported interval {interval}")
            return []

        file_path: Path = folder_path / f"{vt_symbol}.parquet"
        if not file_path.exists():
            logger.error(f"File {file_path} does not exist")
            return []

        df: pl.DataFrame = pl.read_parquet(file_path)
        df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))

        bars: list[BarData] = []
        symbol, exchange = vt_symbol.rsplit(".", 1)

        for row in df.iter_rows(named=True):
            bars.append(BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=row["datetime"],
                interval=interval,
                open_price=row["open"],
                high_price=row["high"],
                low_price=row["low"],
                close_price=row["close"],
                volume=row["volume"],
                turnover=row.get("turnover", 0.0),
                open_interest=row.get("open_interest", 0.0),
            ))

        return bars

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

        start = to_datetime(start) - timedelta(days=extended_days)
        end = to_datetime(end) + timedelta(days=extended_days // 10)

        if interval == "d":
            folder_path: Path = self.daily_path
        elif interval == "m":
            folder_path = self.minute_path
        else:
            logger.error(f"Unsupported interval {interval}")
            return None

        dfs: list[pl.DataFrame] = []

        for vt_symbol in vt_symbols:
            file_path: Path = folder_path / f"{vt_symbol}.parquet"
            if not file_path.exists():
                continue

            df: pl.DataFrame = pl.read_parquet(file_path)
            df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))

            if df.is_empty():
                continue

            df = df.with_columns(
                pl.col("turnover").truediv(pl.col("volume").fill_nan(1e-12)).alias("vwap")
            )

            close_0: float = df.select(pl.col("close")).item(0, 0)
            df = df.with_columns(
                (pl.col("open") / close_0).alias("open"),
                (pl.col("high") / close_0).alias("high"),
                (pl.col("low") / close_0).alias("low"),
                (pl.col("close") / close_0).alias("close"),
            )

            numeric_columns: list[str] = [c for c in df.columns if c not in ["datetime", "vt_symbol"]]
            mask: pl.Series = df[numeric_columns].sum_horizontal() == 0
            df = df.with_columns(
                *[pl.when(mask).then(float("nan")).otherwise(pl.col(c)).alias(c) for c in numeric_columns]
            )

            df = df.with_columns(pl.lit(vt_symbol).alias("vt_symbol"))
            dfs.append(df)

        if not dfs:
            return None

        return pl.concat(dfs)

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
        symbols: set = set()
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

    def save_model(self, name: str, model) -> None:
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
        contracts: dict = {}
        if self.contract_path.exists():
            with open(self.contract_path, encoding="UTF-8") as f:
                contracts = json.load(f)

        contracts[vt_symbol] = {
            "long_rate": long_rate,
            "short_rate": short_rate,
            "size": size,
            "pricetick": pricetick,
        }

        with open(self.contract_path, mode="w+", encoding="UTF-8") as f:
            json.dump(contracts, f, indent=4, ensure_ascii=False)

    def load_contract_setttings(self) -> dict:
        """Load all contract settings."""
        if not self.contract_path.exists():
            return {}
        with open(self.contract_path, encoding="UTF-8") as f:
            return json.load(f)

    # =========================================================================
    # CSV Import methods
    # =========================================================================

    def parse_csv_mapping(
        self,
        columns: list[str],
        custom_mapping: dict[str, str] | None = None
    ) -> dict[str, str]:
        """
        Auto-match CSV columns to standard fields based on configured aliases.

        Returns a dict mapping standard field names -> CSV column names.
        e.g. {"datetime": "trade_date", "open": "open_price", ...}
        """
        matched: dict[str, str] = {}
        column_lower: dict[str, str] = {c.lower(): c for c in columns}

        for std_field, aliases in CSV_FIELD_MAPPING.items():
            for alias in aliases:
                if alias.lower() in column_lower:
                    matched[std_field] = column_lower[alias.lower()]
                    break

        if custom_mapping:
            matched.update(custom_mapping)

        return matched

    def preview_csv(
        self,
        csv_content: bytes,
        custom_mapping: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """
        Parse CSV and return preview information without saving.

        Returns:
            dict with columns, sample_rows, matched_fields, unmapped_columns,
            missing_required, total_rows, date_range, symbols
        """
        df: pl.DataFrame = pl.read_csv(io.BytesIO(csv_content), infer_schema_length=0)
        columns: list[str] = df.columns

        matched: dict[str, str] = self.parse_csv_mapping(columns, custom_mapping)

        unmapped: list[str] = [c for c in columns if c not in matched.values()]

        required: set[str] = {"datetime", "open", "high", "low", "close"}
        missing: list[str] = [f for f in required if f not in matched]

        # Build mapped sample rows with standard field names
        all_standard_fields = ["datetime", "symbol", "open", "high", "low", "close", "volume"]
        if matched.get("turnover"):
            all_standard_fields.append("turnover")
        if matched.get("open_interest"):
            all_standard_fields.append("open_interest")
        if matched.get("change_pct"):
            all_standard_fields.append("change_pct")
        if matched.get("amplitude"):
            all_standard_fields.append("amplitude")

        sample_rows: list[dict[str, Any]] = []
        for row in df.head(5).iter_rows(named=True):
            mapped_row: dict[str, Any] = {}
            for field in all_standard_fields:
                csv_col = matched.get(field, "")
                if csv_col and csv_col in row:
                    val = row.get(csv_col)
                    if isinstance(val, (int, float)):
                        mapped_row[field] = round(float(val), 4)
                    elif val is not None:
                        mapped_row[field] = val
                    else:
                        mapped_row[field] = None
                else:
                    mapped_row[field] = None
            sample_rows.append(mapped_row)

        return {
            "columns": all_standard_fields,
            "sample_rows": sample_rows,
            "matched_fields": matched,
            "unmapped_columns": unmapped,
            "missing_required": missing,
            "total_rows": total_rows,
            "date_range": date_range,
            "symbols": symbols,
        }

    def import_csv(
        self,
        csv_content: bytes,
        interval: str = "d",
        import_mode: str = "merge",
        custom_mapping: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Import CSV data and save to parquet files.

        Args:
            csv_content: Raw CSV file bytes
            interval: "d" for daily, "m" for minute
            import_mode: "merge" (append/update) or "replace" (delete first)
            custom_mapping: Optional custom field mapping overrides

        Returns:
            dict with success, message, imported_count, skipped_count, errors
        """
        df: pl.DataFrame = pl.read_csv(io.BytesIO(csv_content), infer_schema_length=0)
        columns: list[str] = df.columns

        matched: dict[str, str] = self.parse_csv_mapping(columns, custom_mapping)

        required: set[str] = {"datetime", "open", "high", "low", "close"}
        missing: list[str] = [f for f in required if f not in matched]
        if missing:
            return {
                "success": False,
                "message": f"Missing required fields: {missing}",
                "imported_count": 0,
                "skipped_count": 0,
                "errors": [f"Required fields not found: {', '.join(missing)}"],
            }

        vt_symbol_col: str | None = matched.get("vt_symbol")

        if vt_symbol_col:
            symbol_col: str | None = None
            exchange_col: str | None = None
        else:
            symbol_col = matched.get("symbol")
            exchange_col = matched.get("exchange")

        bars_by_vt_symbol: dict[str, list[BarData]] = {}

        for row in df.iter_rows(named=True):
            try:
                raw_dt: Any = row.get(matched.get("datetime", ""), "")
                try:
                    dt: datetime = dateutil_parser.parse(str(raw_dt))
                except Exception:
                    continue

                if vt_symbol_col:
                    vt_sym: str = str(row.get(vt_symbol_col, ""))
                    if "." in vt_sym:
                        symbol, exchange = vt_sym.rsplit(".", 1)
                    else:
                        symbol = vt_sym
                        exchange = ""
                else:
                    symbol_val = row.get(symbol_col, "") if symbol_col else ""
                    if isinstance(symbol_val, int):
                        symbol = str(symbol_val).zfill(6)
                    else:
                        symbol = str(symbol_val)
                    exchange = str(row.get(exchange_col, "")) if exchange_col else ""

                if not symbol:
                    continue

                vt_sym = f"{symbol}.{exchange}" if exchange else symbol

                bar: BarData = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=dt,
                    interval=interval,
                    open_price=float(row.get(matched["open"], 0)),
                    high_price=float(row.get(matched["high"], 0)),
                    low_price=float(row.get(matched["low"], 0)),
                    close_price=float(row.get(matched["close"], 0)),
                    volume=float(row.get(matched.get("volume", ""), 0) or 0),
                    turnover=float(row.get(matched.get("turnover", ""), 0) or 0),
                    open_interest=float(row.get(matched.get("open_interest", ""), 0) or 0),
                )

                if vt_sym not in bars_by_vt_symbol:
                    bars_by_vt_symbol[vt_sym] = []
                bars_by_vt_symbol[vt_sym].append(bar)

            except Exception as e:
                continue

        imported_count: int = 0
        skipped_count: int = 0
        errors: list[str] = []

        for vt_sym, bars in bars_by_vt_symbol.items():
            try:
                if interval == "d":
                    folder_path: Path = self.daily_path
                else:
                    folder_path = self.minute_path

                file_path: Path = folder_path / f"{vt_sym}.parquet"

                if import_mode == "replace" and file_path.exists():
                    file_path.unlink()

                self.save_bar_data(bars)
                imported_count += len(bars)

            except Exception as e:
                errors.append(f"Failed to import {vt_sym}: {str(e)}")
                skipped_count += len(bars)

        return {
            "success": True,
            "message": f"Imported {imported_count} bars for {len(bars_by_vt_symbol)} symbols",
            "imported_count": imported_count,
            "skipped_count": skipped_count,
            "errors": errors,
        }
