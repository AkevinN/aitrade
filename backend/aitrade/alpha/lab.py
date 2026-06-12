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
    """将时间统一为交易所本地裸时间。

    带时区的 datetime 先转换为 ``Asia/Shanghai`` 时区再去掉 tzinfo；
    已是裸时间或 ``None`` 时原样返回。
    用于写入 parquet 前统一格式，避免带时区与不带时区混存导致 8 小时错位。

    Args:
        dt: 任意带/不带时区的 datetime，或 ``None``。

    Returns:
        无时区的本地 datetime；输入为 ``None`` 时返回 ``None``。
    """
    if dt is not None and dt.tzinfo is not None:
        return dt.astimezone(_EXCHANGE_TZ).replace(tzinfo=None)
    return dt


# 模块级纯函数已迁移至 lab_utils.py（interval / vt_symbol / 日期边界辅助），
# 上方通过 re-export 保持 `from aitrade.alpha.lab import normalize_vt_symbol` 等向后兼容。


class BarData:
    """独立 K 线数据类（不依赖 vnpy）。

    持有单根 K 线的 OHLCV 数据及合约标识，供 AlphaLab 内部序列化/反序列化使用。
    ``interval`` 在构造时经 ``_canonical_bar_interval`` 规范化存储。
    """

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
        """初始化 K 线数据对象。

        Args:
            symbol: 合约代码，如 ``"000001"``。
            exchange: 交易所代码，如 ``"SZSE"``、``"SSE"``。
            datetime: K 线对应的时间戳（结束时刻）。
            interval: K 线周期，支持别名（``"daily"``/``"d"``/``"1m"`` 等），
                构造时自动规范化。
            open_price: 开盘价。
            high_price: 最高价。
            low_price: 最低价。
            close_price: 收盘价。
            volume: 成交量，默认 ``0.0``。
            turnover: 成交额，默认 ``0.0``。
            open_interest: 持仓量（期货适用），默认 ``0.0``。
        """
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
        """合成 ``symbol.EXCHANGE`` 格式的统一合约代码；无交易所时仅返回 symbol。"""
        return f"{self.symbol}.{self.exchange}" if self.exchange else self.symbol


class TickData:
    """独立历史 Tick 数据类（不依赖 vnpy）。

    持有单笔成交/快照的价量信息及合约标识，供历史 Tick 文件读写使用。
    """

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
        """初始化历史 Tick 数据对象。

        Args:
            symbol: 合约代码，如 ``"600519"``。
            exchange: 交易所代码，如 ``"SSE"``。
            datetime: Tick 时间戳。
            last_price: 最新成交价。
            volume: 累计成交量，默认 ``0.0``。
            turnover: 累计成交额，默认 ``0.0``。
            bid_price_1: 买一价，默认 ``0.0``。
            ask_price_1: 卖一价，默认 ``0.0``。
            bid_volume_1: 买一量，默认 ``0.0``。
            ask_volume_1: 卖一量，默认 ``0.0``。
        """
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
        """合成 ``symbol.EXCHANGE`` 格式的统一合约代码；无交易所时仅返回 symbol。"""
        return f"{self.symbol}.{self.exchange}" if self.exchange else self.symbol


class AlphaLab:
    """Alpha 因子研究实验室——本地行情数据与研究工件的管理中心。

    负责以下存储类型的读写与维护：

    - ``bars/<interval>/``：原始 K 线 parquet；
    - ``ticks/``：历史 Tick parquet；
    - ``derived/<interval>/``：本地聚合派生 K 线；
    - ``imports/<kind>/<interval>/<vt_symbol>/``：待合并的导入批次；
    - ``component/``：指数成分 shelve 文件；
    - ``dataset/``、``model/``、``signal/``：研究工件 pickle/parquet；
    - ``contract.json``：合约交易参数配置。

    所有写入操作对同一文件加锁，防止并发下载/导入产生竞态；
    K 线写入前做复权口径校验，拒绝不同口径的数据静默混入。
    """

    def __init__(self, lab_path: Path | str) -> None:
        """初始化 AlphaLab，创建所有必要的目录结构。

        Args:
            lab_path: 实验室根目录路径，可为字符串或 ``Path`` 对象。
                目录不存在时自动递归创建；典型路径如
                ``".aitrade/alpha_lab"`` 或绝对路径。
        """
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
        """按文件路径返回（或懒创建）对应的线程锁。

        同一进程内同一路径始终返回同一把锁，保证并发写入安全。
        使用 ``_file_locks_guard`` 做双重保护，支持多线程并发调用。

        Args:
            path: 目标文件路径。

        Returns:
            该路径对应的 ``threading.Lock`` 实例。
        """
        key = str(path)
        with self._file_locks_guard:
            lock = self._file_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._file_locks[key] = lock
            return lock

    @staticmethod
    def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
        """以原子方式将 DataFrame 写入 parquet 文件。

        先写入同目录的唯一临时文件，再用 ``os.replace`` 原子替换目标路径，
        即使进程中途崩溃也不会留下半截文件污染目录。
        临时文件在 ``finally`` 块中清理，即便写入失败也不残留。

        Args:
            df: 待写入的 polars DataFrame。
            path: 目标 parquet 文件路径；父目录必须已存在。
        """
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
        """返回指定周期的 K 线存储目录路径（不含文件名）。

        Args:
            interval: K 线周期，经 ``_canonical_bar_interval`` 规范化。
            derived: ``True`` 时指向 ``derived/<interval>/``，
                ``False``（默认）指向 ``bars/<interval>/``。

        Returns:
            对应的 ``Path`` 目录对象；目录不一定存在，由调用方负责创建。
        """
        canonical = _canonical_bar_interval(interval)
        base = self.derived_path if derived else self.bars_path
        return base / canonical

    def _bar_file_path(self, vt_symbol: str, interval: str, *, derived: bool = False) -> Path:
        """返回指定合约和周期的 K 线 parquet 文件路径，并确保父目录存在。

        Args:
            vt_symbol: 合约代码，用作文件名主体（含交易所后缀，如 ``"000001.SZSE"``）。
            interval: K 线周期，自动规范化。
            derived: ``True`` 时路径位于 ``derived/`` 层，否则位于 ``bars/`` 层。

        Returns:
            形如 ``<base>/<interval>/<vt_symbol>.parquet`` 的 ``Path`` 对象。
        """
        folder = self._bar_interval_path(interval, derived=derived)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{vt_symbol}.parquet"

    def _bar_metadata_path(self, vt_symbol: str, interval: str, *, derived: bool = True) -> Path:
        """返回 K 线 sidecar 元数据文件（``.meta.json``）的路径，并确保父目录存在。

        Args:
            vt_symbol: 合约代码。
            interval: K 线周期，自动规范化。
            derived: ``True``（默认）时指向派生层元数据；
                ``False`` 时指向原始层（用于记录复权口径）。

        Returns:
            形如 ``<base>/<interval>/<vt_symbol>.meta.json`` 的 ``Path`` 对象。
        """
        folder = self._bar_interval_path(interval, derived=derived)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{vt_symbol}.meta.json"

    def _write_raw_bar_metadata(self, vt_symbol: str, interval: str, adjust_type: str) -> None:
        """将原始 K 线的复权口径写入 sidecar 元数据文件。

        每次成功写入原始 K 线后调用，用于后续写入时校验口径一致性——
        若发现已有不同口径的元数据则拦截，防止复权价格错误混入同一 parquet。

        Args:
            vt_symbol: 合约代码。
            interval: K 线周期（规范化后写入）。
            adjust_type: 复权口径字符串，如 ``"none"``、``"hfq"``（后复权）、``"qfq"``（前复权）。
        """
        payload = {
            "vt_symbol": normalize_vt_symbol(vt_symbol),
            "interval": _canonical_bar_interval(interval),
            "adjust_type": adjust_type,
            "updated_at": datetime.now().isoformat(),
        }
        with open(self._bar_metadata_path(vt_symbol, interval, derived=False), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_raw_bar_metadata(self, vt_symbol: str, interval: str) -> dict[str, Any]:
        """加载原始 K 线 sidecar 元数据（复权口径等）。

        Args:
            vt_symbol: 合约代码。
            interval: K 线周期。

        Returns:
            元数据字典；文件不存在时返回空字典。
        """
        metadata_path = self._bar_metadata_path(vt_symbol, interval, derived=False)
        if not metadata_path.exists():
            return {}
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)

    def _tick_file_path(self, vt_symbol: str) -> Path:
        """返回指定合约的历史 Tick parquet 文件路径（不保证文件存在）。

        Args:
            vt_symbol: 合约代码。

        Returns:
            ``ticks/<vt_symbol>.parquet`` 的 ``Path`` 对象。
        """
        return self.ticks_path / f"{vt_symbol}.parquet"

    def _legacy_bar_candidates(self, vt_symbol: str, interval: str) -> list[Path]:
        """返回旧版目录（``daily/``/``minute/``）下可能存在的 K 线候选路径列表。

        用于向后兼容迁移前的存储布局：日线曾存放于 ``daily/``，
        分钟线曾存放于 ``minute/``。

        Args:
            vt_symbol: 合约代码。
            interval: K 线周期，支持别名。

        Returns:
            候选路径列表；非日线/分钟线周期返回空列表。
        """
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
        """按优先级枚举指定合约和周期的所有候选 K 线文件路径（不过滤是否存在）。

        顺序：派生层（derived）→ 原始层（bars）→ 旧版目录（legacy）。
        去重后保持顺序返回，供调用方按顺序探测文件是否存在。

        Args:
            vt_symbol: 合约代码（已规范化）。
            interval: K 线周期。
            include_derived: ``False`` 时跳过派生层候选。

        Returns:
            去重后的候选 ``Path`` 列表，顺序反映读取优先级。
        """
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
        """遍历目录，按规范证券代码匹配所有可能的遗留文件名。

        将扫描到的每个 ``.parquet`` 文件名经 ``canonical_vt_symbol_from_stem``
        还原为规范代码，与目标 ``vt_symbol`` 对比，匹配则纳入结果。
        用于补充 ``_iter_bar_candidates`` 无法覆盖的不规则历史命名。

        Args:
            vt_symbol: 目标合约代码（未规范化；内部自动规范化后比对）。
            interval: K 线周期。
            include_derived: ``False`` 时跳过 ``derived/`` 目录扫描。

        Returns:
            所有匹配文件路径的去重有序列表。
        """
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
        """汇总精确路径候选与目录扫描结果，返回去重后的 parquet 候选文件列表。

        综合两种定位策略：
        1. 对 ``_vt_symbol_lookup_keys`` 生成的每个代码变体，调用
           ``_iter_bar_candidates`` 按优先级枚举精确路径；
        2. 再用 ``_scan_bar_files`` 扫描目录补充无法被精确路径命中的遗留文件。

        Args:
            vt_symbol: 目标合约代码（任意格式均可）。
            interval: K 线周期。
            include_derived: ``False`` 时跳过派生层。

        Returns:
            去重后按定位优先级排列的候选 ``Path`` 列表；
            调用方应按顺序探测文件是否存在。
        """
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
        """从 parquet 文件加载 DataFrame，并按可选的时间范围过滤。

        文件不存在时不抛错，直接返回 ``None``，由调用方选择下一个候选文件。
        过滤使用闭区间 ``[start, end]``；缺少 ``datetime`` 列时跳过过滤直接返回全量数据。

        Args:
            file_path: 目标 parquet 文件路径。
            start: 起始时间（含），``None`` 表示不过滤下界。
            end: 结束时间（含），``None`` 表示不过滤上界。

        Returns:
            按 ``datetime`` 升序排列的 DataFrame；文件不存在时返回 ``None``。
        """
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
        """将派生 K 线的聚合元数据写入 sidecar JSON 文件。

        记录聚合来源（source_kind/source_interval/target_interval/session_profile 等），
        供前端资源详情接口展示；传入的 ``metadata`` 字典可覆盖/追加默认字段。

        Args:
            vt_symbol: 合约代码。
            interval: 目标 K 线周期（派生层）。
            metadata: 额外写入的字段字典，``None`` 时仅写入基础字段。
        """
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
        """加载派生 K 线的聚合元数据。

        Args:
            vt_symbol: 合约代码。
            interval: 目标 K 线周期（派生层）。

        Returns:
            元数据字典；文件不存在时返回空字典。
        """
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
        """将规范化后的 K 线 DataFrame 写入 parquet（增量去重合并）。

        写入前：
        1. 补齐缺失的必填列（以 ``0.0`` 填充），并按 ``datetime`` 排序；
        2. 若文件已存在，读取现有数据与新数据合并，以 ``datetime`` 为主键去重（保留最新）；
        3. 原始层（``derived=False``）且指定了 ``adjust_type`` 时，校验复权口径一致性；
           不一致则抛 ``ValueError``，避免不同口径数据静默混入。

        Args:
            vt_symbol: 合约代码（写入前自动规范化）。
            interval: K 线周期，自动规范化。
            df: 待写入的 polars DataFrame，至少含 ``datetime`` 列；
                其余必填列缺失时自动补 ``0.0``。
            derived: ``True`` 表示写派生层（``derived/``），否则写原始层（``bars/``）。
            metadata: 派生层专用元数据字典，``derived=True`` 时写入 sidecar JSON；
                原始层忽略此参数。
            adjust_type: 复权口径，如 ``"none"``、``"hfq"``。仅对原始层生效；
                ``None`` 表示不写入/校验口径。

        Raises:
            ValueError: 原始层已有数据口径与本次 ``adjust_type`` 不一致时抛出。
        """
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
        """将 BarData 列表转为 DataFrame 后调用 ``save_bar_frame`` 写入 parquet。

        所有 ``bars`` 必须来自同一合约和同一周期（取自第一根 K 线的 ``vt_symbol``/``interval``）。
        时间统一转换为交易所本地裸时间后写入。

        Args:
            bars: 待写入的 K 线列表；空列表时直接返回。
            derived: 同 ``save_bar_frame``，是否写派生层。
            metadata: 同 ``save_bar_frame``，派生层元数据。
            adjust_type: 同 ``save_bar_frame``，复权口径。
        """
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
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """将一组 K 线写入「待合并批次」（imports 层），不直接进正式资源。

        供下载流程复用：下载到的数据先入批次，由用户在合并环节做连续性/一致性
        校验后再并入正式 K 线，避免脏数据或断档直接污染正式资源。

        Args:
            bars: 待写入的 K 线列表；空列表时抛 ``ValueError``。
            adjust_type: 复权口径，默认 ``"none"``；写入批次元数据，用于合并时校验。
            source: 数据来源标识，如 ``"download"``/``"upload"``；仅写入元数据，不影响逻辑。
            file_name: 可选的原始文件名，记录在元数据中便于审计。
            extra_meta: 额外写入批次元数据的字段字典。

        Returns:
            批次摘要字典，结构同 ``_batch_summary_from_file`` 返回值。

        Raises:
            ValueError: ``bars`` 为空时抛出。
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
            extra_meta=extra_meta,
        )

    def save_tick_frame(self, vt_symbol: str, df: pl.DataFrame) -> None:
        """将规范化后的历史 Tick DataFrame 写入 parquet（增量去重合并）。

        补齐缺失的必填列、按 ``datetime`` 排序；若文件已存在，与现有数据合并，
        以 ``datetime`` 为主键去重（保留最新），再原子写回。

        Args:
            vt_symbol: 合约代码（写入前自动规范化）。
            df: 待写入的 polars DataFrame，至少含 ``datetime`` 列；
                其余必填列缺失时自动补 ``0.0``。空 DataFrame 直接返回。
        """
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
        """将 TickData 列表转为 DataFrame 后调用 ``save_tick_frame`` 写入 parquet。

        所有 ``ticks`` 必须来自同一合约（取自第一条 Tick 的 ``vt_symbol``）。
        时间统一转换为交易所本地裸时间后写入。

        Args:
            ticks: 待写入的历史 Tick 列表；空列表时直接返回。
        """
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
        """按时间范围加载指定合约和周期的 K 线 DataFrame。

        依次尝试 ``_collect_bar_file_paths`` 返回的候选文件，返回第一个非空结果。
        时间范围使用闭区间 ``[start, end]``。

        Args:
            vt_symbol: 合约代码，支持非规范格式。
            interval: K 线周期，支持别名。
            start: 起始时间，支持 ``date``、``datetime``、ISO 字符串等。
            end: 结束时间，支持同 ``start``。
            include_derived: ``False`` 时跳过派生层，默认 ``True``。

        Returns:
            按 ``datetime`` 升序排列的 DataFrame；无数据时返回 ``None``。
        """
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
        """不限时间范围加载指定合约和周期的全量 K 线 DataFrame。

        逻辑同 ``load_bar_frame`` 但不做时间过滤，通常用于合并计划或
        可用区间探测时读取完整数据。

        Args:
            vt_symbol: 合约代码。
            interval: K 线周期。
            include_derived: ``False`` 时跳过派生层，默认 ``True``。

        Returns:
            按 ``datetime`` 升序排列的 DataFrame；无数据时返回 ``None``。
        """
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
        """从 parquet 文件加载 K 线数据并返回 BarData 对象列表。

        内部调用 ``load_bar_frame`` 读取 DataFrame，再逐行构造 ``BarData``。
        数据不存在时记录错误日志并返回空列表（不抛异常）。

        Args:
            vt_symbol: 合约代码。
            interval: K 线周期。
            start: 起始时间（含）。
            end: 结束时间（含）。

        Returns:
            按时间升序排列的 ``BarData`` 列表；无数据时返回空列表。
        """
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
        """按时间范围加载指定合约的历史 Tick DataFrame。

        遍历代码变体对应的文件路径，返回第一个非空结果。

        Args:
            vt_symbol: 合约代码，支持非规范格式。
            start: 起始时间（含）。
            end: 结束时间（含）。

        Returns:
            按 ``datetime`` 升序排列的 DataFrame；无数据时返回 ``None``。
        """
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
        """从 parquet 文件加载历史 Tick 数据并返回 TickData 对象列表。

        Args:
            vt_symbol: 合约代码。
            start: 起始时间（含）。
            end: 结束时间（含）。

        Returns:
            按时间升序排列的 ``TickData`` 列表；无数据时返回空列表。
        """
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
        """列出本地已存储的指定合约所有 K 线周期。

        遍历 ``bars/`` 和（可选）``derived/`` 子目录，检测每个周期文件夹下是否存在
        匹配该合约代码（含变体）的 parquet 文件；同时检测旧版目录。

        Args:
            vt_symbol: 合约代码，支持非规范格式。
            include_derived: ``False`` 时跳过派生层，默认 ``True``。

        Returns:
            已存储周期的排序列表，日线 ``"d"`` 优先，分钟线次之，其余按字母序。
        """
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
        """加载或本地聚合 K 线，优先直接读文件，不存在时从更细粒度数据聚合。

        用于训练/预览时解析 K 线输入，按以下优先级尝试：

        1. 直接读取已存储的原始/派生 K 线文件（最快）；
        2. 仅分钟周期：从本地更细粒度分钟 K 线聚合（如 1m → 5m）；
        3. 从历史 Tick 聚合（最慢，仅在上述均无数据时作为兜底）。

        聚合结果自动写入派生层，下次直接命中步骤 1。

        Args:
            vt_symbol: 合约代码。
            interval: 目标 K 线周期。
            start: 起始时间（含）。
            end: 结束时间（含）。
            input_data_kind: ``"tick"`` 时强制从 Tick 聚合，跳过 K 线候选；
                默认 ``"bar"``。
            session_profile: 交易时段配置，目前仅支持 ``"cn_equity"``（A 股）。

        Returns:
            按 ``datetime`` 升序排列的 DataFrame；所有来源均无数据时返回 ``None``。
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
        """批量加载多只标的的 K 线并拼接为规范化宽表，供因子计算使用。

        对每只标的执行以下处理：
        1. 加载指定时间范围（可按 ``extended_days`` 向外延伸）的 K 线；
        2. 补齐缺失的 ``turnover``/``open_interest`` 列；
        3. 计算成交量加权均价 ``vwap``（成交量为 0 时 fallback 到 ``close``）；
        4. 以第一个收盘价为基准做相对价格归一化（open/high/low/close 均除以 close_0）；
        5. 将数值全为 0 的行替换为 NaN（标记无效/停牌行情）。

        Args:
            vt_symbols: 合约代码列表，如 ``["000001.SZSE", "600000.SSE"]``。
            interval: K 线周期。
            start: 起始日期（含）。
            end: 结束日期（含）。
            extended_days: 向左延伸的天数（用于因子计算的 lookback），默认 ``0``；
                右侧额外延伸 ``extended_days // 10`` 天。

        Returns:
            以 ``[datetime, vt_symbol]`` 排序的 polars DataFrame，
            含 ``vt_symbol`` 标识列；所有标的均无数据时返回 ``None``。
        """
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
        """以 reference 的日期为基准，将秒数转换为当日对应的裸时间。

        Args:
            reference: 基准 datetime，仅取其 ``date`` 部分。
            seconds_of_day: 当日秒数，如 ``9*3600+30*60`` 表示 09:30:00。

        Returns:
            与 reference 同日、时分秒对应 ``seconds_of_day`` 的裸 datetime。
        """
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
        """将已排序的行记录列表聚合为目标周期的 K 线 DataFrame。

        核心聚合逻辑：
        1. 调用 ``_session_bucket_end`` 将每行归入对应桶；
        2. 非交易时段（返回 ``None``）的行被丢弃；
        3. 检测并可选丢弃末尾不完整桶（防止写入半根 K 线）；
        4. 每个桶聚合为一根 K 线（OHLCV，bar 来源保留 open_interest，tick 来源置 0）。

        Args:
            rows: 已按 ``datetime`` 升序排列的行记录字典列表。
                bar 来源含 ``open/high/low/close/volume/turnover/open_interest``；
                tick 来源含 ``last_price/volume/turnover``。
            target_interval: 目标聚合周期，必须为分钟周期（``"5m"``/``"30m"`` 等）。
            source_kind: ``"bar"`` 或 ``"tick"``，决定桶分配和聚合方式。
            session_profile: 交易时段配置，``"cn_equity"``（默认）使用 A 股双时段。
            stats: 若传入可变字典，聚合后写入 ``dropped_incomplete``/``bucket_count`` 统计。

        Returns:
            聚合后按 ``datetime`` 升序排列的 polars DataFrame；输入为空返回空 DataFrame。
        """
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
        """将历史 Tick DataFrame 聚合为指定分钟周期的 K 线 DataFrame。

        以 Tick 时刻对应的桶结束时刻作为 K 线时间戳（A 股区间结束时刻约定）；
        每桶首 Tick 为开盘价，末 Tick 为收盘价，成交量/额累加，open_interest 置 0。

        Args:
            tick_df: 历史 Tick DataFrame，至少含 ``datetime``/``last_price`` 列；
                内部按 ``datetime`` 升序排序后处理。
            target_interval: 目标 K 线周期，必须为分钟周期（如 ``"5m"``、``"30m"``）。
            session_profile: 交易时段配置，默认 ``"cn_equity"``（A 股）。
            stats: 若传入可变字典，写入聚合统计（``dropped_incomplete``/``bucket_count``）。

        Returns:
            聚合后按 ``datetime`` 升序排列的 K 线 DataFrame；无有效桶时返回空 DataFrame。
        """
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
        """将细粒度分钟 K 线 DataFrame 聚合为较粗周期。

        要求 ``target_interval`` 为 ``source_interval`` 的整数倍（如 1m → 5m/30m）；
        K 线聚合遵循 A 股区间结束时刻约定（与下载源一致）。

        Args:
            bar_df: 来源 K 线 DataFrame，至少含 ``datetime``/``open``/``high``/``low``/
                ``close`` 列；内部按 ``datetime`` 升序排序后处理。
            source_interval: 来源 K 线周期，必须为分钟周期（如 ``"1m"``）。
            target_interval: 目标 K 线周期，必须为来源周期的整数倍（如 ``"5m"``）。
            session_profile: 交易时段配置，默认 ``"cn_equity"``。
            stats: 若传入可变字典，写入聚合统计（``dropped_incomplete``/``bucket_count``）。

        Returns:
            聚合后按 ``datetime`` 升序排列的 K 线 DataFrame；无有效桶时返回空 DataFrame。

        Raises:
            ValueError: ``target_interval`` 不是 ``source_interval`` 整数倍时抛出。
        """
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
        """批量将本地原始数据聚合为派生 K 线资源（写入 ``derived/`` 层）。

        逐合约执行聚合，成功写入后返回摘要统计；单个合约失败不影响其余合约。
        仅支持聚合到分钟周期目标。

        Args:
            vt_symbols: 待聚合的合约代码列表。
            source_kind: 来源数据类型，``"tick"`` 或 ``"bar"``。
            source_interval: 来源 K 线周期（``source_kind="bar"`` 时使用，
                如 ``"1m"``）；``source_kind="tick"`` 时忽略。
            target_interval: 目标 K 线周期，必须为分钟周期（如 ``"5m"``、``"30m"``）。
            start: 起始时间（含）。
            end: 结束时间（含）。
            session_profile: 交易时段配置，默认 ``"cn_equity"``。

        Returns:
            包含 ``total``/``success``/``failed``/``failed_symbols``/``target_interval`` 的统计字典。

        Raises:
            ValueError: ``target_interval`` 不是分钟周期，或 ``source_interval`` 非分钟周期时抛出。
        """
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
        """生成正式资源的稳定唯一键，用作前端列表的 rowKey。

        Tick 类型键格式为 ``<vt_symbol>``；
        K 线类型键格式为 ``<interval>__<vt_symbol>``。

        Args:
            kind: 资源类型，如 ``"raw_bar"``/``"derived_bar"``/``"raw_tick"``。
            vt_symbol: 合约代码（内部自动规范化）。
            interval: K 线周期；Tick 资源时可忽略。

        Returns:
            唯一资源键字符串。
        """
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
        """生成导入批次的稳定唯一键（含 batch_id 以区分同合约多次上传）。

        格式：``batch__<resource_kind>__<interval>__<vt_symbol>__<batch_id>``，
        可由 ``_parse_batch_key`` 逆解析。

        Args:
            data_kind: ``"bar"`` 或 ``"tick"``。
            vt_symbol: 合约代码（内部自动规范化）。
            interval: K 线/Tick 周期（Tick 固定为 ``"tick"``）。
            batch_id: 批次唯一 ID（含时间戳前缀）。

        Returns:
            批次唯一键字符串。
        """
        vt_symbol = normalize_vt_symbol(vt_symbol)
        resource_kind = "raw_tick" if data_kind == "tick" else "raw_bar"
        canonical = "tick" if data_kind == "tick" else _canonical_bar_interval(interval)
        # 批次 key 必须带 batch_id；同合约同周期多次上传也要在前端 rowKey 下稳定区分。
        return f"batch__{resource_kind}__{canonical}__{vt_symbol}__{batch_id}"

    def _parse_batch_key(self, key: str) -> dict[str, str]:
        """将批次唯一键解析为结构化字段字典。

        Args:
            key: ``_batch_resource_key`` 生成的批次键，格式：
                ``batch__<resource_kind>__<interval>__<vt_symbol>__<batch_id>``。

        Returns:
            含 ``data_kind``/``resource_kind``/``interval``/``vt_symbol``/``batch_id``
            字段的字典。

        Raises:
            ValueError: key 格式不符或 resource_kind 非法时抛出。
        """
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
        """返回指定合约批次文件所在目录路径（不保证存在）。

        Args:
            data_kind: ``"bar"`` 或 ``"tick"``。
            interval: 周期（Tick 自动替换为 ``"tick"``）。
            vt_symbol: 合约代码（内部自动规范化）。

        Returns:
            ``imports/<data_kind>/<interval>/<vt_symbol>/`` 的 ``Path`` 对象。
        """
        vt_symbol = normalize_vt_symbol(vt_symbol)
        canonical = "tick" if data_kind == "tick" else _canonical_bar_interval(interval)
        return self.imports_path / data_kind / canonical / vt_symbol

    def _batch_file_path(self, data_kind: str, interval: str, vt_symbol: str, batch_id: str) -> Path:
        """返回批次 parquet 数据文件路径（不保证存在）。

        Args:
            data_kind: ``"bar"`` 或 ``"tick"``。
            interval: 周期。
            vt_symbol: 合约代码。
            batch_id: 批次唯一 ID。

        Returns:
            ``imports/<data_kind>/<interval>/<vt_symbol>/<batch_id>.parquet`` 的 ``Path``。
        """
        return self._batch_dir(data_kind, interval, vt_symbol) / f"{batch_id}.parquet"

    def _batch_metadata_path(self, data_kind: str, interval: str, vt_symbol: str, batch_id: str) -> Path:
        """返回批次 sidecar 元数据 JSON 文件路径（不保证存在）。

        Args:
            data_kind: ``"bar"`` 或 ``"tick"``。
            interval: 周期。
            vt_symbol: 合约代码。
            batch_id: 批次唯一 ID。

        Returns:
            ``imports/<data_kind>/<interval>/<vt_symbol>/<batch_id>.meta.json`` 的 ``Path``。
        """
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
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """写入批次 sidecar 元数据 JSON 并返回元数据字典。

        元数据包含批次 ID、合约、周期、行数、时间范围、状态、复权口径、来源等信息，
        供合并计划校验和前端展示使用；传入的 ``extra_meta`` 可追加/覆盖字段。

        Args:
            data_kind: ``"bar"`` 或 ``"tick"``。
            vt_symbol: 合约代码（内部自动规范化）。
            interval: 周期。
            batch_id: 批次唯一 ID。
            file_name: 原始上传文件名（可为 ``None``）。
            df: 已写入 parquet 的批次 DataFrame，用于统计行数和时间范围。
            status: 批次状态，``"pending"``（默认）或 ``"merged"``。
            adjust_type: 复权口径，默认 ``"none"``。
            source: 来源标识，``"upload"``（默认）或 ``"download"``。
            extra_meta: 额外写入的字段字典。

        Returns:
            写入文件的完整元数据字典。
        """
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
        if extra_meta:
            metadata.update(extra_meta)
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
        """加载批次 sidecar 元数据 JSON。

        Args:
            data_kind: ``"bar"`` 或 ``"tick"``。
            interval: 周期。
            vt_symbol: 合约代码。
            batch_id: 批次唯一 ID。

        Returns:
            元数据字典；文件不存在时返回空字典。
        """
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
        """从批次 parquet 文件（及可选的元数据）生成批次摘要字典。

        用于 ``list_data_resources`` 和 ``_save_import_batch`` 返回值，
        提供前端展示所需的行数、时间范围、文件大小等信息。
        若 parquet 读取失败（文件损坏等），降级使用元数据中的统计值。

        Args:
            data_kind: ``"bar"`` 或 ``"tick"``。
            vt_symbol: 合约代码（内部自动规范化）。
            interval: 周期。
            batch_id: 批次唯一 ID。
            file_path: 批次 parquet 文件路径（必须存在）。
            metadata: 预先加载的元数据字典；``None`` 时从 JSON 文件读取。

        Returns:
            包含 ``key``/``kind``/``vt_symbol``/``interval``/``row_count``/
            ``start``/``end``/``file_size_kb``/``status`` 等字段的摘要字典。
        """
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
        """从正式资源 parquet 文件生成资源摘要字典。

        用于 ``list_data_resources`` 遍历 ``bars/``/``ticks/``/``derived/`` 目录时
        构建前端资源列表条目。读取失败时行数和时间范围降级为 0/空串。

        Args:
            kind: 资源类型，``"raw_bar"``/``"derived_bar"``/``"raw_tick"``。
            vt_symbol: 合约代码（内部自动规范化）。
            interval: 周期（Tick 传 ``"tick"``）。
            file_path: parquet 文件路径（必须存在）。
            metadata: 派生层元数据字典（含 source_kind/source_interval 等）；
                ``None`` 时相关字段使用默认值。

        Returns:
            含 ``key``/``kind``/``vt_symbol``/``interval``/``row_count``/
            ``start``/``end``/``file_size_kb``/``source_kind`` 等字段的摘要字典。
        """
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
        """列出本地所有数据资源（原始 K 线/Tick、派生 K 线、待合并批次）。

        遍历 ``bars/``、``ticks/``、``derived/``、``imports/`` 目录，
        为每个 parquet 文件构建摘要条目；重复文件（旧版与新版布局同时存在）按规范代码去重。

        Returns:
            包含以下键的字典：
            - ``raw_bars``：原始 K 线摘要列表；
            - ``raw_ticks``：历史 Tick 摘要列表；
            - ``raw_bar_batches``：待合并 K 线批次列表；
            - ``raw_tick_batches``：待合并 Tick 批次列表；
            - ``derived_bars``：派生 K 线摘要列表；
            - ``raw_bar_intervals``：原始 K 线已有周期列表（排序）；
            - ``derived_intervals``：派生 K 线已有周期列表（排序）。
        """
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
        """将资源类型和键解析为本地文件路径、周期和合约代码三元组。

        不同资源类型的键格式各异：
        - ``raw_tick``：键即合约代码；
        - ``raw_bar_batch``/``raw_tick_batch``：键为 ``_batch_resource_key`` 格式；
        - ``raw_bar``/``derived_bar``：键为 ``<interval>__<vt_symbol>`` 格式。

        Args:
            kind: 资源类型字符串。
            key: 资源唯一键（由 ``_resource_key`` 或 ``_batch_resource_key`` 生成）。

        Returns:
            ``(file_path, interval, vt_symbol)`` 三元组；
            文件路径不保证存在，由调用方校验。

        Raises:
            ValueError: key 格式不符或资源类型不支持时抛出。
        """
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
        """获取单个资源的详情（含分页数据预览）。

        支持游标分页：传入 ``before`` 可向前翻页，每次返回最新的 ``limit`` 条记录。
        时间字段自动转换为 ISO 字符串，浮点字段保留原精度。

        Args:
            kind: 资源类型，如 ``"raw_bar"``/``"derived_bar"``/``"raw_tick"``/
                ``"raw_bar_batch"``/``"raw_tick_batch"``。
            key: 资源唯一键。
            limit: 每页预览行数，``0`` 表示返回全量，默认 ``100``。
            before: 游标，ISO 格式时间字符串（含 ``Z`` 后缀）；
                返回早于该时间的记录（不含边界）。

        Returns:
            含资源元信息和预览行列表的详情字典（key/kind/vt_symbol/interval/
            row_count/start/end/columns/preview/has_more/next_before 等）。

        Raises:
            FileNotFoundError: 资源文件不存在时抛出。
        """
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
        """将原始 K 线资源移动到正确的周期目录，更正错误的周期标签。

        仅重命名/移动文件，不修改数据内容；同步迁移复权口径 sidecar 元数据并记录
        ``relocated_from`` 字段便于审计。若目标周期下已存在同合约文件则拒绝覆盖。

        Args:
            key: 原始 K 线资源键（``<interval>__<vt_symbol>`` 格式）。
            new_interval: 新周期字符串，必须在支持列表内（``d/1m/5m/15m/30m/60m``）。

        Returns:
            含 ``success``/``message``/``key``/``interval``/``vt_symbol`` 的结果字典；
            周期未变化时返回 ``success=True`` 并附说明。

        Raises:
            FileNotFoundError: 资源文件不存在时抛出。
            ValueError: ``new_interval`` 不在支持列表内，或目标周期已有同合约文件时抛出。
        """
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
        """删除指定的原始/派生/批次数据资源文件（含 sidecar 元数据）。

        同时清理对应的 ``.meta.json`` sidecar 文件；文件不存在时返回 ``False`` 而非报错。

        Args:
            kind: 资源类型，支持 ``"raw_bar"``/``"derived_bar"``/``"raw_tick"``/
                ``"raw_bar_batch"``/``"raw_tick_batch"``。
            key: 资源唯一键。

        Returns:
            ``True`` 表示成功删除文件；``False`` 表示文件本不存在。
        """
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
        """按 A 股日内时段检验相邻分钟 K 线时间间隔是否合法。

        跨日、或两个时间点分属不同交易小节（上午/下午）时视为合法断口；
        同一小节内要求相差恰好为 ``interval`` 分钟。

        Args:
            prev_dt: 上一根 K 线时间戳。
            curr_dt: 当前根 K 线时间戳。
            interval: 分钟 K 线周期，如 ``"1m"``、``"5m"``。

        Returns:
            ``True`` 表示间隔合法（含允许跨日/跨小节）；``False`` 表示同小节内断档。
        """
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
        """校验批次 DataFrame 的基本完整性，返回错误信息列表。

        检查项：
        1. 非空；
        2. 含 ``datetime`` 列；
        3. 无空 datetime；
        4. datetime 按升序排列；
        5. datetime 无内部重复；
        6. 分钟 K 线：同交易小节内无断档（调用 ``_batch_minutes_expected``）。

        Args:
            df: 待校验的批次 DataFrame。
            data_kind: ``"bar"`` 或 ``"tick"``；Tick 不做频率连续性校验。
            interval: K 线周期（``data_kind="bar"`` 时使用）。

        Returns:
            错误信息列表；无错误则返回空列表。
        """
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
        """加载现有正式（raw）资源全量数据，作为合并计划的基底。

        仅读取 ``bars/``/``ticks/`` 正式层，不含派生数据，确保合并校验针对已确认数据。

        Args:
            kind: ``"raw_bar"`` 或 ``"raw_tick"``。
            vt_symbol: 合约代码。
            interval: K 线周期（``kind="raw_tick"`` 时忽略）。

        Returns:
            全量正式资源 DataFrame；不存在或为空时返回 ``None``。
        """
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
        """加载并校验指定键列表对应的批次文件，返回批次信息列表和错误列表。

        对每个键：解析批次键 → 校验类型匹配 → 检查文件存在 → 加载元数据 →
        拒绝已合并批次 → 读取 DataFrame → 调用 ``_validate_batch_frame`` 校验完整性。
        成功加载的批次按创建时间升序排列（``batch_id`` 作为时间戳回退键）。

        Args:
            kind: 合并目标类型，``"raw_bar"`` 或 ``"raw_tick"``。
            keys: 批次唯一键列表。

        Returns:
            ``(batches, errors)`` 二元组；``batches`` 中每项含 ``df``/``metadata``/
            ``file_path`` 等字段；``errors`` 为各条失败原因的字符串列表。
        """
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
        """预演合并计划：校验批次与现有正式数据，返回摘要但不写入任何文件。

        是 ``_build_merge_plan`` 的公开只读接口；内部临时对象（DataFrame 缓存等）
        在返回前被过滤掉，不向外泄露。

        Args:
            kind: 合并目标类型，``"raw_bar"`` 或 ``"raw_tick"``。
            keys: 待合并的批次唯一键列表（至少 1 个）。

        Returns:
            合并计划摘要字典，含 ``can_merge``/``reason``/``errors``/
            ``intersection_start``/``intersection_end``/``conflict_count``/
            ``estimated_rows``/``batch_count``/``has_official`` 等字段。
        """
        plan = self._build_merge_plan(kind, keys)
        # 预览结果不向外暴露内部缓存（df / 批次对象）。
        return {key: value for key, value in plan.items() if not key.startswith("_")}

    def merge_import_batches(self, *, kind: str, keys: list[str]) -> dict[str, Any]:
        """执行合并：将通过校验的批次写入正式原始资源（``bars/``/``ticks/``）。

        调用 ``_build_merge_plan`` 再次校验（不缓存前次 preview 结果）；
        合并成功后将各批次元数据状态更新为 ``"merged"``（保留原始批次文件，便于审计）。

        Args:
            kind: 合并目标类型，``"raw_bar"`` 或 ``"raw_tick"``。
            keys: 待合并的批次唯一键列表（至少 1 个）。

        Returns:
            合并结果字典；``success=True`` 时额外含 ``message``/``row_count``/
            ``start``/``end``；``success=False`` 时含 ``reason``/``errors``。
        """
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
        """将指数成分数据批量写入 shelve 文件。

        shelve key 为日期字符串（``"YYYY-MM-DD"``），value 为该日成分股代码列表。
        已有 key 会被覆盖（``db.update``）；新 key 直接追加。

        Args:
            index_symbol: 指数代码，如 ``"000300.SZSE"``；用作 shelve 文件名。
            index_components: ``{"YYYY-MM-DD": ["code1", "code2", ...]}`` 格式字典。
        """
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
        """从 shelve 文件加载指定时间范围内的指数成分数据（带进程内缓存）。

        结果以 ``@lru_cache`` 缓存，适合在因子计算循环中重复调用相同参数。
        缓存依赖参数不可变性：``start``/``end`` 须为可哈希类型（字符串/date）。

        Args:
            index_symbol: 指数代码，如 ``"000300.SZSE"``。
            start: 起始日期（含），传入 ``date``/``datetime``/ISO 字符串均可。
            end: 结束日期（含），同 ``start``。

        Returns:
            ``{datetime: [symbol, ...]}`` 字典，key 为解析后的 datetime 对象；
            shelve 文件不存在时返回空字典。
        """
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
        """获取指数在指定时间范围内出现过的所有成分股代码（去重）。

        常用于构建回测/研究的标的池，无需关心具体日期维度。
        内部调用 ``load_component_data``，结果集合化后返回。

        Args:
            index_symbol: 指数代码。
            start: 起始日期（含）。
            end: 结束日期（含）。

        Returns:
            去重后的成分股代码列表（无序）；无数据时返回空列表。
        """
        components: dict = self.load_component_data(index_symbol, start, end)
        symbols: set[str] = set()
        for syms in components.values():
            symbols.update(syms)
        return list(symbols)

    def save_dataset(self, name: str, dataset: AlphaDataset) -> None:
        """将 AlphaDataset 对象序列化为 pickle 文件保存到 dataset 目录。

        Args:
            name: 数据集名称，用作文件名（不含扩展名）。
            dataset: 待保存的 ``AlphaDataset`` 对象。
        """
        file_path: Path = self.dataset_path / f"{name}.pkl"
        with open(file_path, mode="wb") as f:
            pickle.dump(dataset, f)

    def load_dataset(self, name: str) -> Optional[AlphaDataset]:
        """从 pickle 文件加载 AlphaDataset 对象。

        Args:
            name: 数据集名称（不含扩展名）。

        Returns:
            对应的 ``AlphaDataset`` 对象；文件不存在时记录错误日志并返回 ``None``。
        """
        file_path: Path = self.dataset_path / f"{name}.pkl"
        if not file_path.exists():
            logger.error(f"Dataset file {name} does not exist")
            return None
        with open(file_path, mode="rb") as f:
            return pickle.load(f)

    def remove_dataset(self, name: str) -> bool:
        """删除指定数据集的 pickle 文件。

        Args:
            name: 数据集名称（不含扩展名）。

        Returns:
            ``True`` 表示成功删除；``False`` 表示文件本不存在。
        """
        file_path: Path = self.dataset_path / f"{name}.pkl"
        if not file_path.exists():
            return False
        file_path.unlink()
        return True

    def list_all_datasets(self) -> list[str]:
        """列出 dataset 目录下所有已保存的数据集名称。

        Returns:
            数据集名称列表（文件名去掉 ``.pkl`` 后缀）；无数据集时返回空列表。
        """
        return [f.stem for f in self.dataset_path.glob("*.pkl")]

    def save_model(self, name: str, model: AlphaModel) -> None:
        """将 AlphaModel 对象序列化为 pickle 文件保存到 model 目录。

        Args:
            name: 模型名称（不含扩展名）。
            model: 待保存的 ``AlphaModel`` 对象。
        """
        file_path: Path = self.model_path / f"{name}.pkl"
        with open(file_path, mode="wb") as f:
            pickle.dump(model, f)

    def load_model(self, name: str):
        """从 pickle 文件加载 AlphaModel 对象。

        Args:
            name: 模型名称（不含扩展名）。

        Returns:
            对应的 ``AlphaModel`` 对象；文件不存在时记录错误日志并返回 ``None``。
        """
        file_path: Path = self.model_path / f"{name}.pkl"
        if not file_path.exists():
            logger.error(f"Model file {name} does not exist")
            return None
        with open(file_path, mode="rb") as f:
            return pickle.load(f)

    def remove_model(self, name: str) -> bool:
        """删除指定模型的 pickle 文件。

        Args:
            name: 模型名称（不含扩展名）。

        Returns:
            ``True`` 表示成功删除；``False`` 表示文件本不存在。
        """
        file_path: Path = self.model_path / f"{name}.pkl"
        if not file_path.exists():
            return False
        file_path.unlink()
        return True

    def list_all_models(self) -> list[str]:
        """列出 model 目录下所有已保存的模型名称。

        Returns:
            模型名称列表（文件名去掉 ``.pkl`` 后缀）；无模型时返回空列表。
        """
        return [f.stem for f in self.model_path.glob("*.pkl")]

    def save_signal(self, name: str, signal: pl.DataFrame) -> None:
        """将信号 DataFrame 保存为 parquet 文件到 signal 目录。

        Args:
            name: 信号名称（不含扩展名）。
            signal: 待保存的 polars DataFrame（通常含 ``datetime``/``vt_symbol``/信号值列）。
        """
        file_path: Path = self.signal_path / f"{name}.parquet"
        signal.write_parquet(file_path)

    def load_signal(self, name: str) -> Optional[pl.DataFrame]:
        """从 parquet 文件加载信号 DataFrame。

        Args:
            name: 信号名称（不含扩展名）。

        Returns:
            信号 DataFrame；文件不存在时记录错误日志并返回 ``None``。
        """
        file_path: Path = self.signal_path / f"{name}.parquet"
        if not file_path.exists():
            logger.error(f"Signal file {name} does not exist")
            return None
        return pl.read_parquet(file_path)

    def remove_signal(self, name: str) -> bool:
        """删除指定信号的 parquet 文件。

        Args:
            name: 信号名称（不含扩展名）。

        Returns:
            ``True`` 表示成功删除；``False`` 表示文件本不存在。
        """
        file_path: Path = self.signal_path / f"{name}.parquet"
        if not file_path.exists():
            return False
        file_path.unlink()
        return True

    def list_all_signals(self) -> list[str]:
        """列出 signal 目录下所有已保存的信号名称。

        Returns:
            信号名称列表（文件名去掉 ``.parquet`` 后缀）；无信号时返回空列表。
        """
        return [f.stem for f in self.signal_path.glob("*.parquet")]

    def add_contract_setting(
        self,
        vt_symbol: str,
        long_rate: float,
        short_rate: float,
        size: float,
        pricetick: float,
        stamp_duty: float | None = None,
        slippage: float | None = None,
        limit_ratio: float | None = None,
        t_plus1: bool | None = None,
    ) -> None:
        """添加或更新合约的交易参数配置，持久化到 ``contract.json``。

        每次调用读取现有 JSON → 更新对应合约条目 → 整体写回。
        可选字段（``stamp_duty``/``slippage``/``limit_ratio``/``t_plus1``）
        传 ``None`` 时不写入，避免以空值覆盖已有配置。

        Args:
            vt_symbol: 合约代码（写入前自动规范化）。
            long_rate: 多头手续费率。
            short_rate: 空头手续费率。
            size: 合约乘数（股票通常为 1）。
            pricetick: 最小价格变动单位。
            stamp_duty: 印花税率（A 股仅卖出方收取）；``None`` 表示不写入。
            slippage: 滑点（单边，单位与 pricetick 一致）；``None`` 表示不写入。
            limit_ratio: 涨跌停限制幅度（如 0.1 表示 ±10%）；``None`` 表示不写入。
            t_plus1: ``True`` 表示 T+1 交割（A 股股票），``False`` 表示 T+0；
                ``None`` 表示不写入。
        """
        vt_symbol = normalize_vt_symbol(vt_symbol)
        contracts: dict[str, Any] = {}
        if self.contract_path.exists():
            with open(self.contract_path, encoding="utf-8") as f:
                contracts = json.load(f)

        entry: dict[str, Any] = {
            "long_rate": long_rate,
            "short_rate": short_rate,
            "size": size,
            "pricetick": pricetick,
        }
        # 可选字段：仅在明确传值时写入（None 跳过，YAGNI）
        if stamp_duty is not None:
            entry["stamp_duty"] = stamp_duty
        if slippage is not None:
            entry["slippage"] = slippage
        if limit_ratio is not None:
            entry["limit_ratio"] = limit_ratio
        if t_plus1 is not None:
            entry["t_plus1"] = t_plus1

        contracts[vt_symbol] = entry

        with open(self.contract_path, mode="w+", encoding="utf-8") as f:
            json.dump(contracts, f, indent=4, ensure_ascii=False)

    def load_contract_settings(self) -> dict:
        """从 ``contract.json`` 加载所有合约交易参数配置。

        Returns:
            以合约代码为键的配置字典，格式与 ``add_contract_setting`` 写入一致；
            ``contract.json`` 不存在时返回空字典。
        """
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
        """将 CSV 列名自动匹配到标准字段（兼容中英文别名）。

        遍历 ``CSV_FIELD_MAPPING``（K 线）和可选的 ``TICK_FIELD_MAPPING``，
        将每个标准字段的别名表与 CSV 列名做大小写不敏感比对；
        ``custom_mapping`` 优先级最高，可覆盖自动匹配结果。

        Args:
            columns: CSV 文件的列名列表。
            custom_mapping: 用户自定义的 ``{标准字段: CSV列名}`` 映射；
                ``None`` 时不做覆盖。
            data_kind: ``"bar"``（默认）或 ``"tick"``；
                ``"tick"`` 时额外引入 ``TICK_FIELD_MAPPING`` 别名表。

        Returns:
            ``{标准字段: CSV列名}`` 字典；未匹配到任何别名的标准字段不出现在结果中。

        Example:
            >>> lab.parse_csv_mapping(["日期", "收盘价", "成交量"])
            {"datetime": "日期", "close": "收盘价", "volume": "成交量"}
        """
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
        """解析 CSV 内容并返回预览信息，不写入任何文件。

        自动识别列名、检测缺失必填字段、提取时间范围和标的列表，
        供前端在正式导入前展示映射结果和数据质量摘要。

        Args:
            csv_content: CSV 文件的原始字节内容（支持任何 polars 可读编码）。
            custom_mapping: 用户自定义列名映射（覆盖自动匹配），
                格式 ``{标准字段: CSV列名}``；``None`` 时纯自动匹配。
            data_kind: ``"bar"``（默认）或 ``"tick"``。

        Returns:
            包含以下键的字典：
            - ``data_kind``：数据类型；
            - ``columns``：展示列（``BAR_PREVIEW_FIELDS``/``TICK_PREVIEW_FIELDS``）；
            - ``sample_rows``：前 5 行映射后的预览数据（列表）；
            - ``matched_fields``：自动/自定义列名映射结果；
            - ``unmapped_columns``：未被任何标准字段引用的多余列；
            - ``missing_required``：缺失的必填标准字段；
            - ``total_rows``：总行数；
            - ``date_range``：``("YYYY-MM-DD", "YYYY-MM-DD")`` 起止日期；
            - ``symbols``：识别到的规范化合约代码列表。
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
        """将 BarData 列表转换为标准列的 polars DataFrame。

        时间统一转换为交易所本地裸时间。供 ``save_bar_data`` 和 ``_save_import_batch`` 使用。

        Args:
            bars: 待转换的 K 线列表。

        Returns:
            含 ``datetime/open/high/low/close/volume/turnover/open_interest`` 列的 DataFrame。
        """
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
        """将 TickData 列表转换为标准列的 polars DataFrame。

        时间统一转换为交易所本地裸时间。供 ``save_tick_data`` 和 ``_save_import_batch`` 使用。

        Args:
            ticks: 待转换的历史 Tick 列表。

        Returns:
            含 ``datetime/last_price/volume/turnover/bid_price_1/ask_price_1/
            bid_volume_1/ask_volume_1`` 列的 DataFrame。
        """
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
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """将 K 线/Tick 记录列表写入 imports 批次层，不进入正式资源。

        生成带时间戳前缀的唯一 ``batch_id``，将 DataFrame 写入
        ``imports/<data_kind>/<interval>/<vt_symbol>/<batch_id>.parquet``，
        并写入同名 sidecar 元数据 JSON。

        Args:
            data_kind: ``"bar"`` 或 ``"tick"``。
            vt_symbol: 合约代码（内部自动规范化）。
            interval: K 线周期；``data_kind="tick"`` 时固定为 ``"tick"``。
            records: ``BarData`` 或 ``TickData`` 列表。
            file_name: 原始文件名，记录在元数据中；``None`` 时留空。
            adjust_type: 复权口径，默认 ``"none"``。
            source: 来源标识，默认 ``"upload"``。
            extra_meta: 额外写入元数据的字段字典。

        Returns:
            批次摘要字典（同 ``_batch_summary_from_file`` 返回值）。
        """
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
            extra_meta=extra_meta,
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
        """解析 CSV 文件并将数据保存为 parquet（批次模式或直接合并模式）。

        默认使用 ``save_mode="batch"``：将每个合约的数据写入 imports 批次层，
        待用户手动执行合并；``save_mode="direct"`` 则跳过批次直接写入正式资源
        （``import_mode="replace"`` 时先删除旧文件）。

        Args:
            csv_content: CSV 文件的原始字节内容。
            data_kind: ``"bar"``（默认）或 ``"tick"``。
            interval: K 线周期，仅 ``data_kind="bar"`` 时使用，默认 ``"d"``。
            import_mode: 直接模式下的冲突处理策略：``"merge"``（默认，追加去重）
                或 ``"replace"``（先删除再写入）。
            save_mode: ``"batch"``（默认）写入批次层；``"direct"`` 直接写入正式资源。
            file_name: 原始文件名，记录在批次元数据中；``None`` 时留空。
            custom_mapping: 自定义列名映射，优先级高于自动匹配。

        Returns:
            包含 ``success``/``message``/``imported_count``/``skipped_count``/
            ``errors``/``batches``（批次模式下）的结果字典。
        """
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
