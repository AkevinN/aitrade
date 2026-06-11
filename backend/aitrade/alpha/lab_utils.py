"""
AlphaLab 模块级纯函数工具集。

从 lab.py 抽出：周期(interval)规范化、vt_symbol 规范化与本地文件名变体、
日期边界与预览辅助。均为无状态纯函数，供 AlphaLab 及其他模块复用。
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

from dateutil import parser as dateutil_parser

from .dataset import to_datetime


def _canonical_bar_interval(interval: str | None) -> str:
    """Normalize user/provider interval strings to storage keys."""
    raw = (interval or "").strip().lower()
    mapping = {
        "daily": "d",
        "day": "d",
        "d": "d",
        "weekly": "w",
        "week": "w",
        "w": "w",
        "minute": "1m",
        "m": "1m",
        "1m": "1m",
        "5m": "5m",
        "10m": "10m",
        "15m": "15m",
        "30m": "30m",
        "60m": "60m",
        "1h": "60m",
    }
    return mapping.get(raw, raw)


def _display_bar_interval(interval: str) -> str:
    """Convert storage interval to UI-facing short label."""
    return "m" if interval == "1m" else interval


def _is_minute_interval(interval: str) -> bool:
    canonical = _canonical_bar_interval(interval)
    return canonical.endswith("m") and canonical[:-1].isdigit()


def _interval_minutes(interval: str) -> int:
    canonical = _canonical_bar_interval(interval)
    if not _is_minute_interval(canonical):
        raise ValueError(f"不支持的分钟周期: {interval}")
    return int(canonical[:-1])


def _parse_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    if "." in vt_symbol:
        return vt_symbol.rsplit(".", 1)
    return vt_symbol, ""


_EXCHANGE_ALIASES: dict[str, str] = {
    "SZ": "SZSE",
    "SZSE": "SZSE",
    "SH": "SSE",
    "SSE": "SSE",
    "BJ": "BSE",
    "BSE": "BSE",
}

_PREFIX_EXCHANGES: dict[str, str] = {
    "sz": "SZSE",
    "sh": "SSE",
    "bj": "BSE",
}


def normalize_vt_symbol(raw: str) -> str:
    """将常见证券代码写法统一为 symbol.EXCHANGE 格式。"""
    value = (raw or "").strip()
    if not value:
        return value

    lower = value.lower()
    if len(lower) >= 8 and lower[:2] in _PREFIX_EXCHANGES:
        return f"{lower[2:8]}.{_PREFIX_EXCHANGES[lower[:2]]}"

    if "." in value:
        symbol, exchange = value.rsplit(".", 1)
        normalized_exchange = _EXCHANGE_ALIASES.get(exchange.upper(), exchange.upper())
        return f"{symbol}.{normalized_exchange}"

    digits = value
    if digits.isdigit() and len(digits) == 6:
        if digits.startswith(("5", "6", "9")):
            return f"{digits}.SSE"
        if digits.startswith(("4", "8")):
            return f"{digits}.BSE"
        return f"{digits}.SZSE"

    return value


def _append_lookup_key(keys: list[str], candidate: str) -> None:
    value = (candidate or "").strip()
    if value and value not in keys:
        keys.append(value)


def _vt_symbol_lookup_keys(vt_symbol: str) -> list[str]:
    """生成本地数据文件可能使用的证券代码变体。"""
    raw = (vt_symbol or "").strip().rstrip(".")
    normalized = normalize_vt_symbol(raw)
    symbol, exchange = _parse_vt_symbol(normalized)

    keys: list[str] = []
    for candidate in [normalized, raw]:
        _append_lookup_key(keys, candidate)
    if symbol:
        _append_lookup_key(keys, symbol)
        for prefix in _PREFIX_EXCHANGES:
            prefixed = f"{prefix}{symbol}"
            for variant in (prefixed, f"{prefixed}.", f"{prefixed}.."):
                _append_lookup_key(keys, variant)
        for variant in (f"{symbol}.", f"{symbol}.."):
            _append_lookup_key(keys, variant)
    if symbol and exchange:
        for variant in (f"{symbol}.{exchange}", f"{symbol}.{exchange}."):
            _append_lookup_key(keys, variant)
    return keys


def canonical_vt_symbol_from_stem(stem: str) -> str:
    """将 parquet 文件名（不含扩展名）规范为统一证券代码。"""
    return normalize_vt_symbol(stem.rstrip("."))


def _datetime_preview_values(values: list[Any]) -> tuple[str, str]:
    parsed_dates: list[datetime] = []
    for value in values:
        try:
            parsed_dates.append(dateutil_parser.parse(str(value)))
        except Exception:
            continue
    if not parsed_dates:
        return "", ""
    parsed_dates.sort()
    return parsed_dates[0].strftime("%Y-%m-%d"), parsed_dates[-1].strftime("%Y-%m-%d")


def _normalize_bound(value: Any, *, is_end: bool) -> datetime:
    """Normalize date-like values into explicit datetime bounds."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, time.max if is_end else time.min)

    converted = to_datetime(value)
    if isinstance(converted, datetime):
        return converted.replace(tzinfo=None)
    if isinstance(converted, date):
        return datetime.combine(converted, time.max if is_end else time.min)
    return datetime.combine(value, time.max if is_end else time.min)
