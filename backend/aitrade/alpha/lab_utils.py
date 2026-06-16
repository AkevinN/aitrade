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
    """将用户/数据提供方的周期字符串规范化为内部存储键。

    支持别名：``"daily"``/``"day"``/``"d"`` → ``"d"``；
    ``"1h"`` → ``"60m"``；``"m"``/``"minute"`` → ``"1m"`` 等。
    未在映射表内的值原样返回（小写，去首尾空白）。

    Args:
        interval: 原始周期字符串，大小写不敏感，可含首尾空白；传 ``None`` 按空串处理。

    Returns:
        规范化后的周期键，如 ``"d"``、``"1m"``、``"5m"``、``"30m"``、``"60m"``；
        无法识别时返回原值（小写）。
    """
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
    """将内部存储周期键转换为前端展示用的短标签。

    当前唯一映射：``"1m"`` → ``"m"``；其余值原样返回。
    用于列表/选择器中减少视觉噪音。

    Args:
        interval: 内部存储周期键，如 ``"d"``、``"1m"``、``"5m"``。

    Returns:
        UI 短标签字符串；非 ``"1m"`` 的值原样返回。
    """
    return "m" if interval == "1m" else interval


def _is_minute_interval(interval: str) -> bool:
    """判断给定周期是否为分钟周期（如 1m/5m/30m/60m）。

    先经 ``_canonical_bar_interval`` 规范化，再检查是否为 ``<数字>m`` 格式。
    日线（``"d"``）和周线（``"w"``）返回 ``False``。

    Args:
        interval: 任意格式的周期字符串，大小写不敏感。

    Returns:
        ``True`` 表示为分钟周期；``False`` 表示日线、周线或未知周期。
    """
    canonical = _canonical_bar_interval(interval)
    return canonical.endswith("m") and canonical[:-1].isdigit()


def _interval_minutes(interval: str) -> int:
    """将分钟周期字符串解析为对应的分钟数。

    仅支持分钟周期；日线等非分钟周期会抛出 ``ValueError``。
    常用于判断周期整除关系（聚合时 target_minutes % source_minutes == 0）。

    Args:
        interval: 周期字符串，如 ``"1m"``、``"5m"``、``"30m"``、``"60m"``。
            非规范形式先经 ``_canonical_bar_interval`` 转换。

    Returns:
        该周期对应的整数分钟数，如 ``1``、``5``、``30``、``60``。

    Raises:
        ValueError: ``interval`` 规范化后不是分钟周期（如传入 ``"d"``）。
    """
    canonical = _canonical_bar_interval(interval)
    if not _is_minute_interval(canonical):
        raise ValueError(f"不支持的分钟周期: {interval}")
    return int(canonical[:-1])


def _parse_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    """将 ``symbol.EXCHANGE`` 格式拆分为 ``(symbol, exchange)`` 二元组。

    按最后一个 ``.`` 分割，不含 ``.`` 的字符串返回 ``(vt_symbol, "")``。

    Args:
        vt_symbol: 证券代码字符串，标准格式如 ``"000001.SZSE"``；
            不含交易所后缀时也合法。

    Returns:
        ``(symbol, exchange)`` 二元组；无法识别交易所时 exchange 为空串。
    """
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
    """将常见证券代码写法统一为 symbol.EXCHANGE 格式。

    支持多种输入形态：
    - 带交易所后缀（已规范）：``"600519.SSE"`` → 原样或别名映射后返回。
    - 带前缀：``"sh600519"`` / ``"SH600519"`` → ``"600519.SSE"``。
    - 纯 6 位数字：按首字符规则推断交易所：
        - 沪市转债前缀（110/111/113/118）→ ``SSE``（优先于通用首字符规则，
          避免 "1" 开头的转债代码被误归为深市）
        - 5/6/9 开头 → ``SSE``
        - 4/8 开头 → ``BSE``（北交所）
        - 其余 → ``SZSE``

    Args:
        raw: 原始证券代码字符串，可含首字母前缀或 "." 分隔交易所后缀。

    Returns:
        规范化后的 ``symbol.EXCHANGE`` 字符串；输入为空时返回空串；
        无法识别时原样返回（不抛错，由调用方决定处理方式）。
    """
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
        # 沪市转债（110/111/113/118 开头）在通用首字符规则之前判定，
        # 否则 "1" 开头会落入 SZSE 默认分支
        if digits.startswith(("110", "111", "113", "118")):
            return f"{digits}.SSE"
        if digits.startswith(("5", "6", "9")):
            return f"{digits}.SSE"
        if digits.startswith(("4", "8")):
            return f"{digits}.BSE"
        return f"{digits}.SZSE"

    return value


def _append_lookup_key(keys: list[str], candidate: str) -> None:
    """将候选字符串去空白后追加到列表（原地去重，保持插入顺序）。

    空串或已在列表中的值被静默跳过，不抛错。
    供 ``_vt_symbol_lookup_keys`` 内联使用。

    Args:
        keys: 目标列表，原地修改。
        candidate: 待追加的候选代码字符串；空串或纯空白字符直接跳过。
    """
    value = (candidate or "").strip()
    if value and value not in keys:
        keys.append(value)


def _vt_symbol_lookup_keys(vt_symbol: str) -> list[str]:
    """生成本地数据文件可能使用的证券代码变体列表。

    历史遗留文件可能以 ``"000001"``、``"sz000001"``、``"000001.SZSE"``、
    ``"000001."`` 等多种形式命名。此函数穷举常见变体，使 ``_iter_bar_candidates``
    和 ``_scan_bar_files`` 能通过「文件名试探」定位到正确文件。

    Args:
        vt_symbol: 任意格式的证券代码，如 ``"000001"``、``"SZ000001"``、
            ``"000001.SZSE"``。

    Returns:
        去重后的代码变体列表（保持生成顺序）；规范形式排在首位，
        纯 symbol 和各类前缀变体依次追加。
    """
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
    """将 parquet 文件名（不含扩展名）规范化为统一证券代码。

    先去除尾部多余的 ``.`` 后调用 ``normalize_vt_symbol``，
    用于遍历 parquet 文件目录时将文件名反推为标准代码。

    Args:
        stem: parquet 文件名去掉 ``.parquet`` 扩展名后的字符串，
            如 ``"000001.SZSE"``、``"sh600000."``、``"600519"``。

    Returns:
        规范化的 ``symbol.EXCHANGE`` 格式代码；无法识别时原样返回。
    """
    return normalize_vt_symbol(stem.rstrip("."))


def _datetime_preview_values(values: list[Any]) -> tuple[str, str]:
    """从一列原始值中尝试解析日期，返回最早和最晚日期字符串。

    用于 CSV 预览时快速提取时间范围，输入值类型不限（字符串、datetime 等）；
    无法解析的项静默跳过，不影响其余项。

    Args:
        values: 待解析的原始值列表，通常来自 CSV 的 datetime 列。

    Returns:
        ``(start_str, end_str)`` 二元组，格式 ``"YYYY-MM-DD"``；
        所有值均无法解析时返回 ``("", "")``。
    """
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
    """将日期/时间类值统一转换为无时区的 datetime 边界。

    用于将 ``start``/``end`` 参数规范化，使 polars 过滤条件一致：
    - 已是 ``datetime``：去掉时区（转 Asia/Shanghai 后 strip tzinfo）；
    - 仅是 ``date``：补上 ``time.min``（起始边界）或 ``time.max``（结束边界）；
    - 字符串或其他类型：先经 ``to_datetime`` 转换再套用上述规则。

    Args:
        value: 日期/时间值，支持 ``datetime``、``date``、ISO 格式字符串等。
        is_end: ``True`` 表示作为结束边界（补 ``time.max``），
            ``False`` 表示作为起始边界（补 ``time.min``）。

    Returns:
        无时区的 ``datetime`` 对象，可直接用于 polars datetime 列过滤。
    """
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
