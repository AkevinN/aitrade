"""
CNN 特征工程与行情数据加载辅助。

从 model.py 抽出：证券规范化、观测分组、本地 K 线加载、按时间轴对齐、
技术特征通道计算、分组张量装配。训练（dataset）与推理（predictor）共用本模块，
避免两处实现漂移。
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = [
    "pct_change",
    "volume_ratio",
    "amplitude",
    "ma5_diff",
    "ma20_diff",
    "high_low_ratio",
]


def check_torch_available() -> bool:
    """检查 PyTorch 是否可用。"""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _normalize_symbol_list(symbols: list[str]) -> list[str]:
    from ..alpha.lab import normalize_vt_symbol

    return list(dict.fromkeys(
        normalize_vt_symbol(symbol)
        for symbol in symbols
        if symbol and str(symbol).strip()
    ))


def normalize_observation_groups(
    *,
    target_symbol: str | None,
    observation_groups: list[dict[str, Any]] | None,
    vt_symbols: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Build a normalized semantic group list.

    Compatibility behavior:
    - if observation_groups is empty, map legacy vt_symbols into target + custom
    - target group is always injected as the first group
    """
    from ..alpha.lab import normalize_vt_symbol

    symbol_list = _normalize_symbol_list(list(vt_symbols or []))
    if target_symbol:
        target_symbol = normalize_vt_symbol(target_symbol)
    if target_symbol is None and symbol_list:
        target_symbol = symbol_list[0]
    if not target_symbol:
        raise ValueError("必须提供 target_symbol，或至少在 vt_symbols 中包含一个证券")

    groups: list[dict[str, Any]] = []
    seen_symbols: set[str] = {target_symbol}

    groups.append({
        "role": "target",
        "name": "目标证券",
        "symbols": [target_symbol],
    })

    if observation_groups:
        for group in observation_groups:
            raw_role = group.get("role") or "custom"
            role = getattr(raw_role, "value", str(raw_role))
            # 处理历史遗留的 "ObservationRole.XXX" 格式
            if role.startswith("ObservationRole."):
                role = role.split(".", 1)[1].lower()
            if role == "target":
                continue
            symbols = _normalize_symbol_list(list(group.get("symbols", [])))
            if not symbols:
                continue
            seen_symbols.update(symbols)
            groups.append(
                {
                    "role": role,
                    "name": str(group.get("name") or role),
                    "symbols": symbols,
                }
            )
    elif len(symbol_list) > 1:
        groups.append(
            {
                "role": "custom",
                "name": "自定义观测组",
                "symbols": symbol_list[1:],
            }
        )

    if not groups:
        raise ValueError("至少需要一个观测组")

    return target_symbol, groups


def _load_market_frame(
    vt_symbol: str,
    start: date,
    end: date,
    *,
    input_data_kind: str,
    input_interval: str,
) -> pl.DataFrame:
    """Load local raw/derived bar frame for CNN input."""
    from ..alpha import AlphaLab
    from ..alpha.lab import normalize_vt_symbol
    from ..config import ALPHA_LAB_PATH

    canonical_symbol = normalize_vt_symbol(vt_symbol)
    lab = AlphaLab(ALPHA_LAB_PATH)
    frame = lab.load_or_aggregate_bar_frame(
        vt_symbol=canonical_symbol,
        interval=input_interval,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.max.time()),
        input_data_kind=input_data_kind,
    )
    if frame is None or frame.is_empty():
        full_frame = lab.load_bar_frame_any_range(
            canonical_symbol,
            input_interval,
            include_derived=True,
        )
        if full_frame is not None and not full_frame.is_empty() and "datetime" in full_frame.columns:
            local_start = str(full_frame["datetime"].min())[:10]
            local_end = str(full_frame["datetime"].max())[:10]
            raise ValueError(
                f"{canonical_symbol} 在 [{start}, {end}] 范围内无可用 {input_interval} 数据"
                f"（本地区间: {local_start} ~ {local_end}）"
            )
        hint = (
            f"，请确认已在数据准备中下载/导入 {input_interval} 数据"
        )
        if canonical_symbol != vt_symbol:
            hint += f"（已尝试将 {vt_symbol} 规范为 {canonical_symbol}）"
        raise ValueError(f"{canonical_symbol} 无本地 {input_interval} 数据{hint}")
    return frame.sort("datetime")


def _align_frames_by_datetime(symbol_frames: dict[str, pl.DataFrame]) -> tuple[list[str], pl.DataFrame]:
    """Inner join all symbols by common datetimes."""
    merged: pl.DataFrame | None = None
    symbols = list(symbol_frames.keys())

    for vt_symbol in symbols:
        frame = symbol_frames[vt_symbol]
        rename_map = {
            column: f"{vt_symbol}__{column}"
            for column in frame.columns
            if column != "datetime"
        }
        frame = frame.rename(rename_map)
        merged = frame if merged is None else merged.join(frame, on="datetime", how="inner")

    if merged is None or merged.is_empty():
        raise ValueError("观测证券无法按公共时间轴对齐，请检查本地数据是否齐全")

    return symbols, merged.sort("datetime")


def _extract_aligned_bars(aligned_df: pl.DataFrame, vt_symbol: str) -> list[dict[str, Any]]:
    """Extract one symbol's aligned OHLCV sequence from the joined frame."""
    rows: list[dict[str, Any]] = []
    for row in aligned_df.iter_rows(named=True):
        rows.append(
            {
                "datetime": row["datetime"],
                "open": row[f"{vt_symbol}__open"],
                "high": row[f"{vt_symbol}__high"],
                "low": row[f"{vt_symbol}__low"],
                "close": row[f"{vt_symbol}__close"],
                "volume": row[f"{vt_symbol}__volume"],
            }
        )
    return rows


def _compute_features(bars: list[dict[str, Any]]) -> np.ndarray:
    """Calculate six simple technical channels for one symbol."""
    closes = np.array([bar["close"] for bar in bars], dtype=np.float64)
    highs = np.array([bar["high"] for bar in bars], dtype=np.float64)
    lows = np.array([bar["low"] for bar in bars], dtype=np.float64)
    volumes = np.array([bar["volume"] for bar in bars], dtype=np.float64)

    n = len(closes)
    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

    features[1:, 0] = (closes[1:] - closes[:-1]) / np.maximum(closes[:-1], 1e-8)
    for i in range(5, n):
        avg_vol = np.mean(volumes[i - 5:i])
        features[i, 1] = volumes[i] / max(avg_vol, 1e-8)
    features[1:, 2] = (highs[1:] - lows[1:]) / np.maximum(closes[:-1], 1e-8)
    for i in range(5, n):
        ma5 = np.mean(closes[i - 5:i])
        features[i, 3] = closes[i] / max(ma5, 1e-8) - 1
    for i in range(20, n):
        ma20 = np.mean(closes[i - 20:i])
        features[i, 4] = closes[i] / max(ma20, 1e-8) - 1
    features[:, 5] = (highs - lows) / np.maximum(closes, 1e-8)

    return features


def _build_grouped_tensor(
    groups: list[dict[str, Any]],
    all_features: dict[str, np.ndarray],
    feature_channels: int,
    total_steps: int,
    max_group_width: int,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按观测分组把各证券特征填入 [C, T, S, G] 张量，并生成分组掩码 [1, 1, 1, S, G]。

    训练（build_dataset）与推理（predictor）共用同一套填充逻辑，避免两处实现漂移。
    某个位置缺少特征时保持 0 值且掩码为 0（视为占位/无效证券）。
    """
    aligned = np.zeros(
        (feature_channels, total_steps, max_group_width, group_count), dtype=np.float32
    )
    group_mask = np.zeros((1, 1, 1, max_group_width, group_count), dtype=np.float32)

    for group_index, group in enumerate(groups):
        for symbol_index, vt_symbol in enumerate(group.get("symbols", [])):
            features = all_features.get(vt_symbol)
            if features is None:
                continue
            aligned[:, :, symbol_index, group_index] = features.T
            group_mask[0, 0, 0, symbol_index, group_index] = 1.0

    return aligned, group_mask
