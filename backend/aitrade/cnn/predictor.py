"""
CNN model inference — load a trained checkpoint and generate prediction signals.

The output DataFrame has columns [datetime, vt_symbol, signal] which is
compatible with the shared backtesting engine's signal format.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Callable, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# 各输入周期下每个交易日的 bar 数（A 股 4 小时连续竞价）。
_BARS_PER_DAY: dict[str, int] = {
    "d": 1, "60m": 4, "30m": 8, "15m": 16, "10m": 24, "5m": 48, "1m": 240,
}


def warmup_days(lookback: int, input_interval: str) -> int:
    """推理 warm-up 回退的日历天数。

    `lookback` 是 **bar 数**：先按周期折算成所需交易日数，再乘 2.5 的日历裕量
    （周末/节假日/停牌），下限 5 天。日频与历史公式 `lookback * 2.5` 等价；
    分钟频不再把 bar 数当天数（消除数百日分钟数据的过度拉取）。
    """
    import math

    bars_per_day = _BARS_PER_DAY.get(input_interval, 1)
    return max(5, math.ceil(lookback / bars_per_day * 2.5))


def predict_cnn_signals(
    model_name: str,
    start: date,
    end: date,
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_meta: Optional[Callable[[dict], None]] = None,
) -> pl.DataFrame:
    """加载已训练的 CNN 模型并在指定区间生成预测信号。

    流程：
    1. 加载 checkpoint，重建模型结构并恢复权重；
    2. 以 warmup_days 向前扩展 start 日期，加载预热数据；
    3. 对齐多证券时间轴、计算技术特征、填入分组张量；
    4. 使用训练时的归一化统计量对特征标准化；
    5. 滑动窗口批量推理，仅保留 [start, end] 区间内的信号。

    Args:
        model_name: 模型名称（不含 .pt 后缀），对应 CNN_MODEL_DIR/<name>.pt。
        start: 信号生成起始日期（含）；实际加载数据会向前扩展 warmup_days 天。
        end: 信号生成结束日期（含）。
        on_progress: 进度回调 ``(percent, message)``，可为 None。
        on_meta: 推理完成后调用一次的元信息回调 ``(meta_dict) -> None``，可为 None；
            meta_dict 含 target_symbol/lookback/input_interval/objective 等观测信息，不含凭证。

    Returns:
        polars DataFrame，输出列因 objective 而异：

        - **classification / regression**（三列）：
          ``[datetime, vt_symbol, signal]``；
          classification 的 signal 为上涨概率（0~1），regression 为预测收益（无界）。

        - **path_class**（七列）：
          ``[datetime, vt_symbol, signal, prob_tp, prob_sl, prob_time_up, prob_time_down]``；
          signal 恒等于 prob_tp（止盈先触发的概率）；
          四列概率由 softmax 计算，行内和严格为 1。

        所有 objective 下 datetime 均去除时区信息，与回测引擎的 bar datetime 对齐。

    Raises:
        FileNotFoundError: 模型文件不存在时抛出。
        ValueError: checkpoint 的 num_classes 与 path_class 要求不符（非 4）时抛出；
            或加载观测证券数据失败、推理后无结果时抛出。
    """
    import torch

    from .storage import CNN_MODEL_DIR
    from .model import (
        create_market_cnn,
        _compute_features,
        _load_market_frame,
        _align_frames_by_datetime,
        _extract_aligned_bars,
        _build_grouped_tensor,
    )

    # 1. Load checkpoint
    model_path = CNN_MODEL_DIR / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"CNN 模型不存在: {model_name}")

    if on_progress:
        on_progress(5, f"加载模型 {model_name}...")

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    train_config = checkpoint["train_config"]
    normalization = checkpoint["normalization"]

    target_symbol: str = train_config["target_symbol"]
    lookback: int = train_config["lookback"]
    input_data_kind: str = train_config.get("input_data_kind", "bar")
    input_interval: str = train_config.get("input_interval", "d")
    dropout: float = model_config.get("dropout", 0.5)
    # 分类模型 signal 为上涨概率(0~1)；回归模型 signal 为预测收益(无界)；
    # path_class 模型 signal == prob_tp，另附 prob_sl/prob_time_up/prob_time_down。
    objective: str = train_config.get("objective", "classification")

    # 冗余校验：path_class checkpoint 若存在 num_classes 键，其值必须为 4；
    # 键缺失（旧 checkpoint 向后兼容）时不报错，仅在值存在且不等于 4 时拒绝。
    if objective == "path_class":
        num_classes = model_config.get("num_classes")
        if num_classes is not None and num_classes != 4:
            raise ValueError(
                f"path_class checkpoint 的 num_classes 应为 4，实得 {num_classes}；"
                "checkpoint 可能被手工篡改。"
            )

    # Rebuild observation groups
    raw_groups = train_config.get("observation_groups", [])
    vt_symbols: list[str] = []
    for group in raw_groups:
        for sym in group.get("symbols", []):
            if sym not in vt_symbols:
                vt_symbols.append(sym)

    if on_progress:
        on_progress(10, f"目标证券: {target_symbol}, 观测组: {len(raw_groups)} 个")

    # 2. Load market data — warm-up 回退按 input_interval 换算（lookback 是 bar 数）：
    #    覆盖 lookback 根 bar 所需的交易日数 ×2.5 日历裕量（节假日/停牌），下限 5 天。
    from datetime import timedelta
    extended_start = start - timedelta(days=warmup_days(lookback, input_interval))

    if on_progress:
        on_progress(15, f"加载 {len(vt_symbols)} 个观测证券的 {input_interval} 数据...")

    symbol_frames: dict[str, pl.DataFrame] = {}
    for idx, vt_symbol in enumerate(vt_symbols):
        try:
            symbol_frames[vt_symbol] = _load_market_frame(
                vt_symbol, extended_start, end,
                input_data_kind=input_data_kind,
                input_interval=input_interval,
            )
        except Exception as exc:
            logger.warning(f"加载 {vt_symbol} 失败: {exc}")
            raise ValueError(f"推理时加载 {vt_symbol} 失败: {exc}")

        if on_progress:
            on_progress(15 + 20 * (idx + 1) / len(vt_symbols),
                        f"已加载 {vt_symbol} ({idx + 1}/{len(vt_symbols)})")

    # 3. Align frames and compute features
    if on_progress:
        on_progress(38, "按公共时间轴对齐...")

    symbols, aligned_df = _align_frames_by_datetime(symbol_frames)

    all_features: dict[str, np.ndarray] = {}
    for vt_symbol in symbols:
        all_features[vt_symbol] = _compute_features(
            _extract_aligned_bars(aligned_df, vt_symbol)
        )

    total_steps = aligned_df.height
    if total_steps <= lookback:
        raise ValueError(f"数据时间步不足: {total_steps}, 至少需要 > {lookback}")

    # 4. Build grouped tensor
    feature_channels = model_config["in_channels"]
    group_count = model_config["group_count"]
    max_group_width = model_config["max_group_width"]

    aligned_tensor, group_mask = _build_grouped_tensor(
        raw_groups,
        all_features,
        feature_channels,
        total_steps,
        max_group_width,
        group_count,
    )

    # 5. Normalize using training stats
    channel_mean = np.array(normalization["channel_mean"], dtype=np.float32).reshape(
        feature_channels, 1, 1, 1
    )
    channel_std = np.array(normalization["channel_std"], dtype=np.float32).reshape(
        feature_channels, 1, 1, 1
    )

    aligned_tensor = (aligned_tensor - channel_mean) / channel_std
    # Apply group mask
    mask_expanded = np.broadcast_to(
        group_mask[0, 0, 0],  # [S, G]
        (feature_channels, total_steps, max_group_width, group_count),
    )
    aligned_tensor = aligned_tensor * mask_expanded
    aligned_tensor = np.nan_to_num(aligned_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    if on_progress:
        on_progress(50, "构建推理模型...")

    # 6. Create model and load weights
    C = model_config["in_channels"]
    T = model_config["time_steps"]
    S = model_config["max_group_width"]
    G = model_config["group_count"]
    model = create_market_cnn(C, T, S, G, dropout, objective=objective)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 7. Sliding window inference
    aligned_dates: list[datetime] = aligned_df["datetime"].to_list()
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())

    predictions: list[dict[str, Any]] = []
    valid_indices: list[int] = []

    for anchor in range(lookback - 1, total_steps):
        dt = aligned_dates[anchor].replace(tzinfo=None)
        # 仅保留 [start, end] 区间内的信号：前段是 warm-up 数据，end 之后为越界数据
        if dt < start_dt or dt > end_dt:
            continue
        valid_indices.append(anchor)

    if on_progress:
        on_progress(55, f"开始推理 {len(valid_indices)} 个时间点...")

    # Batch inference
    batch_size = 64
    for batch_start in range(0, len(valid_indices), batch_size):
        batch_indices = valid_indices[batch_start:batch_start + batch_size]
        batch_x = []
        batch_masks = []

        for anchor in batch_indices:
            snapshot = aligned_tensor[:, anchor - lookback + 1: anchor + 1, :, :]
            batch_x.append(snapshot)
            batch_masks.append(group_mask[0])  # [1, 1, S, G]

        x_tensor = torch.FloatTensor(np.array(batch_x)).to(device)
        m_tensor = torch.FloatTensor(np.array(batch_masks)).to(device)

        with torch.no_grad():
            out = model(x_tensor, m_tensor)

        if objective == "path_class":
            # path_class：softmax 得到四类概率矩阵 [B, 4]；
            # 列顺序与 dataset.PATH_CLASS_NAMES 对应：0=tp,1=sl,2=time_up,3=time_down。
            probs_mat = torch.softmax(out, dim=1).cpu().numpy()  # [B, 4]
            for i, anchor in enumerate(batch_indices):
                dt = aligned_dates[anchor]
                p = probs_mat[i]
                predictions.append({
                    "datetime": dt.replace(tzinfo=None),
                    "vt_symbol": target_symbol,
                    "signal": float(p[0]),       # signal 恒等于 prob_tp
                    "prob_tp": float(p[0]),
                    "prob_sl": float(p[1]),
                    "prob_time_up": float(p[2]),
                    "prob_time_down": float(p[3]),
                })
        else:
            probs = out.cpu().numpy().flatten()
            for i, anchor in enumerate(batch_indices):
                dt = aligned_dates[anchor]
                predictions.append({
                    "datetime": dt.replace(tzinfo=None),
                    "vt_symbol": target_symbol,
                    "signal": float(probs[i]),
                })

        if on_progress:
            pct = 55 + 40 * min(batch_start + batch_size, len(valid_indices)) / max(len(valid_indices), 1)
            on_progress(pct, f"已推理 {min(batch_start + batch_size, len(valid_indices))}/{len(valid_indices)}")

    if not predictions:
        raise ValueError("推理未产生任何预测结果，请检查日期范围")

    signal_df = pl.DataFrame(predictions)

    if on_progress:
        on_progress(98, f"推理完成: {len(predictions)} 个信号, 均值={signal_df['signal'].mean():.4f}")

    # 推理可观测信息采集：仅当调用方传入 on_meta 时，一次性吐出结构化元信息
    # （仅符号、计数与时间，绝不含任何凭证）。默认 None，向后兼容。
    if on_meta is not None:
        on_meta({
            "target_symbol": target_symbol,
            "lookback": lookback,
            "input_interval": input_interval,
            "objective": objective,
            "observation_symbols": list(vt_symbols),
            "observation_group_count": len(raw_groups),
            "warmup_start": extended_start.isoformat(),
            "total_steps": total_steps,
            "valid_points": len(valid_indices),
            "per_symbol_bars": {
                sym: frame.height for sym, frame in symbol_frames.items()
            },
        })

    return signal_df
