"""
MarketCNN 模型定义 — 多尺度卷积神经网络股价预测模型。

结构:
- 多尺度 Conv2D (1x3, 3x3, 5x3) → 捕捉不同时间跨度的模式
- 时序 Conv1D → 捕捉趋势
- FC → 二分类输出 P(涨)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def check_torch_available() -> bool:
    """检查 PyTorch 是否可用"""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================
# 数据处理
# ============================================================


def _load_bar_records(vt_symbol: str, start: date, end: date) -> list[dict]:
    """
    从 DataSourceManager 加载 K 线记录，返回 list[dict]。
    优先使用真实数据源，回退到 MockProvider。
    """
    from ..datasource import datasource_manager
    from ..datasource.base import BaseProvider
    from ..datasource.mock_provider import MockProvider

    # 解析 vt_symbol
    symbol, exchange = vt_symbol.rsplit(".", 1)

    records = datasource_manager.get_bar_history(
        symbol=symbol,
        exchange=exchange,
        interval="d",
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.max.time()),
    )

    # 无数据且未注册 MockProvider 时，尝试直接调用 MockProvider
    if not records:
        mock = MockProvider()
        mock.init()
        records = mock.get_bar_history(
            symbol=symbol,
            exchange=exchange,
            interval="d",
            start=datetime.combine(start, datetime.min.time()),
            end=datetime.combine(end, datetime.max.time()),
        ) or []

    if not records:
        raise ValueError(f"{vt_symbol} 在 [{start}, {end}] 范围内无数据")

    bars = []
    for r in records:
        bars.append({
            "datetime": r.datetime,
            "open": r.open_price,
            "high": r.high_price,
            "low": r.low_price,
            "close": r.close_price,
            "volume": r.volume,
        })

    bars.sort(key=lambda x: x["datetime"])
    return bars


def _compute_features(bars: list[dict]) -> np.ndarray:
    """
    计算单只股票的技术特征矩阵。

    返回 shape: [T, C] 其中 C=6 个特征:
        0: pct_change (涨跌幅)
        1: volume_ratio (量比)
        2: amplitude (振幅)
        3: ma5_diff (MA5偏差)
        4: ma20_diff (MA20偏差)
        5: high_low_ratio (高低比)
    """
    closes = np.array([b["close"] for b in bars], dtype=np.float64)
    highs = np.array([b["high"] for b in bars], dtype=np.float64)
    lows = np.array([b["low"] for b in bars], dtype=np.float64)
    volumes = np.array([b["volume"] for b in bars], dtype=np.float64)

    n = len(closes)
    features = np.zeros((n, 6), dtype=np.float32)

    # pct_change
    features[1:, 0] = (closes[1:] - closes[:-1]) / np.maximum(closes[:-1], 1e-8)

    # volume_ratio (当日/5日均量)
    for i in range(5, n):
        avg_vol = np.mean(volumes[i - 5:i])
        features[i, 1] = volumes[i] / max(avg_vol, 1e-8)

    # amplitude
    features[1:, 2] = (highs[1:] - lows[1:]) / np.maximum(closes[:-1], 1e-8)

    # ma5_diff
    for i in range(5, n):
        ma5 = np.mean(closes[i - 5:i])
        features[i, 3] = closes[i] / max(ma5, 1e-8) - 1

    # ma20_diff
    for i in range(20, n):
        ma20 = np.mean(closes[i - 20:i])
        features[i, 4] = closes[i] / max(ma20, 1e-8) - 1

    # high_low_ratio
    features[:, 5] = (highs - lows) / np.maximum(closes, 1e-8)

    return features


def build_dataset(
    vt_symbols: list[str],
    start: date,
    end: date,
    lookback: int = 30,
    on_progress: Optional[Callable] = None,
) -> tuple:
    """
    构建 CNN 训练数据集。

    将多只股票的特征堆叠为张量:
        X: [N_samples, C, T, W]  其中 C=6特征, T=lookback, W=股票数
        y: [N_samples]  标签 (0/1 明日涨跌)

    Returns:
        (X, y, dates_info)
    """
    if on_progress:
        on_progress(5, "加载K线数据...")

    # 加载所有股票数据
    all_bars = {}
    for i, vt in enumerate(vt_symbols):
        try:
            bars = _load_bar_records(vt, start, end)
            all_bars[vt] = bars
            if on_progress:
                on_progress(5 + 20 * (i + 1) / len(vt_symbols),
                            f"已加载 {vt} ({len(bars)} 条)")
        except Exception as e:
            logger.warning(f"加载 {vt} 失败: {e}")

    if not all_bars:
        raise ValueError("没有成功加载任何股票数据")

    if on_progress:
        on_progress(30, "计算技术特征...")

    # 计算特征
    all_features = {}
    for vt, bars in all_bars.items():
        all_features[vt] = _compute_features(bars)

    # 找到公共日期范围（所有股票都有数据的日期）
    symbols = list(all_bars.keys())
    min_len = min(len(all_bars[s]) for s in symbols)

    C = 6  # 特征通道数
    W = len(symbols)

    # 对齐：每只股票取最后 min_len 条
    aligned = np.zeros((C, min_len, W), dtype=np.float32)
    for col, sym in enumerate(symbols):
        feat = all_features[sym]
        # 取最后 min_len 行
        feat_aligned = feat[-min_len:]
        # 转置为 [C, T]
        aligned[:, :, col] = feat_aligned.T  # [C=6, T=min_len]

    # Z-Score 标准化（按通道）
    for c in range(C):
        ch = aligned[c]
        mask = np.isfinite(ch)
        if mask.sum() > 0:
            mean = np.nanmean(ch[mask])
            std = np.nanstd(ch[mask]) + 1e-8
            aligned[c] = (ch - mean) / std
    aligned = np.nan_to_num(aligned, nan=0.0)

    if on_progress:
        on_progress(50, "构建训练样本...")

    # 构建滑动窗口样本
    # 标签: 第一只股票的明日涨跌
    first_bars = all_bars[symbols[0]]
    first_closes = np.array([b["close"] for b in first_bars[-min_len:]])

    X_list = []
    y_list = []
    for t in range(lookback, min_len - 1):
        snapshot = aligned[:, t - lookback:t, :]  # [C, lookback, W]
        X_list.append(snapshot)

        ret = (first_closes[t + 1] - first_closes[t]) / max(first_closes[t], 1e-8)
        y_list.append(1.0 if ret > 0 else 0.0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if on_progress:
        on_progress(55, f"数据集就绪: X={X.shape}, 正样本比例={y.mean():.2%}")

    return X, y, {"symbols": symbols, "n_dates": min_len, "lookback": lookback}


# ============================================================
# CNN 模型定义
# ============================================================


def create_market_cnn(
    in_channels: int,
    time_steps: int,
    width: int,
    dropout: float = 0.5,
):
    """创建 MarketCNN 模型（延迟导入 torch 以支持 check_torch_available）"""
    import torch.nn as nn

    class MarketCNN(nn.Module):
        """
        多尺度卷积 + 时序池化的股价预测模型。

        结构:
        - 多尺度 Conv2D (1x3, 3x3, 5x3) → 捕捉不同时间跨度的模式
        - 时序 Conv1D → 捕捉趋势
        - FC → 二分类输出 P(涨)
        """

        def __init__(self, C: int, T: int, W: int, drop: float = 0.5) -> None:
            super().__init__()
            # 多尺度卷积
            self.conv_s = nn.Sequential(
                nn.Conv2d(C, 16, (1, min(3, W)), padding=(0, min(3, W) // 2)),
                nn.BatchNorm2d(16), nn.ReLU(),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(C, 16, (3, min(3, W)), padding=(1, min(3, W) // 2)),
                nn.BatchNorm2d(16), nn.ReLU(),
            )
            self.conv_l = nn.Sequential(
                nn.Conv2d(C, 16, (5, min(3, W)), padding=(2, min(3, W) // 2)),
                nn.BatchNorm2d(16), nn.ReLU(),
            )

            # 时序处理
            self.temporal = nn.Sequential(
                nn.AdaptiveAvgPool2d((T, 1)),   # → [B, 48, T, 1]
                nn.Flatten(2),                   # → [B, 48, T]
            )
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(48, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )

            # 分类头
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 8, 64), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(64, 1), nn.Sigmoid(),
            )

        def forward(self, x):
            f = torch.cat([self.conv_s(x), self.conv_m(x), self.conv_l(x)], dim=1)
            f = self.temporal(f)
            f = self.temporal_conv(f)
            return self.head(f)

    import torch
    return MarketCNN(in_channels, time_steps, width, dropout)


# 兼容旧名称
def _create_model(in_channels: int, time_steps: int, width: int, dropout: float = 0.5):
    """兼容别名"""
    return create_market_cnn(in_channels, time_steps, width, dropout)
