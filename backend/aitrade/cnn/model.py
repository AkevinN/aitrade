"""
CNN model 模块 —— 向后兼容门面（facade）。

真正的实现已按职责拆分到：
- features.py : 特征工程 + 行情数据加载/对齐 + 分组张量装配
- dataset.py  : 标签生成 + 分组样本构建（build_dataset）
- network.py  : 分组感知 CNN 工厂（create_market_cnn，保留 lazy torch）

本模块保留为 re-export 层，确保既有 `from .model import X`
以及 `from aitrade.cnn import model` 后 `model.X` 的访问方式继续可用。
"""

from __future__ import annotations

from .features import (
    FEATURE_NAMES,
    check_torch_available,
    normalize_observation_groups,
    _normalize_symbol_list,
    _load_market_frame,
    _align_frames_by_datetime,
    _extract_aligned_bars,
    _compute_features,
    _build_grouped_tensor,
)
from .dataset import (
    build_dataset,
    _normalize_label_spec,
    _build_session_last_index,
    _label_future_index,
)
from .network import create_market_cnn, _create_model

__all__ = [
    "FEATURE_NAMES",
    "check_torch_available",
    "normalize_observation_groups",
    "build_dataset",
    "create_market_cnn",
    "_create_model",
    "_compute_features",
    "_load_market_frame",
    "_align_frames_by_datetime",
    "_extract_aligned_bars",
    "_build_grouped_tensor",
    "_normalize_symbol_list",
    "_normalize_label_spec",
    "_build_session_last_index",
    "_label_future_index",
]
