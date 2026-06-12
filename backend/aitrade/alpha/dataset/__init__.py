"""Alpha 数据集子包——公开核心构建块供外部直接引用。

导出 AlphaDataset（数据集模板）、Segment（区间枚举）、to_datetime（时间转换）
以及五种标准预处理器，隐藏内部实现细节。
"""

from .template import AlphaDataset
from .utility import Segment, to_datetime
from .processor import (
    process_drop_na,
    process_fill_na,
    process_cs_norm,
    process_robust_zscore_norm,
    process_cs_rank_norm
)


__all__ = [
    "AlphaDataset",
    "Segment",
    "to_datetime",
    "process_drop_na",
    "process_fill_na",
    "process_cs_norm",
    "process_robust_zscore_norm",
    "process_cs_rank_norm"
]
