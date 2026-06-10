"""
标的画像（Symbol Profiling）只读诊断模块。

该模块在给定 (vt_symbol, interval, as_of, lookback) 的前提下，只读地计算标的的
工程特性画像（数据质量 / 流动性 / 波动性 / 可预测性），并据此产出与 Scheme 兼容的
方案建议草稿。模块严格遵守时间窗口隔离与只读无副作用约束，唯一允许写入的位置为
config.PROFILE_PATH。
"""

from aitrade.profiling.types import (
    GroupProfile,
    MetricBlock,
    MetricValue,
    ProfileInput,
    SchemeSuggestion,
    SuggestionItem,
    SymbolProfile,
)
from aitrade.profiling.profiler import Profiler
from aitrade.profiling.store import ProfileStore

__all__ = [
    "Profiler",
    "ProfileStore",
    "MetricValue",
    "MetricBlock",
    "GroupProfile",
    "ProfileInput",
    "SymbolProfile",
    "SuggestionItem",
    "SchemeSuggestion",
]
