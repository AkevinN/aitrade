"""
CNN 选股（CNN Stock Screening）编排包。

对外导出数据模型与配置；编排逻辑（runner/universe/scoring/edge/store 等）
在各子模块中实现，按需导入以保持包导入轻量。
"""

from aitrade.screening.rules import DEFAULT_SCREENING_RULES, ScreeningRules
from aitrade.screening.store import ScreeningStore, build_screening_governance_store
from aitrade.screening.types import (
    LeaderboardRow,
    ScoreContribution,
    ScreeningResult,
    Tier1Score,
    Tier2Verdict,
)

__all__ = [
    "DEFAULT_SCREENING_RULES",
    "ScreeningRules",
    "ScreeningStore",
    "build_screening_governance_store",
    "ScoreContribution",
    "Tier1Score",
    "Tier2Verdict",
    "LeaderboardRow",
    "ScreeningResult",
]
