"""
CNN 选股（CNN Stock Screening）编排包。

对外导出数据模型与配置；编排逻辑（runner/universe/scoring/edge/store 等）
在各子模块中实现，按需导入以保持包导入轻量。

注意：``ScreeningRunner`` / ``run_cnn_screening_batch`` 经 PEP 562 ``__getattr__``
**惰性导出**——它们所在的 ``runner`` 模块依赖 ``aitrade.models.screening``，
而后者又 import 本包的 ``rules``，若在包初始化时即 eager 导入 runner 会形成
循环导入。惰性导出使 ``from aitrade.screening import ScreeningRunner`` 仍可用，
但仅在真正访问该名字时才加载 runner，打破循环。
"""

from typing import TYPE_CHECKING, Any

from aitrade.screening.rules import DEFAULT_SCREENING_RULES, ScreeningRules
from aitrade.screening.store import ScreeningStore, build_screening_governance_store
from aitrade.screening.types import (
    LeaderboardRow,
    ScoreContribution,
    ScreeningResult,
    Tier1Score,
    Tier2Verdict,
)

if TYPE_CHECKING:  # 仅供类型检查器/IDE 解析，不在运行时触发导入
    from aitrade.screening.runner import ScreeningRunner, run_cnn_screening_batch

#: 惰性导出名 → 其所在子模块的映射（避免包初始化时的循环导入）。
_LAZY_EXPORTS = {
    "ScreeningRunner": "aitrade.screening.runner",
    "run_cnn_screening_batch": "aitrade.screening.runner",
}


def __getattr__(name: str) -> Any:
    """PEP 562 惰性属性解析：按需从 runner 子模块加载导出名。

    Args:
        name: 被访问的属性名。

    Returns:
        对应的导出对象（如 ``ScreeningRunner`` 类）。

    Raises:
        AttributeError: ``name`` 不是本包的已知导出名时抛出。
    """
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name)


__all__ = [
    "DEFAULT_SCREENING_RULES",
    "ScreeningRules",
    "ScreeningRunner",
    "run_cnn_screening_batch",
    "ScreeningStore",
    "build_screening_governance_store",
    "ScoreContribution",
    "Tier1Score",
    "Tier2Verdict",
    "LeaderboardRow",
    "ScreeningResult",
]
