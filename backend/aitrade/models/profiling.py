"""
Symbol Profiling API 请求 / 响应模型。
"""

from datetime import datetime

from pydantic import BaseModel, Field


class SymbolProfileRequest(BaseModel):
    """标的画像请求参数。

    画像为只读诊断：仅使用 ``as_of`` 之前的数据计算指标与建议。
    """

    vt_symbol: str  # 标的代码，如 '600030.SSE'
    interval: str  # 周期，如 '30m' / 'd'
    as_of: datetime  # 截止时间，必填、无隐式默认（Requirement 2.1）
    lookback_days: int = Field(gt=0)  # 回看天数，必为正（Requirement 12.3）
    observation_symbols: list[str] = Field(default_factory=list)  # 观察标的列表
    with_suggestion: bool = True  # 是否生成方案建议草稿
    persist: bool = False  # 是否持久化画像产物到 PROFILE_PATH
