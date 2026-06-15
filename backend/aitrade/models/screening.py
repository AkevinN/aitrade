"""
CNN 选股（CNN Stock Screening）API 请求模型。

本文件定义 ``CNNScreeningRequest``——启动一次跨标的分层漏斗选股的入口契约。
请求经 ``POST /api/cnn/screening/batch`` 接收后由 task_manager 异步调度
``ScreeningRunner.run``，最终产出 ``ScreeningResult``（见 screening/types.py）。

时间窗口隔离（红线）：``as_of`` 为必填字段，无任何隐式"全量"默认。
Tier-1 画像窗口与 Tier-2 评估区间均严格不越过 ``as_of``（Requirement 9.1）。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field

from aitrade.screening.rules import ConfidenceLevel, ObjectiveType


class CNNScreeningRequest(BaseModel):
    """CNN 选股批量任务请求参数。

    触发一次跨标的的分层漏斗选股：Tier-1 对候选池批量打 CNN 适配度综合分，
    Tier-2 对排名靠前的 top_k 只运行 WF/OOS 实证验证（可关闭）。

    时间窗口隔离（红线）：as_of 必填、无隐式"全量"默认；Tier-1 画像与
    Tier-2 评估区间均严格不越过 as_of（Requirement 9.1）。
    """

    name: str  # 本次选股任务名称，用于任务列表展示与产物归档
    interval: str = "d"  # K 线周期，默认日线；支持 "1m"/"30m"/"d" 等 AlphaLab 支持的周期
    as_of: datetime  # 评估截止时间，必填（Requirement 9.1）；Tier-1/Tier-2 数据均不超过此时点

    lookback_days: int = Field(gt=0)  # Tier-1 画像回看天数；必须 > 0（Requirement 1.2）

    # ---- Universe 过滤 ----
    # 交易所过滤：取 None 不过滤；取 "SSE"/"SZSE"/"BSE" 时仅保留对应交易所标的
    exchange: Optional[str] = None
    # 最小历史 bar 数：本地存档低于此值的标的被排除出 universe（Requirement 1.2）
    min_bar_count: int = Field(default=250, ge=1)
    # 显式候选池：非空时以此清单为 universe（仍做规范化与本地数据校验），忽略 exchange 过滤
    include_symbols: list[str] = Field(default_factory=list)
    # 强制排除清单：从最终 universe 中剔除（在 include_symbols 后生效）
    exclude_symbols: list[str] = Field(default_factory=list)

    # ---- 漏斗配置 ----
    # Tier-1 后入围 Tier-2 的最大标的数（Requirement 4.1）
    top_k: int = Field(default=15, ge=1)
    # 是否执行 Tier-2 WF/OOS 实证；False 时只产出 Tier-1 排名榜单（Requirement 4.3）
    run_tier2: bool = True
    # 入围 Tier-2 的最低综合置信度门槛；低于此等级的标的不入围（Requirement 2.5）
    min_confidence: ConfidenceLevel = "low"

    # ---- Tier-2 默认超参（偏快，可由 ScreeningRules 覆盖）----
    # CNN 训练目标（分类 / 回归 / 路径分类），供 Tier-2 构造 CNNWalkForwardRequest 使用
    objective: ObjectiveType = "classification"
    # Tier-2 评估区间起点（可选）；缺省时由 as_of 与 ScreeningRules 推导，强制 end <= as_of
    eval_start: Optional[date] = None

    # ---- 持久化 ----
    # 是否将 ScreeningResult 写入 SCREENING_PATH；False 时只在内存返回（Requirement 6.3）
    persist: bool = False
