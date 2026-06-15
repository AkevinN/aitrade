"""
交易操作台（Trading Console）请求模型。

把前端配置表单映射为编排器所需的请求对象，并提供到既有领域对象
（`RiskConfig` / `PortfolioSnapshot`）的映射辅助。本层不含决策逻辑，
仅做校验与字段转换。
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ..alpha.lab_utils import normalize_vt_symbol
from ..live.decision_instant import SUPPORTED_BAR_FREQS
from ..live.risk import RiskConfig
from ..live.signal_service import PortfolioSnapshot


class PortfolioSnapshotRequest(BaseModel):
    """组合快照（与既有 PortfolioSnapshot 字段一一对应）。"""
    portfolio_value: float = Field(description="组合总市值（现金+持仓）")
    total_position_value: float = Field(default=0.0, description="当前总持仓市值")
    current_position: int = Field(default=0, description="目标标的当前持仓股数")
    current_symbol_value: float = Field(default=0.0, description="目标标的当前持仓市值")

    def to_domain(self) -> PortfolioSnapshot:
        """把请求模型逐字段映射为既有 PortfolioSnapshot 领域对象。

        纯字段拷贝，不做任何校验或换算；供编排器在收到前端请求后转换使用。

        Returns:
            字段与本请求一一对应的 PortfolioSnapshot 实例。
        """
        return PortfolioSnapshot(
            portfolio_value=self.portfolio_value,
            total_position_value=self.total_position_value,
            current_position=self.current_position,
            current_symbol_value=self.current_symbol_value,
        )


class RiskConfigRequest(BaseModel):
    """风控配置（映射 RiskConfig）。"""
    blacklist: list[str] = Field(default_factory=list, description="禁止买入的标的列表")
    max_total_position_ratio: float = Field(default=0.95, description="总持仓市值 / 组合市值 上限")
    max_single_position_ratio: float = Field(default=0.30, description="单票市值 / 组合市值 上限")
    allow_when_halted: bool = Field(default=False, description="停牌/涨跌停封死时是否允许交易")

    def to_domain(self) -> RiskConfig:
        """把请求模型映射为既有 RiskConfig 领域对象。

        其余字段直传，仅 blacklist 做转换：逐个经 normalize_vt_symbol 归一化后
        由 list[str] 去重为 set[str]，便于风控层做 O(1) 黑名单命中判断。

        Returns:
            字段对应本请求、blacklist 已归一化去重的 RiskConfig 实例。
        """
        return RiskConfig(
            blacklist={normalize_vt_symbol(symbol) for symbol in self.blacklist},
            max_total_position_ratio=self.max_total_position_ratio,
            max_single_position_ratio=self.max_single_position_ratio,
            allow_when_halted=self.allow_when_halted,
        )


class LiveDecisionRequest(BaseModel):
    """触发一次今日决策的请求体。"""
    model: str = Field(description="CNN 模型名（必填）")
    vt_symbol: str = Field(description="目标标的（必填）")
    scheme: str = Field(description="方案名（必填）")
    as_of: Optional[datetime] = Field(default=None, description="决策时刻，缺省=当前；仅 close_time<=as_of 的 bar 可见")
    bar_freq: str = Field(
        default="1d",
        description="决策 bar 频率（须与所选模型训练间隔一致）；1d=日频，分钟频=盘中逐 bar 决策",
    )
    data_source: Literal["upload", "pull"] = Field(default="pull", description="数据源")
    portfolio: PortfolioSnapshotRequest
    risk: RiskConfigRequest = Field(default_factory=RiskConfigRequest)
    buy_threshold: float = Field(default=0.6, description="买入阈值")
    position_ratio: float = Field(default=0.95, description="目标仓位比例")
    min_volume: int = Field(default=100, description="最小成交手数")
    model_version: str = Field(default="", description="模型版本，参与 signal_id")
    halted: bool = Field(default=False, description="目标标的当日是否停牌/封死")
    should_exit: bool = Field(default=False, description="是否触发出场，见“出场逻辑”")

    @field_validator("bar_freq")
    @classmethod
    def _valid_bar_freq(cls, v: str) -> str:
        """pydantic 校验器：拒绝不在 SUPPORTED_BAR_FREQS 内的 bar 频率。

        Args:
            v: 待校验的 bar_freq 取值，如 "1d"/"30m"。

        Returns:
            原样返回校验通过的 v（不做归一化）。

        Raises:
            ValueError: v 不在 SUPPORTED_BAR_FREQS 列表内时抛出（pydantic 转为校验错误）。
        """
        if v not in SUPPORTED_BAR_FREQS:
            raise ValueError(f"bar_freq 仅支持 {SUPPORTED_BAR_FREQS}：{v!r}")
        return v


class RebalanceRequest(BaseModel):
    """手动触发一次组合调仓决策的请求体。

    优先使用 plan_id 引用已有 rule 计划（展开参数），也可内联传参。
    plan_id 与内联参数二选一：传入 plan_id 时以计划配置为准；否则使用内联字段。
    """

    # -- plan_id 引用模式（从 _plan_store 取 rule 计划展开参数）--
    plan_id: str | None = Field(
        default=None,
        description="已存在的 rule 计划 ID；传入时内联字段被忽略",
    )

    # -- 内联模式字段（plan_id 缺失时使用）--
    plan_name: str = Field(default="", description="计划名（用于 scheme 命名空间）")
    signal_source: str = Field(default="", description="注册表信号源名（如 'etf_momentum'）")
    signal_params: dict[str, Any] = Field(default_factory=dict, description="信号参数")
    strategy_params: dict[str, Any] = Field(
        default_factory=lambda: {"top_k": 5},
        description="组合选股参数（top_k 等）",
    )
    portfolio_id: str = Field(default="", description="持仓账本 ID")
    capital: float = Field(default=1_000_000.0, description="组合目标市值（计算目标仓位用）")
    as_of: datetime | None = Field(default=None, description="决策时刻，缺省=当前")
    bar_freq: str = Field(default="1d", description="决策 bar 频率")

    @field_validator("bar_freq")
    @classmethod
    def _valid_bar_freq(cls, v: str) -> str:
        """pydantic 校验器：拒绝不在 SUPPORTED_BAR_FREQS 内的 bar 频率。

        Args:
            v: 待校验的 bar_freq 取值，如 "1d"/"30m"。

        Returns:
            原样返回校验通过的 v（不做归一化）。

        Raises:
            ValueError: v 不在 SUPPORTED_BAR_FREQS 列表内时抛出（pydantic 转为校验错误）。
        """
        if v not in SUPPORTED_BAR_FREQS:
            raise ValueError(f"bar_freq 仅支持 {SUPPORTED_BAR_FREQS}：{v!r}")
        return v
