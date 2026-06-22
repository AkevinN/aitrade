"""
量化方案（Scheme）配置层（迭代 3）：把「一个量化方案」做成一条可持久化配置。

一个 Scheme = 预测器配置 + label 口径 + 策略名与参数 + 标的/周期/成本。
共享地基（引擎/撮合/盯市/统计/数据源/信号契约）保持不变；
新增方案 = 新增一条配置（+ 可选独立策略子类），方案之间互不污染、分支独立。

持久化：每个方案一份 JSON，存于 config.SCHEME_PATH。
运行：run_scheme_backtest 由方案解析 策略/成本/标的，复用共享回测引擎。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import polars as pl
from pydantic import BaseModel, Field

from ..config import SCHEME_PATH
from .artifacts import serialize_equity_curve, serialize_trades
from .engine import BacktestingEngine, BarDataLoader
from .registry import get_strategy


class CostConfig(BaseModel):
    """成本与制度约束（与回测引擎成本字段一一对应）。"""
    commission_rate: float = Field(default=0.0003, description="单边佣金率")
    stamp_duty: float = Field(default=0.0005, description="卖出印花税率")
    slippage: float = Field(default=0.0005, description="每笔不利滑点率")
    t_plus1: bool = Field(default=True, description="T+1 卖出限制；默认开启，贴近 A 股现实")


class PredictorConfig(BaseModel):
    """信号生产者配置。type 标识种类（cnn/alpha/rule...），params 为其参数。"""
    type: str = Field(description="预测器类型，如 cnn")
    params: dict[str, Any] = Field(default_factory=dict, description="预测器参数，如 {model: xxx}")


class StrategyConfig(BaseModel):
    """策略配置。name 为注册表键，params 走 setting 注入策略。"""
    name: str = Field(description="已注册的策略名，如 cnn_signal")
    params: dict[str, Any] = Field(default_factory=dict, description="策略参数（exit_mode/hold_days/阈值等）")


class Scheme(BaseModel):
    """一个完整的量化方案配置。"""
    name: str = Field(description="方案名（唯一标识，用作文件名）")
    description: str = Field(default="", description="方案说明")
    vt_symbols: list[str] = Field(default_factory=list, description="标的列表")
    interval: str = Field(default="d", description="K 线周期")
    capital: float = Field(default=1_000_000, description="初始资金")
    predictor: PredictorConfig
    strategy: StrategyConfig
    cost: CostConfig = Field(default_factory=CostConfig)
    label_spec: dict[str, Any] = Field(default_factory=dict, description="训练 label 口径（供一致性自检）")


class SchemeStore:
    """方案配置的 JSON 持久化（每方案一文件）。

    每个 Scheme 以 ``{name}.json`` 存储于 base_path 目录，
    文件名即方案名，便于直接查看与手动编辑。
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        """初始化持久化存储，并确保目录存在。

        Args:
            base_path: 方案 JSON 根目录；None 时使用 config.SCHEME_PATH。
        """
        self.base_path = Path(base_path) if base_path else SCHEME_PATH
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        """返回方案名对应的 JSON 文件路径。

        Args:
            name: 方案名（不含 .json 后缀）。

        Returns:
            完整文件路径 ``{base_path}/{name}.json``。
        """
        return self.base_path / f"{name}.json"

    def save(self, scheme: Scheme) -> Path:
        """将方案序列化为 JSON 并写入磁盘。

        Args:
            scheme: 待持久化的 Scheme 对象，其 name 用作文件名。

        Returns:
            写入的 JSON 文件绝对路径。
        """
        path = self._path(scheme.name)
        path.write_text(
            json.dumps(scheme.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def load(self, name: str) -> Scheme:
        """从磁盘读取方案 JSON 并反序列化为 Scheme 对象。

        Args:
            name: 方案名（不含 .json 后缀）。

        Returns:
            反序列化后的 Scheme 对象。

        Raises:
            FileNotFoundError: 方案文件不存在时抛出。
        """
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"方案不存在：{name}")
        return Scheme.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def list_names(self) -> list[str]:
        """返回所有已保存方案的名称列表（升序）。

        Returns:
            base_path 下所有 .json 文件的 stem（不含后缀）列表。
        """
        return sorted(p.stem for p in self.base_path.glob("*.json"))

    def delete(self, name: str) -> bool:
        """删除指定方案的 JSON 文件。

        Args:
            name: 方案名（不含 .json 后缀）。

        Returns:
            True 表示文件存在且已删除；False 表示文件不存在。
        """
        path = self._path(name)
        if path.exists():
            path.unlink()
            return True
        return False


def run_scheme_backtest(
    scheme: Scheme,
    data_loader: BarDataLoader,
    signal_df: pl.DataFrame,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """按方案配置驱动共享回测引擎。signal_df 由方案的预测器在外部产出后传入。

    关键点：本函数不含任何特定方案/模型的硬编码——策略按名从注册表取、成本从方案取，
    因此新增方案不需要改本函数，也不会污染其它方案。
    """
    strategy_cls = get_strategy(scheme.strategy.name)

    engine = BacktestingEngine(data_loader=data_loader)
    engine.set_parameters(
        vt_symbols=scheme.vt_symbols,
        interval=scheme.interval,
        start=start,
        end=end,
        capital=int(scheme.capital),
    )
    for vt_symbol in scheme.vt_symbols:
        if vt_symbol not in engine.sizes:
            engine.sizes[vt_symbol] = 1
            engine.priceticks[vt_symbol] = 0.01
        engine.long_rates[vt_symbol] = scheme.cost.commission_rate
        engine.short_rates[vt_symbol] = scheme.cost.commission_rate
        engine.stamp_duties[vt_symbol] = scheme.cost.stamp_duty
        engine.slippages[vt_symbol] = scheme.cost.slippage
    engine.t_plus1 = scheme.cost.t_plus1

    engine.add_strategy(strategy_cls, dict(scheme.strategy.params), signal_df)
    engine.load_data()
    engine.run_backtesting()

    daily_df = engine.calculate_result()
    if daily_df is None or engine.trade_count == 0:
        return {
            "scheme": scheme.name,
            "trade_count": 0,
            "statistics": {"total_trade_count": 0},
            # 字段恒在：无成交即空成交列表与空净值曲线
            "trades": [],
            "equity_curve": [],
        }

    statistics = engine.calculate_statistics()
    return {
        "scheme": scheme.name,
        "trade_count": engine.trade_count,
        "statistics": statistics,
        # 成交明细与逐日净值序列：equity_curve 必须在 calculate_statistics() 之后取
        # （此时 engine.daily_df 才补入 balance/drawdown 列）
        "trades": serialize_trades(engine.trades),
        "equity_curve": serialize_equity_curve(engine.daily_df),
    }
