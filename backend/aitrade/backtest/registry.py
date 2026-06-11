"""
回测复用基础设施（迭代 3）：信号提供者协议 + 策略注册表。

设计目标：新增一个量化方案 = 注册一个策略 + 提供一个信号源，**不改回测引擎**。
- SignalProvider：信号生产者统一接口，产出 [datetime, vt_symbol, signal] 表。
  CNN 推理、Alpha 模型、纯规则都可实现它，互相可替换。
- 策略注册表：name → BaseStrategy 子类，回测/实盘按名取用，参数走 setting 注入。
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Protocol, runtime_checkable

import polars as pl

from .strategy import BaseStrategy


@runtime_checkable
class SignalProvider(Protocol):
    """信号生产者协议：产出 [datetime, vt_symbol, signal] 的 Polars DataFrame。"""

    def predict(
        self,
        start: date,
        end: date,
        on_progress: Optional[object] = None,
    ) -> pl.DataFrame:
        ...


# name → 策略类
_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register_strategy(name: str, strategy_cls: type[BaseStrategy]) -> None:
    """注册策略类。重复注册同名将覆盖（便于热更新/测试）。"""
    if not isinstance(name, str) or not name:
        raise ValueError("策略名必须为非空字符串")
    if not (isinstance(strategy_cls, type) and issubclass(strategy_cls, BaseStrategy)):
        raise TypeError(f"{strategy_cls!r} 必须是 BaseStrategy 的子类")
    _STRATEGY_REGISTRY[name] = strategy_cls


def get_strategy(name: str) -> type[BaseStrategy]:
    """按名取策略类，未注册抛错。"""
    if name not in _STRATEGY_REGISTRY:
        raise KeyError(
            f"未注册的策略：{name}（已注册：{sorted(_STRATEGY_REGISTRY)}）"
        )
    return _STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """列出已注册策略名。"""
    return sorted(_STRATEGY_REGISTRY)
