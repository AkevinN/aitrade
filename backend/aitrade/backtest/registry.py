"""
回测复用基础设施（迭代 3）：信号提供者协议 + 策略注册表 + 信号源注册表。

设计目标：新增一个量化方案 = 注册一个策略 + 提供一个信号源，**不改回测引擎**。
- SignalProvider：信号生产者统一接口，产出 [datetime, vt_symbol, signal] 表。
  CNN 推理、Alpha 模型、纯规则都可实现它，互相可替换。
- 策略注册表：name → BaseStrategy 子类，回测/实盘按名取用，参数走 setting 注入。
- 信号源注册表：name → SignalSourceFactory（params dict → 已绑定参数的 SignalProvider），
  回测与实盘按名取源，参数在构造期注入。param_spec 是给前端展示的轻量描述 dict，
  注册表不校验其内容。
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable
from collections.abc import Callable

import polars as pl

from .strategy import BaseStrategy


@runtime_checkable
class SignalProvider(Protocol):
    """信号生产者协议：产出 [datetime, vt_symbol, signal] 的 Polars DataFrame。"""

    def predict(
        self,
        start: date,
        end: date,
        on_progress: object | None = None,
    ) -> pl.DataFrame:
        ...


# ──────────────────────────────────────────────
# 策略注册表
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# 信号源注册表
# ──────────────────────────────────────────────

# params dict → 已绑定参数的 SignalProvider
SignalSourceFactory = Callable[[dict], SignalProvider]

# name → (factory, description, param_spec)
_SIGNAL_SOURCE_REGISTRY: dict[str, tuple[SignalSourceFactory, str, dict | None]] = {}


def register_signal_source(
    name: str,
    factory: SignalSourceFactory,
    *,
    description: str = "",
    param_spec: dict | None = None,
) -> None:
    """注册信号源工厂。重复注册同名将覆盖（便于热更新/测试）。

    Args:
        name: 非空字符串，信号源唯一标识。
        factory: 可调用对象，接受 params dict，返回实现 SignalProvider 的对象。
        description: 人读描述，供前端/日志展示。
        param_spec: 参数说明 dict（结构由调用方定义，注册表不校验），
                    如 ``{"model_name": {"type": "str", "required": True}}``.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("信号源名必须为非空字符串")
    if not callable(factory):
        raise TypeError(f"{factory!r} 必须是可调用对象")
    _SIGNAL_SOURCE_REGISTRY[name] = (factory, description, param_spec)


def build_signal_source(name: str, params: dict) -> SignalProvider:
    """按名构造信号源，未注册时抛 KeyError 并列出已注册名。"""
    if name not in _SIGNAL_SOURCE_REGISTRY:
        raise KeyError(
            f"未注册的信号源：{name}（已注册：{sorted(_SIGNAL_SOURCE_REGISTRY)}）"
        )
    factory, _desc, _spec = _SIGNAL_SOURCE_REGISTRY[name]
    return factory(params)


def list_signal_sources() -> list[dict]:
    """列出已注册信号源元信息，每项包含 name / description / param_spec。"""
    return [
        {"name": name, "description": desc, "param_spec": spec}
        for name, (_factory, desc, spec) in sorted(_SIGNAL_SOURCE_REGISTRY.items())
    ]
