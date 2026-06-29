"""做 T 声明式档位策略配置与编译工厂（policy_spec）。

把前端可配置的「声明式档位策略」编译成既有 ``TickPolicy``/``Rule``。核心安全立场：

- **只认白名单枚举**（``kind``/``lhs``/``op`` 均为 ``Literal``），**后端自行构建 ``condition``/``ticks`` 回调**，
  **绝不执行前端传来的任意代码**——本模块代码路径中无 ``eval``/``exec``/``compile``/``__import__``。
- 任意自定义 Python 算法仍走代码/SDK 的 ``Rule(condition=…)``，**不在此声明式接口暴露**。
- **无前视**：所有左值（``gap``/``mean_range``/``momentum``/``signal``）只读 ``TickContext``（截至昨收 + 今开）。

声明式 DTO 用 Pydantic 表达（供 API 直接复用为请求体的判别联合），编译为既有的
``FixedTick``/``VolScaledTick``/``TrendTiltTick``/``ConditionalTickPolicy``，参数逐项透传、不改其行为。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from .tick_policy import (
    ConditionalTickPolicy, FixedTick, Rule, TickContext, TickPolicy,
    TrendTiltTick, VolScaledTick,
)

# —— 声明式 DTO ——————————————————————————————————————————————


class RuleCfg(BaseModel):
    """单条件规则声明：单个左值 ``op`` 阈值 → ``(sell_tick, buy_tick)``（命中即用该档）。

    Attributes:
        name: 规则名（报告/调试用，可空）。
        lhs: 左值来源，白名单枚举：``gap``（今开/昨收−1）/ ``mean_range``（近 window 日均振幅）/
            ``momentum``（近 window 日动量）/ ``signal``（命名持久化信号值）。
        op: 比较运算，白名单枚举 ``gt``/``ge``/``lt``/``le``。
        threshold: 阈值（``gap`` 为小数，如 0.003=0.3%；``signal`` 为原始信号值）。
        window: 回看日数，``mean_range``/``momentum`` 适用（其余忽略）；缺省按 20/5 兜底。
        signal_name: 信号名，``lhs="signal"`` 时必填。
        sell_tick: 命中时卖单挂高价差（元，>0）。
        buy_tick: 命中时买单挂低价差（元，>0）。
    """

    name: str = ""
    lhs: Literal["gap", "mean_range", "momentum", "signal"]
    op: Literal["gt", "ge", "lt", "le"]
    threshold: float
    window: int | None = Field(default=None, ge=1)
    signal_name: str | None = None
    sell_tick: float = Field(gt=0)
    buy_tick: float = Field(gt=0)


class FixedCfg(BaseModel):
    """固定档位策略声明（编译为 ``FixedTick``）。"""

    kind: Literal["fixed"] = "fixed"
    label: str
    sell_tick: float = Field(gt=0)
    buy_tick: float = Field(gt=0)


class VolScaledCfg(BaseModel):
    """波动缩放档位策略声明（编译为 ``VolScaledTick``）。"""

    kind: Literal["vol_scaled"] = "vol_scaled"
    label: str
    k: float = Field(gt=0)
    n: int = Field(default=20, ge=1)
    fallback: float = Field(default=0.02, gt=0)


class TrendTiltCfg(BaseModel):
    """趋势倾斜档位策略声明（编译为 ``TrendTiltTick``）。"""

    kind: Literal["trend_tilt"] = "trend_tilt"
    label: str
    base: float = Field(gt=0)
    tilt: float = Field(ge=0)
    n: int = Field(default=5, ge=1)


class ConditionalCfg(BaseModel):
    """条件规则档位策略声明（编译为 ``ConditionalTickPolicy``）。

    Attributes:
        label: 策略标签（报告用、一次请求内唯一）。
        rules: 有序单条件规则；按序首个命中即用其档位。
        default_sell_tick: 无规则命中时的卖档（元，>0）。
        default_buy_tick: 无规则命中时的买档（元，>0）。
        pricetick: 最小价位，用于对齐与下限（默认 0.01）。
    """

    kind: Literal["conditional"] = "conditional"
    label: str
    rules: list[RuleCfg] = Field(default_factory=list)
    default_sell_tick: float = Field(gt=0)
    default_buy_tick: float = Field(gt=0)
    pricetick: float = Field(default=0.01, gt=0)


TickPolicyCfg = Annotated[
    Union[FixedCfg, VolScaledCfg, TrendTiltCfg, ConditionalCfg],
    Field(discriminator="kind"),
]
"""档位策略声明的判别联合（判别键 ``kind``），供 API 请求体直接复用。"""


# —— 编译 ————————————————————————————————————————————————————

_OPS: dict[str, Callable[[float, float], bool]] = {
    "gt": lambda x, t: x > t,
    "ge": lambda x, t: x >= t,
    "lt": lambda x, t: x < t,
    "le": lambda x, t: x <= t,
}


def _lhs_value(ctx: TickContext, lhs: str, window: int | None, signal_name: str | None) -> float | None:
    """按白名单左值名从 ``ctx`` 取 point-in-time 值（无前视）；不可得返回 None。

    Args:
        ctx: 无前视上下文（截至昨收 + 今开）。
        lhs: 左值名，必须是 ``gap``/``mean_range``/``momentum``/``signal`` 之一。
        window: ``mean_range``/``momentum`` 的回看日数（缺省 20/5）。
        signal_name: ``lhs="signal"`` 时的信号名。

    Returns:
        左值浮点；历史不足或信号缺失时返回 None。

    Raises:
        ValueError: lhs 不在白名单内（双重保险，正常被 Pydantic 拦截）。
    """
    if lhs == "gap":
        return ctx.gap
    if lhs == "mean_range":
        return ctx.hist.mean_range(window or 20)
    if lhs == "momentum":
        return ctx.hist.momentum(window or 5)
    if lhs == "signal":
        return ctx.signals.get(signal_name) if signal_name is not None else None
    raise ValueError(f"未知左值 lhs={lhs!r}")


def _compile_rule(rc: RuleCfg) -> tuple[Rule, str | None]:
    """把单条件 ``RuleCfg`` 编译成 ``Rule``（后端构建回调，不执行用户代码）。

    Args:
        rc: 单条件规则声明。

    Returns:
        ``(Rule, signal_name | None)``：第二项为该规则引用的信号名（``lhs="signal"`` 时非空），
        供上层汇总到 ``signal_names``。
    """
    op = _OPS[rc.op]
    lhs, win, sig, thr = rc.lhs, rc.window, rc.signal_name, rc.threshold
    s, b = rc.sell_tick, rc.buy_tick

    def cond(ctx: TickContext, lhs=lhs, win=win, sig=sig, thr=thr, op=op) -> bool:
        x = _lhs_value(ctx, lhs, win, sig)
        return x is not None and op(x, thr)

    rule = Rule(rc.name or f"{lhs}{rc.op}{thr}", cond, lambda c, s=s, b=b: (s, b))
    return rule, (sig if lhs == "signal" else None)


def compile_tick_policy(cfg) -> tuple[str, TickPolicy, tuple[str, ...]]:
    """把声明式档位策略配置编译成 ``(label, TickPolicy, 引用的信号名集合)``。

    只认白名单 ``kind``；``conditional`` 把每条单条件规则编译成 ``Rule`` 并汇总其引用的信号名
    （供策略经 ``signal_names`` 预取）。基础策略参数逐项透传，不改其行为。

    Args:
        cfg: 一个 ``TickPolicyCfg``（``FixedCfg``/``VolScaledCfg``/``TrendTiltCfg``/``ConditionalCfg``）。

    Returns:
        ``(label, TickPolicy, signal_names)``。``signal_names`` 仅 ``conditional`` 含 signal 左值时非空。

    Raises:
        ValueError: ``cfg.kind`` 不在白名单内。

    Example:
        >>> label, pol, names = compile_tick_policy(FixedCfg(label="固定2分", sell_tick=0.02, buy_tick=0.02))
        >>> label
        '固定2分'
    """
    kind = cfg.kind
    if kind == "fixed":
        return cfg.label, FixedTick(cfg.sell_tick, cfg.buy_tick), ()
    if kind == "vol_scaled":
        return cfg.label, VolScaledTick(cfg.k, cfg.n, fallback=cfg.fallback), ()
    if kind == "trend_tilt":
        return cfg.label, TrendTiltTick(cfg.base, cfg.tilt, cfg.n), ()
    if kind == "conditional":
        compiled = [_compile_rule(r) for r in cfg.rules]
        names = tuple({n for _, n in compiled if n})
        pol = ConditionalTickPolicy(
            rules=[r for r, _ in compiled],
            default=(cfg.default_sell_tick, cfg.default_buy_tick),
            pricetick=cfg.pricetick,
            signal_names=names,
        )
        return cfg.label, pol, names
    raise ValueError(f"未知档位策略 kind={kind!r}")
