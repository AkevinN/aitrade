"""做 T 条件挂单的信号提供器（SignalProvider）：为规则条件提供 point-in-time 信号值。

条件挂单规则（见 ``tick_policy.ConditionalTickPolicy``）除了能读 ``TickContext`` 里的
今开/昨收/历史，还可按名字读取「信号」——可以是 Alpha101/158 因子产物，也可以是任意自定义
算法的输出。本模块定义信号的统一读取口径：

- ``SignalProvider`` 协议：``value(symbol, day, name)`` 返回该标的在 ``day`` **当日开盘可用**的
  信号值，即只依赖 ``≤ day−1 收盘`` 的数据（滞后、无前视）；不可得返回 ``None``。
- ``DictSignalProvider``：v1 桩实现，从预先注入的字典读，便于测试与快速试算。
- ``AlphaFactorSignalProvider``：从既有 alpha 因子产物读取的扩展点，v1 留桩（返回 ``None``）。

无前视纪律：``SignalProvider`` 的实现方负责保证注入/读取的值是滞后的（point-in-time），
``DictSignalProvider`` 仅做纯字典查找，绝不跨键插值或回看未来键。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable


@runtime_checkable
class SignalProvider(Protocol):
    """point-in-time 信号读取协议：按 (标的, 日期, 信号名) 返回当日开盘可用的信号值。"""

    def value(self, symbol: str, day: date, name: str) -> float | None:
        """返回该标的在 ``day`` 当日开盘可用的 point-in-time 信号值。

        实现必须保证返回值只依赖 ``≤ day−1 收盘`` 的数据（滞后、无前视）；
        信号不可得（无该标的/日期/信号名）时返回 ``None``，由调用方的规则条件安全跳过。

        Args:
            symbol: 合约代码，如 ``"000415.SZSE"``。
            day: 评估日（当日开盘挂单时点）。
            name: 信号名，如 ``"alpha012"`` 或自定义算法名。

        Returns:
            滞后的信号浮点值；不可得时返回 ``None``。
        """
        ...


@dataclass
class DictSignalProvider:
    """v1 桩：从预先注入的 ``{(symbol, day, name): value}`` 字典读信号。

    仅做纯字典查找——查到返回注入值，查不到返回 ``None``，绝不跨键插值或回看其他日期。
    无前视由「注入端只填滞后值」保证；本类在结构上不可能读到未注入的未来值。

    Attributes:
        table: 键为 ``(标的, 日期, 信号名)``、值为该时点可用信号值的字典。

    Example:
        >>> from datetime import date
        >>> sp = DictSignalProvider({("000415.SZSE", date(2025, 1, 2), "mom"): 0.8})
        >>> sp.value("000415.SZSE", date(2025, 1, 2), "mom")
        0.8
        >>> sp.value("000415.SZSE", date(2025, 1, 3), "mom") is None
        True
    """

    table: dict[tuple[str, date, str], float] = field(default_factory=dict)

    def value(self, symbol: str, day: date, name: str) -> float | None:
        """纯字典查找：命中返回注入值，未命中返回 ``None``。

        Args:
            symbol: 合约代码。
            day: 评估日。
            name: 信号名。

        Returns:
            ``table[(symbol, day, name)]``，缺失时为 ``None``。
        """
        return self.table.get((symbol, day, name))


@dataclass
class AlphaFactorSignalProvider:
    """扩展点：从既有 Alpha101/158 因子产物读取信号（滞后一日），v1 留桩。

    与在线因子管线的完整打通（逐日因子产物的滞后读取、缺值对齐）记为后续工作，
    不阻塞 v1——v1 用 :class:`DictSignalProvider` 即可端到端验证条件挂单机制。
    桩实现对任何查询返回 ``None``（信号不可得），规则条件据此安全跳过，绝不引入前视。

    Attributes:
        source: 因子产物来源句柄（v1 未接线，占位）。

    Example:
        >>> from datetime import date
        >>> AlphaFactorSignalProvider().value("000415.SZSE", date(2025, 1, 2), "alpha012") is None
        True
    """

    source: object | None = None

    def value(self, symbol: str, day: date, name: str) -> float | None:
        """v1 桩：未接线，恒返回 ``None``（信号不可得）。

        Args:
            symbol: 合约代码。
            day: 评估日。
            name: 因子名，如 ``"alpha012"``。

        Returns:
            恒为 ``None``（v1 未打通因子产物读取）。
        """
        return None
