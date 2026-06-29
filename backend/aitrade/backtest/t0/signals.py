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

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import polars as pl


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


@dataclass
class LabSignalProvider:
    """从**持久化模型信号帧**建 point-in-time 查询的 ``SignalProvider`` 实现。

    信号帧由 ``alpha`` 管线落盘（列含 ``datetime``/``vt_symbol``/信号值），经 ``lab.load_signal(name)``
    读取。本类把若干 ``{信号名: 帧}`` 预处理成 ``{(标的, 信号名): (升序日期, 对应值)}`` 索引，
    查询时二分定位「严格早于评估日（``date < day``）的最近一行」——保证滞后、无前视；
    同一日有多行（盘中）时取该日最后一行。

    Attributes:
        index: ``{(vt_symbol, signal_name): (sorted_days, values)}``，``sorted_days`` 升序。

    Example:
        >>> import polars as pl
        >>> from datetime import date, datetime
        >>> f = pl.DataFrame({"datetime": [datetime(2025, 1, 2)], "vt_symbol": ["A.SZSE"], "signal": [0.8]})
        >>> sp = LabSignalProvider.from_frames({"mom": f})
        >>> sp.value("A.SZSE", date(2025, 1, 3), "mom")
        0.8
    """

    index: dict[tuple[str, str], tuple[list[date], list[float]]] = field(default_factory=dict)

    @classmethod
    def from_frames(cls, frames: dict[str, "pl.DataFrame"], value_col: str = "signal") -> "LabSignalProvider":
        """从 ``{信号名: 信号帧}`` 预建查询索引。

        每帧按 ``(vt_symbol, 日期, 输入行序)`` 稳定排序后，按 ``(标的, 信号名)`` 收集升序日期与对应值；
        同一日期保留输入顺序，故「该日最后一行」即输入序最后一条。

        Args:
            frames: ``{信号名: polars DataFrame}``；帧需含 ``datetime``/``vt_symbol``/``value_col`` 列。
                值为 None 或空帧者跳过。
            value_col: 信号值列名，默认 ``"signal"``（与 ``alpha`` 落盘口径一致）。

        Returns:
            构建好的 LabSignalProvider。
        """
        import polars as pl

        index: dict[tuple[str, str], tuple[list[date], list[float]]] = {}
        for name, df in frames.items():
            if df is None or df.height == 0:
                continue
            with_idx = df.with_row_index("_i") if hasattr(pl.DataFrame, "with_row_index") else df.with_row_count("_i")
            date_expr = (pl.col("datetime") if df.schema["datetime"] == pl.Date
                         else pl.col("datetime").dt.date())
            ordered = (with_idx
                       .select([pl.col("vt_symbol"), date_expr.alias("_d"),
                                pl.col(value_col).alias("_v"), pl.col("_i")])
                       .sort(["vt_symbol", "_d", "_i"]))
            for row in ordered.iter_rows(named=True):
                bucket = index.setdefault((row["vt_symbol"], name), ([], []))
                bucket[0].append(row["_d"])
                bucket[1].append(row["_v"])
        return cls(index)

    def value(self, symbol: str, day: date, name: str) -> float | None:
        """返回该标的 ``date < day`` 的最近一行信号值；不可得返回 ``None``（point-in-time）。

        Args:
            symbol: 合约代码。
            day: 评估日（当日开盘时点）。
            name: 信号名。

        Returns:
            滞后的信号值；无更早行（或无该标的/信号）时返回 ``None``。
        """
        days, vals = self.index.get((symbol, name), ([], []))
        i = bisect_left(days, day) - 1
        return vals[i] if i >= 0 else None
