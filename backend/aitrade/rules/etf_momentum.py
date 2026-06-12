"""
ETF 动量轮动信号源：基于绝对动量过滤的免训练规则信号源。

设计要点
--------
- 纯规则，无需训练；依赖 AlphaLab 本地行情，不调外部接口。
- 动量定义：``momentum = close / close.shift(lookback) - 1``（Polars 向量化）。
- 绝对动量过滤：momentum < min_momentum 的行不输出——全体过滤掉的日子整天无信号，
  下游 TopK 策略自然空仓（现金防御），这是设计核心而非缺陷。
- 预热：从 start 前移 ``lookback * 2.5`` 个自然日加载，保证区间首日有完整回看窗口。
- 某标的数据缺失/不足 lookback：静默跳过 + logger.warning；
  全部无数据时抛 RuntimeError 中文错误。
- torch 红线：本模块不引入 torch（天然满足，测试守护已存在）。
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Any

import polars as pl

from ..backtest.registry import SignalProvider, register_signal_source

logger = logging.getLogger(__name__)

# 默认 ETF 宇宙：沪深300 / 中证500 / 创业板 / 黄金 / 国债
_DEFAULT_UNIVERSE: list[str] = [
    "510300.SSE",
    "510500.SSE",
    "159915.SZSE",
    "518880.SSE",
    "511010.SSE",
]


class _EtfMomentumSource:
    """ETF 动量轮动信号源的私有实现，实现 SignalProvider 协议。"""

    def __init__(
        self,
        universe: list[str],
        lookback: int,
        min_momentum: float,
        interval: str,
        lab: Any,  # AlphaLab 实例（类型声明用 Any 避免循环/延迟 import 问题）
    ) -> None:
        self._universe = universe
        self._lookback = lookback
        self._min_momentum = min_momentum
        self._interval = interval
        self._lab = lab

    def predict(
        self,
        start: date,
        end: date,
        on_progress: object | None = None,
    ) -> pl.DataFrame:
        """计算 ETF 动量信号，返回 [datetime, vt_symbol, signal] DataFrame。

        Args:
            start: 信号区间起始日（含）。
            end: 信号区间截止日（含）。
            on_progress: 可选进度回调（本信号源轻量，仅在结束时调用一次）。

        Returns:
            Polars DataFrame，列为 datetime(Datetime) / vt_symbol(Utf8) / signal(Float64)，
            按 (datetime, vt_symbol) 升序排序。
            负动量或不足 lookback 的标的/日期不输出行。

        Raises:
            RuntimeError: universe 中所有标的均无本地行情时抛出。
        """
        # 预热回退：lookback 交易日 * 2.5 的日历裕量（参考 cnn/predictor warmup 思路）
        warmup_calendar_days = max(5, math.ceil(self._lookback * 2.5))
        extended_start: date = start - timedelta(days=warmup_calendar_days)

        # 逐标的加载、计算动量
        all_parts: list[pl.DataFrame] = []
        skipped: list[str] = []

        for vt_symbol in self._universe:
            df = self._lab.load_bar_frame(
                vt_symbol,
                self._interval,
                extended_start,
                end,
                include_derived=True,
            )

            # 数据缺失 → 静默跳过并记录
            if df is None or df.is_empty():
                skipped.append(vt_symbol)
                logger.warning("etf_momentum：%s 无本地行情数据，已跳过", vt_symbol)
                continue

            # 数据不足 lookback → 跳过
            if df.height < self._lookback + 1:
                skipped.append(vt_symbol)
                logger.warning(
                    "etf_momentum：%s 行情仅 %d 行，不足 lookback=%d，已跳过",
                    vt_symbol,
                    df.height,
                    self._lookback,
                )
                continue

            # 动量计算（Polars 向量化）
            df = df.sort("datetime").with_columns(
                (pl.col("close") / pl.col("close").shift(self._lookback) - 1.0).alias("momentum")
            )

            # 裁剪到 [start, end] 区间
            start_dt = pl.lit(start).cast(pl.Date)
            end_dt = pl.lit(end).cast(pl.Date)
            df = df.filter(
                (pl.col("datetime").cast(pl.Date) >= start_dt)
                & (pl.col("datetime").cast(pl.Date) <= end_dt)
            )

            # 丢弃动量为 null（预热期不足）的行
            df = df.drop_nulls(subset=["momentum"])

            if df.is_empty():
                # start 区间内没有有效数据，继续
                continue

            # 构造输出片段
            part = df.select([
                pl.col("datetime"),
                pl.lit(vt_symbol).alias("vt_symbol"),
                pl.col("momentum").alias("signal"),
            ])
            all_parts.append(part)

        # 所有标的均无数据 → 致命错误
        if not all_parts:
            raise RuntimeError(
                "universe 中所有标的均无本地行情，请先在数据准备页下载"
            )

        # 合并 + 绝对动量过滤
        result = pl.concat(all_parts)
        result = result.filter(pl.col("signal") >= self._min_momentum)

        # 按 (datetime, vt_symbol) 排序
        result = result.sort(["datetime", "vt_symbol"])

        if on_progress is not None and callable(on_progress):
            on_progress(1.0, "etf_momentum 信号计算完成")  # type: ignore[operator]

        return result


def _build_etf_momentum_source(params: dict) -> SignalProvider:
    """工厂函数：从 params 构造 _EtfMomentumSource。

    Args:
        params: 参数字典，键值说明见 param_spec。
                可通过 ``params["_lab"]`` 注入 AlphaLab 实例（测试用途，
                下划线前缀表示内部参数，param_spec 不展示）。

    Raises:
        ValueError: 参数类型不合法或 lookback < 1。
    """
    # universe
    universe: list[str] = params.get("universe", list(_DEFAULT_UNIVERSE))
    if not isinstance(universe, list):
        raise ValueError("universe 必须是字符串列表")
    if not all(isinstance(s, str) for s in universe):
        raise ValueError("universe 中每个元素必须是字符串")

    # lookback
    lookback_raw = params.get("lookback", 20)
    if not isinstance(lookback_raw, int):
        raise ValueError(f"lookback 必须是整数，收到 {type(lookback_raw).__name__}")
    lookback: int = lookback_raw
    if lookback < 1:
        raise ValueError(f"lookback 必须 >= 1，收到 {lookback}")

    # min_momentum
    min_momentum_raw = params.get("min_momentum", 0.0)
    if not isinstance(min_momentum_raw, (int, float)):
        raise ValueError(f"min_momentum 必须是数值，收到 {type(min_momentum_raw).__name__}")
    min_momentum: float = float(min_momentum_raw)

    # interval
    interval_raw = params.get("interval", "d")
    if not isinstance(interval_raw, str):
        raise ValueError(f"interval 必须是字符串，收到 {type(interval_raw).__name__}")
    interval: str = interval_raw

    # AlphaLab 依赖：支持测试注入（_lab），否则从项目路径构造
    lab = params.get("_lab")
    if lab is None:
        from ..alpha.lab import AlphaLab  # noqa: PLC0415  延迟 import 防循环
        from ..config import ALPHA_LAB_PATH  # noqa: PLC0415

        lab = AlphaLab(ALPHA_LAB_PATH)

    return _EtfMomentumSource(
        universe=universe,
        lookback=lookback,
        min_momentum=min_momentum,
        interval=interval,
        lab=lab,
    )


# 自注册到共享信号源注册表（模块被 import 时执行，模式同 cnn_adapter.py）
register_signal_source(
    "etf_momentum",
    _build_etf_momentum_source,
    description="ETF 动量轮动信号（绝对动量过滤，负动量日空仓防御）",
    param_spec={
        "universe": {
            "type": "list[str]",
            "required": False,
            "label": "ETF 宇宙",
            "description": "参与轮动的 ETF vt_symbol 列表",
            "default": _DEFAULT_UNIVERSE,
        },
        "lookback": {
            "type": "int",
            "required": False,
            "label": "动量回看天数",
            "description": "动量计算回看交易日数（>= 1）",
            "default": 20,
        },
        "min_momentum": {
            "type": "float",
            "required": False,
            "label": "绝对动量门槛",
            "description": "低于此动量的标的/日期不输出信号（空仓防御）",
            "default": 0.0,
        },
        "interval": {
            "type": "str",
            "required": False,
            "label": "行情周期",
            "description": "数据加载周期（如 'd'、'1m'）",
            "default": "d",
        },
    },
)
