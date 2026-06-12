"""
小市值轮动信号源（small_cap）。

设计要点
--------
- 信号定义：signal = -total_mv（万元原值取负）。
  市值越小 → signal 越高 → TopK 降序优先选入，量纲一致即可排序。
  docstring 特别注明：total_mv 单位为**万元**（tushare daily_basic 原始值），
  信号为其相反数，仅保证排序方向正确，不代表绝对量级。
- 市值数据来源：FundamentalStore（本地落盘，不调外部接口）。
  逐日查询：该日或最近一个 ≤ 该日的 total_mv，**严禁使用未来日市值**（无前视红线）。
- 宇宙：fundamental_store.list_symbols() ∩ lab 有行情。
  个别缺数据：logger.warning + 跳过（与 etf_momentum 语义一致）。
  全部缺数据：抛 RuntimeError 中文（提示刷新基本面与行情）。
- 过滤条件（逐日生效）：
  1. close < min_price → 剔除（低价股过滤）
  2. 近 20 日均成交额 < min_amount → 剔除（流动性过滤，不足 20 日取可得天数）
  3. exclude_st=True 时，合约名含 "ST"（含 *ST）→ 剔除
     合约名来自 _contracts 注入（{vt_symbol: {name, list_date}}）；
     若无 _contracts 注入则 warning "ST 过滤需要合约名称数据，当前未生效"（诚实降级）。
  4. list_date 距该日 < min_list_days → 剔除
     list_date 来自 _contracts 注入；缺失（None）保守保留。
     复用 rules/universe.py 的 filter_by_listing 纯函数。
- 趋势闸门联动说明：趋势检查由 PortfolioRiskManager 在调仓编排器层生效，
  本信号源**不做**趋势过滤，职责边界在此处截止。
- 不引入 torch（规则信号源红线）。
- top_k：信息用参数，真实持仓数在策略层（TopK rule）配置，此处不截断。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import polars as pl

from ..backtest.registry import SignalProvider, register_signal_source
from .universe import filter_by_listing

logger = logging.getLogger(__name__)

# 回看成交额的天数
_TURNOVER_LOOKBACK = 20


class _SmallCapSource:
    """小市值轮动信号源的私有实现，实现 SignalProvider 协议。"""

    def __init__(
        self,
        top_k: int,
        min_price: float,
        min_list_days: int,
        min_amount: float,
        exclude_st: bool,
        interval: str,
        lab: Any,
        fundamental_store: Any,
        contracts: dict[str, dict] | None,
    ) -> None:
        self._top_k = top_k  # 信息用：真实持仓数在策略层（TopK rule）配置，此处不截断
        self._min_price = min_price
        self._min_list_days = min_list_days
        self._min_amount = min_amount
        self._exclude_st = exclude_st
        self._interval = interval
        self._lab = lab
        self._fundamental_store = fundamental_store
        # contracts: {vt_symbol: {"name": str, "list_date": "YYYY-MM-DD" | None}}
        # None 表示未注入（仅支持 list_date 过滤降级 + ST 过滤降级）
        self._contracts = contracts

    def predict(
        self,
        start: date,
        end: date,
        on_progress: object | None = None,
    ) -> pl.DataFrame:
        """计算小市值信号，返回 [datetime, vt_symbol, signal] DataFrame。

        signal = -total_mv（万元原值取负），市值越小信号越高，对齐 TopK 降序消费。

        Args:
            start: 信号区间起始日（含）。
            end: 信号区间截止日（含）。
            on_progress: 可选进度回调。

        Returns:
            Polars DataFrame，列为 datetime(Datetime) / vt_symbol(Utf8) / signal(Float64)，
            按 (datetime, vt_symbol) 升序排序。

        Raises:
            RuntimeError: universe 中所有标的均无可用数据时抛出中文错误。
        """
        # ---- ST 过滤能力检查（诚实降级） ----
        if self._exclude_st and self._contracts is None:
            logger.warning(
                "small_cap：ST 过滤需要合约名称数据，当前未生效（未注入 _contracts）"
            )

        # ---- 宇宙：fundamental_store 有落盘的标的 ----
        fund_symbols = set(self._fundamental_store.list_symbols())
        if not fund_symbols:
            raise RuntimeError(
                "small_cap：基本面数据为空，请先在数据准备页下载基本面（fundamental）数据"
            )

        # ---- 预热：为成交额均值回看多加载 N 天数据 ----
        lookback_buffer = timedelta(days=_TURNOVER_LOOKBACK * 2)
        extended_start = start - lookback_buffer

        # ---- 逐标的加载行情 ----
        all_parts: list[pl.DataFrame] = []
        skipped: list[str] = []

        for vt_symbol in sorted(fund_symbols):
            bar_df = self._lab.load_bar_frame(
                vt_symbol,
                self._interval,
                extended_start,
                end,
                include_derived=True,
            )

            if bar_df is None or bar_df.is_empty():
                skipped.append(vt_symbol)
                logger.warning("small_cap：%s 无本地行情数据，已跳过", vt_symbol)
                continue

            # 保证按时间排序
            bar_df = bar_df.sort("datetime")

            # 裁出正式区间用于输出，保留预热数据用于均值计算
            start_dt = pl.lit(start).cast(pl.Date)
            end_dt = pl.lit(end).cast(pl.Date)

            # 全区间 bar（含预热）用于滚动计算
            full_df = bar_df

            # 正式区间 bar
            target_df = bar_df.filter(
                (pl.col("datetime").cast(pl.Date) >= start_dt)
                & (pl.col("datetime").cast(pl.Date) <= end_dt)
            )

            if target_df.is_empty():
                skipped.append(vt_symbol)
                logger.warning("small_cap：%s 在区间 [%s, %s] 内无行情，已跳过", vt_symbol, start, end)
                continue

            # ---- 加载该标的基本面数据（宽泛时间段，用于 as_of 查询） ----
            fund_df = self._fundamental_store.load(vt_symbol)

            part_rows: list[dict] = []

            for row in target_df.to_dicts():
                row_datetime = row["datetime"]
                # 兼容 datetime 对象与字符串
                if hasattr(row_datetime, "date"):
                    row_date: date = row_datetime.date()
                elif isinstance(row_datetime, date):
                    row_date = row_datetime
                else:
                    # fallback：字符串前缀
                    row_date = date.fromisoformat(str(row_datetime)[:10])

                # ---- 过滤 1：close < min_price ----
                close_val = float(row["close"])
                if close_val < self._min_price:
                    continue

                # ---- 过滤 2：近 20 日均成交额 < min_amount ----
                # 在 full_df 中取 <= row_date 的最近 _TURNOVER_LOOKBACK 行
                hist = full_df.filter(
                    pl.col("datetime").cast(pl.Date) <= pl.lit(row_date)
                ).tail(_TURNOVER_LOOKBACK)
                if hist.is_empty():
                    # 无历史数据，跳过
                    continue
                avg_turnover = hist["turnover"].mean()
                if avg_turnover is None or avg_turnover < self._min_amount:
                    continue

                # ---- 过滤 3：ST 过滤 ----
                if self._exclude_st and self._contracts is not None:
                    contract_info = self._contracts.get(vt_symbol)
                    if contract_info is not None:
                        name = str(contract_info.get("name", "") or "")
                        if "ST" in name.upper():
                            continue

                # ---- 过滤 4：上市天数（filter_by_listing 复用） ----
                list_date_val: date | None = None
                if self._contracts is not None:
                    contract_info2 = self._contracts.get(vt_symbol)
                    if contract_info2 is not None:
                        raw_ld = contract_info2.get("list_date")
                        if raw_ld is not None:
                            try:
                                if isinstance(raw_ld, date):
                                    list_date_val = raw_ld
                                else:
                                    list_date_val = date.fromisoformat(str(raw_ld)[:10])
                            except (ValueError, TypeError):
                                list_date_val = None

                if self._min_list_days > 0:
                    # 使用 filter_by_listing 纯函数（list_date 缺失时保守保留）
                    kept, _ = filter_by_listing(
                        [vt_symbol],
                        as_of=row_date,
                        list_dates={vt_symbol: list_date_val},
                        delist_dates={vt_symbol: None},
                        min_list_days=self._min_list_days,
                    )
                    if not kept:
                        continue

                # ---- 信号值：最近 ≤ row_date 的 total_mv 取负（无前视） ----
                total_mv = self._lookup_total_mv(fund_df, row_date)
                if total_mv is None:
                    # 无基本面数据，跳过（conservative）
                    logger.warning(
                        "small_cap：%s 在 %s 无可用 total_mv，已跳过该行", vt_symbol, row_date
                    )
                    continue

                signal_val = -total_mv

                part_rows.append(
                    {
                        "datetime": row_datetime,
                        "vt_symbol": vt_symbol,
                        "signal": signal_val,
                    }
                )

            if part_rows:
                part = pl.DataFrame(part_rows).select(
                    [
                        pl.col("datetime").cast(pl.Datetime),
                        pl.col("vt_symbol").cast(pl.Utf8),
                        pl.col("signal").cast(pl.Float64),
                    ]
                )
                all_parts.append(part)

        # ---- 全空检查 ----
        if not all_parts:
            raise RuntimeError(
                "small_cap：区间内无任何信号输出。"
                "请先在数据准备页下载基本面（fundamental）数据与股票日线行情，"
                "并确认过滤参数不过于严格。"
            )

        result = pl.concat(all_parts)
        result = result.sort(["datetime", "vt_symbol"])

        if on_progress is not None and callable(on_progress):
            on_progress(1.0, "small_cap 信号计算完成")  # type: ignore[operator]

        return result

    # -------------------------------------------------------------------------
    # 内部辅助
    # -------------------------------------------------------------------------

    @staticmethod
    def _lookup_total_mv(
        fund_df: pl.DataFrame | None,
        as_of: date,
    ) -> float | None:
        """在基本面历史中查找最近 ≤ as_of 日的 total_mv。

        严格使用 ≤ as_of 的最新一条记录，**禁止使用未来日（无前视红线）**。

        Returns:
            total_mv 值（万元）；无数据时返回 None。
        """
        if fund_df is None or fund_df.is_empty():
            return None

        # 过滤 <= as_of
        filtered = fund_df.filter(pl.col("datetime") <= pl.lit(as_of))
        if filtered.is_empty():
            return None

        # 取最新一行
        latest = filtered.sort("datetime").tail(1)
        raw = latest["total_mv"][0]
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None


def _build_small_cap_source(params: dict) -> SignalProvider:
    """工厂函数：从 params 构造 _SmallCapSource。

    Args:
        params: 参数字典，键值说明见 param_spec。
                可通过 ``params["_lab"]`` 注入 AlphaLab 实例（测试用途），
                ``params["_fundamental_store"]`` 注入 FundamentalStore（测试用途），
                ``params["_contracts"]`` 注入合约信息字典
                  {vt_symbol: {"name": str, "list_date": str | None}}（测试用途）。
                下划线前缀表示内部参数，param_spec 不展示。

    Raises:
        ValueError: 参数类型不合法。
    """
    top_k = int(params.get("top_k", 20))
    min_price = float(params.get("min_price", 2.0))
    min_list_days = int(params.get("min_list_days", 60))
    min_amount = float(params.get("min_amount", 3_000_000))
    exclude_st_raw = params.get("exclude_st", True)
    exclude_st = bool(exclude_st_raw)
    interval = str(params.get("interval", "d"))

    # AlphaLab 依赖注入
    lab = params.get("_lab")
    if lab is None:
        from ..alpha.lab import AlphaLab  # noqa: PLC0415  延迟 import 防循环
        from ..config import ALPHA_LAB_PATH  # noqa: PLC0415

        lab = AlphaLab(ALPHA_LAB_PATH)

    # FundamentalStore 依赖注入
    fundamental_store = params.get("_fundamental_store")
    if fundamental_store is None:
        from .store import FundamentalStore  # noqa: PLC0415

        fundamental_store = FundamentalStore()

    # contracts 注入（可选，用于 ST 过滤 + 上市天数过滤生产路径）
    contracts: dict[str, dict] | None = params.get("_contracts")

    return _SmallCapSource(
        top_k=top_k,
        min_price=min_price,
        min_list_days=min_list_days,
        min_amount=min_amount,
        exclude_st=exclude_st,
        interval=interval,
        lab=lab,
        fundamental_store=fundamental_store,
        contracts=contracts,
    )


# 自注册到共享信号源注册表（模块被 import 时执行，模式同 etf_momentum.py）
register_signal_source(
    "small_cap",
    _build_small_cap_source,
    description="小市值轮动（流动性/ST/上市天数过滤，周/月频建议）",
    param_spec={
        "top_k": {
            "type": "int",
            "required": False,
            "label": "持仓数量",
            "description": "信息用参数（真实持仓数在策略层配置），用于描述组合目标规模",
            "default": 20,
        },
        "min_price": {
            "type": "float",
            "required": False,
            "label": "最低收盘价（元）",
            "description": "低于此价格的股票不纳入（低价股过滤）",
            "default": 2.0,
        },
        "min_list_days": {
            "type": "int",
            "required": False,
            "label": "最小上市天数",
            "description": "上市天数不足此值的新股不纳入（需 _contracts 注入才生效）",
            "default": 60,
        },
        "min_amount": {
            "type": "float",
            "required": False,
            "label": "日均最低成交额（元）",
            "description": "近 20 日均成交额低于此值的标的不纳入（流动性过滤）",
            "default": 3_000_000,
        },
        "exclude_st": {
            "type": "bool",
            "required": False,
            "label": "剔除 ST 股",
            "description": "是否剔除名称含 ST（含 *ST）的股票（需 _contracts 注入才生效）",
            "default": True,
        },
        "interval": {
            "type": "str",
            "required": False,
            "label": "行情周期",
            "description": "数据加载周期（默认日线 'd'）",
            "default": "d",
        },
    },
)
