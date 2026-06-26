"""做 T 标的画像/标定器（T0Profiler）：交易前统计某标的的日内"偏离-回归"结构。

给定标定窗内的日线（O/H/L/C），按偏离开盘价的档位网格 ``x``（如 1..15 分），**分买/卖腿**
统计：成交率、条件回归边际收益（净于成本）、全日期望盈亏，并据此给出**天然非对称**的
建议买/卖档位。仅用标定窗内历史，**无前视**——调用方有责任保证标定窗严格早于评估窗
（design Property 6 / Requirement 6.3）。

核心结论是"**理想撮合前提**"下的统计画像：它假设"触价即成交"，最终档位仍须经
``Fill_Policy`` 网格的成交模型回测验证，不得仅凭画像下交易结论（Requirement 6.5）。

腿口径（x 为元，O/H/L/C 为当日开高低收）：
- 卖腿成交：``high >= O + x``；卖在 O+x、收盘买回，单边 PnL = ``(O+x) − C``。
- 买腿成交：``low  <= O − x``；买在 O−x、收盘卖出，单边 PnL = ``C − (O−x)``。
- 每腿"净于成本"边际收益 = 已成交日 PnL 均值 − 往返成本，再 ×100 表为"分"。
- 全日期望：每日 ``both·2x + onlyS·(x−ΔC) + onlyB·(x+ΔC)``（ΔC=C−O），全样本均值 ×100。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from .tick_policy import DailyHistory


@dataclass
class BandEdgeRow:
    """单个偏离档位 x（分）的逐腿画像行。

    Args:
        x_fen: 偏离开盘价的档位，单位"分"（1 分 = 0.01 元）。
        sell_fill: 卖腿成交率 ``P(high >= O + x)``，取值 [0, 1]。
        sell_edge_fen: 卖腿"净于成本"条件回归边际收益（分）；
            ``(mean[(O+x)−C | 成交] − 往返成本) × 100``。无成交日时为 0。
        buy_fill: 买腿成交率 ``P(low <= O − x)``，取值 [0, 1]。
        buy_edge_fen: 买腿"净于成本"条件回归边际收益（分）；
            ``(mean[C−(O−x) | 成交] − 往返成本) × 100``。无成交日时为 0。
        day_pnl_fen: 全日期望盈亏（分），全样本（含未触价日）均值 ×100。
    """

    x_fen: int
    sell_fill: float
    sell_edge_fen: float
    buy_fill: float
    buy_edge_fen: float
    day_pnl_fen: float

    def to_dict(self) -> dict:
        """序列化为可 JSON 落盘的 dict（字段与本 dataclass 同名）。"""
        return {
            "x_fen": self.x_fen,
            "sell_fill": self.sell_fill,
            "sell_edge_fen": self.sell_edge_fen,
            "buy_fill": self.buy_fill,
            "buy_edge_fen": self.buy_edge_fen,
            "day_pnl_fen": self.day_pnl_fen,
        }


@dataclass
class T0Profile:
    """做 T 画像：逐档位逐腿回归边际曲线 + 建议 (sell, buy) 档位 + 标定窗诊断。

    建议档位为"理想撮合前提"下的统计峰值，须经 ``Fill_Policy`` 成交网格回测验证后方可
    用于交易（``note`` 字段显式标注该前提）。可序列化落盘并作为 ``ReversionCalibratedTick``
    与人工选参的输入。

    Args:
        symbol: 标的合约代码，如 ``"000415.SZSE"``。
        window: 标定窗 ``(start, end)``（含），必须严格早于任何评估窗。
        rows: 逐档位画像行，按 ``x_fen`` 升序。
        suggested_sell_tick: 建议卖档（元），卖腿净 edge 峰值对应的偏离档。
        suggested_buy_tick: 建议买档（元），买腿净 edge 峰值对应的偏离档（天然可与卖档不等）。
        note: 前提/限制说明（默认标注"理想撮合前提，须经 Fill_Policy 网格验证"）。
        calib_mean_range: 标定窗内日均振幅 ``mean(high−low)``（元），作为 ``suggested_ticks``
            按近期振幅缩放的基准；无数据时为 None。
    """

    symbol: str
    window: tuple[date, date]
    rows: list[BandEdgeRow]
    suggested_sell_tick: float
    suggested_buy_tick: float
    note: str = "理想撮合前提；最终档位须经 Fill_Policy 网格的成交模型回测验证"
    calib_mean_range: float | None = None

    def suggested_ticks(self, scale: float | None = None) -> tuple[float, float]:
        """返回建议的 (sell_tick, buy_tick)，可按近期振幅相对标定窗均振幅缩放。

        画像在标定窗上钉出的是绝对分钱档位；当标的近期振幅放大/收缩时，按
        ``scale / calib_mean_range`` 等比缩放档位以适配当前波动量级（仍无前视——
        ``scale`` 由调用方用"截至前一日"的振幅传入）。

        Args:
            scale: 近期日均振幅（元）。为 None（或标定窗均振幅不可用/非正）时不缩放，
                原样返回建议档位。

        Returns:
            ``(sell_tick, buy_tick)``（元）。``scale`` 给定且基准有效时按比例缩放。

        Example:
            >>> prof.suggested_ticks(None)          # 原样
            (0.02, 0.04)
            >>> prof.suggested_ticks(0.20)          # 近期振幅是标定窗均振幅 0.10 的 2 倍
            (0.04, 0.08)
        """
        if scale is None or not self.calib_mean_range or self.calib_mean_range <= 0:
            return (self.suggested_sell_tick, self.suggested_buy_tick)
        factor = scale / self.calib_mean_range
        return (self.suggested_sell_tick * factor, self.suggested_buy_tick * factor)

    def to_dict(self) -> dict:
        """序列化为可 JSON 落盘的 dict（日期转 ISO 字符串，rows 逐行展开）。

        Returns:
            含 ``symbol/window/rows/suggested_sell_tick/suggested_buy_tick/note/
            calib_mean_range`` 的 dict；``window`` 为 ``[start_iso, end_iso]``。
        """
        return {
            "symbol": self.symbol,
            "window": [self.window[0].isoformat(), self.window[1].isoformat()],
            "rows": [r.to_dict() for r in self.rows],
            "suggested_sell_tick": self.suggested_sell_tick,
            "suggested_buy_tick": self.suggested_buy_tick,
            "note": self.note,
            "calib_mean_range": self.calib_mean_range,
        }


class T0Profiler:
    """做 T 画像器：在标定窗日线上算偏离-回归边际曲线并给出建议档位。

    纯统计、无副作用。调用方负责传入"严格早于评估窗"的标定窗日线，从而保证无前视
    与样本内/外隔离（design Property 6）。
    """

    def profile(
        self,
        symbol: str,
        daily: pl.DataFrame,
        x_grid_fen=range(1, 16),
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.0005,
    ) -> T0Profile:
        """在标定窗日线上逐档位逐腿统计成交率/条件回归边际收益/全日期望，给出建议档位。

        对每个 ``x = x_fen / 100``（元）：

        - ``sell_fill = mean(high >= O + x)``；卖腿单边 PnL = ``(O+x) − C``，
          ``sell_edge_fen = (mean[PnL | high>=O+x] − roundtrip_cost) × 100``。
        - ``buy_fill  = mean(low  <= O − x)``；买腿单边 PnL = ``C − (O−x)``，
          ``buy_edge_fen  = (mean[PnL | low<=O−x] − roundtrip_cost) × 100``。
        - ``roundtrip_cost ≈ (commission_rate*2 + stamp_duty) × C``（逐日近似的每股往返成本，
          复用与引擎同源的成本率口径）。
        - ``day_pnl_fen = mean(both?2x : onlyS?(x−ΔC) : onlyB?(x+ΔC) : 0) × 100``，
          其中 ``ΔC = C − O``、``both = high>=O+x 且 low<=O−x``、
          ``onlyS = high>=O+x 且 low>O−x``、``onlyB = low<=O−x 且 high<O+x``。

        建议档位：``suggested_sell_tick``/``suggested_buy_tick`` 取使**日均贡献 = 成交率×每笔均益**
        （``fill × edge_fen``）最大且 ``fill > 0.1`` 的 x——把成交次数算进来，避免偏向"很宽、单笔大
        但很少成交"的稀疏档（剔除小样本尾部，天然非对称）。无任一腿满足成交率门槛时，回退为网格最小档。

        无前视：只读传入的 ``daily``；调用方须保证其严格早于评估窗（Requirement 6.3）。

        Args:
            symbol: 标的合约代码，仅用于回填 ``T0Profile.symbol``。
            daily: 标定窗日线，含列 ``d``(date)/``open``/``high``/``low``/``close``，
                每个交易日一行、时间升序。
            x_grid_fen: 偏离档位网格（分），默认 ``range(1, 16)`` 即 1..15 分。
            commission_rate: 单边佣金率，默认 0.0003。
            stamp_duty: 卖出印花税率，默认 0.0005。

        Returns:
            ``T0Profile``，含逐档位 ``rows``、建议 ``(sell, buy)`` 档位与标定窗均振幅。

        Example:
            >>> prof = T0Profiler().profile("000415.SZSE", daily_df)
            >>> prof.suggested_buy_tick
            0.05
        """
        o = daily["open"].to_numpy()
        h = daily["high"].to_numpy()
        low = daily["low"].to_numpy()
        c = daily["close"].to_numpy()
        n = len(daily)

        d_vals = daily["d"].to_list()
        window = (d_vals[0], d_vals[-1]) if n else (date.min, date.min)
        calib_mean_range = float((h - low).mean()) if n else None

        roundtrip_cost = (commission_rate * 2 + stamp_duty) * c  # 逐日每股往返成本（元）
        dC = c - o  # ΔC = close − open

        rows: list[BandEdgeRow] = []
        for x_fen in x_grid_fen:
            x = x_fen / 100.0

            sell_hit = h >= o + x
            buy_hit = low <= o - x

            sell_fill = float(sell_hit.mean()) if n else 0.0
            buy_fill = float(buy_hit.mean()) if n else 0.0

            sell_edge_fen = self._leg_edge_fen((o + x) - c, roundtrip_cost, sell_hit)
            buy_edge_fen = self._leg_edge_fen(c - (o - x), roundtrip_cost, buy_hit)

            both = sell_hit & buy_hit
            only_s = sell_hit & ~buy_hit
            only_b = buy_hit & ~sell_hit
            day_pnl = (
                both * (2 * x)
                + only_s * (x - dC)
                + only_b * (x + dC)
            )
            day_pnl_fen = float(day_pnl.mean()) * 100 if n else 0.0

            rows.append(
                BandEdgeRow(
                    x_fen=int(x_fen),
                    sell_fill=sell_fill,
                    sell_edge_fen=sell_edge_fen,
                    buy_fill=buy_fill,
                    buy_edge_fen=buy_edge_fen,
                    day_pnl_fen=day_pnl_fen,
                )
            )

        suggested_sell_tick = self._best_tick(rows, leg="sell")
        suggested_buy_tick = self._best_tick(rows, leg="buy")

        return T0Profile(
            symbol=symbol,
            window=window,
            rows=rows,
            suggested_sell_tick=suggested_sell_tick,
            suggested_buy_tick=suggested_buy_tick,
            calib_mean_range=calib_mean_range,
        )

    @staticmethod
    def _leg_edge_fen(pnl, roundtrip_cost, hit) -> float:
        """单腿"净于成本"条件回归边际收益（分）。

        Args:
            pnl: 逐日单边 PnL（元）的向量。
            roundtrip_cost: 逐日每股往返成本（元）的向量。
            hit: 逐日是否成交的布尔向量（卖腿 high>=O+x / 买腿 low<=O−x）。

        Returns:
            ``(mean[pnl − cost | 成交]) × 100``；无成交日时返回 0.0。
        """
        if hit.sum() == 0:
            return 0.0
        net = (pnl - roundtrip_cost)[hit]
        return float(net.mean()) * 100

    @staticmethod
    def _best_tick(rows: list[BandEdgeRow], leg: str) -> float:
        """取某腿"日均贡献"峰值对应的偏离档（元），剔除成交率≤0.1 的小样本尾部。

        日均贡献 = ``成交率 × 每笔均益``（``fill × edge_fen``）——即把成交次数算进来：
        一个很宽、单笔均益大但很少成交的档位，总贡献其实很小，不应被选中。只看每笔均益会
        系统性偏向稀疏成交的宽档，故这里按频率加权。

        Args:
            rows: 逐档位画像行。
            leg: ``"sell"`` 或 ``"buy"``。

        Returns:
            建议档位（元）；无任何档位成交率 > 0.1 时回退为网格最小档。
        """
        fill_attr = f"{leg}_fill"
        edge_attr = f"{leg}_edge_fen"
        eligible = [r for r in rows if getattr(r, fill_attr) > 0.1]
        if not eligible:
            return min(r.x_fen for r in rows) / 100.0 if rows else 0.0
        best = max(eligible, key=lambda r: getattr(r, fill_attr) * getattr(r, edge_attr))
        return best.x_fen / 100.0


class ReversionCalibratedTick:
    """由 T0Profile 标定的档位策略（TickPolicy）：天然非对称、按近期振幅缩放。

    每日向 ``profile`` 询问建议档位，并按"截至前一日"的近 20 日均振幅相对标定窗均振幅
    缩放（无前视——只读 ``hist``，不读 ``day``）。历史不足 20 日时 ``mean_range`` 返回 None，
    退化为原样返回画像建议档位。

    Args:
        profile: 由 ``T0Profiler.profile`` 产出的画像（须严格早于评估窗标定）。
    """

    def __init__(self, profile: T0Profile) -> None:
        self.profile = profile

    def ticks_for(self, day: date, hist: DailyHistory) -> tuple[float, float]:
        """返回当日 (sell_tick, buy_tick)，按近 20 日均振幅缩放画像建议档位。

        Args:
            day: 当前交易日（仅满足 TickPolicy 协议签名，**不被读取**，以保证确定性/无前视）。
            hist: 截至前一交易日收盘的只读历史视图；用其 ``mean_range(20)`` 做缩放。

        Returns:
            ``profile.suggested_ticks(scale=hist.mean_range(20))``；
            历史不足 20 日时 ``mean_range`` 为 None，原样返回画像建议档位。
        """
        return self.profile.suggested_ticks(scale=hist.mean_range(20))


def load_daily_from_1m(parquet_path: str, start_year: int, end_year: int) -> pl.DataFrame:
    """把 1 分钟 bar parquet 聚合为日线 OHLC，限定在 [start_year, end_year] 年份内。

    按 ``datetime`` 的日期分组：open 取当日首笔、high 取最大、low 取最小、close 取末笔，
    并附一个 ``d`` 日期列。用于在为标的标定做 T 画像前准备日线输入。

    Args:
        parquet_path: 1m bar parquet 路径，至少含列
            ``datetime``/``open``/``high``/``low``/``close``/``volume``。
        start_year: 起始年份（含）。
        end_year: 结束年份（含）。

    Returns:
        含列 ``d``(date)/``open``/``high``/``low``/``close`` 的日线 DataFrame，按 ``d`` 升序，
        每个交易日一行。

    Example:
        >>> daily = load_daily_from_1m("bars/1m/000415.parquet", 2019, 2021)
    """
    lf = pl.scan_parquet(parquet_path)
    lf = lf.with_columns(pl.col("datetime").dt.date().alias("d"))
    lf = lf.filter(
        (pl.col("d").dt.year() >= start_year) & (pl.col("d").dt.year() <= end_year)
    )
    daily = (
        lf.sort("datetime")
        .group_by("d", maintain_order=True)
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
            ]
        )
        .sort("d")
        .collect()
    )
    return daily
