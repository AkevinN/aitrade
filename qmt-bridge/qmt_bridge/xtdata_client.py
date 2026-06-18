"""xtdata 封装模块：把 QMT 原始数据归一化为 aitrade 契约形状。

懒加载 xtquant，支持注入假对象以便脱机单测。所有"调 xtdata + 归一化 +
后缀/复权口径"逻辑集中在此，路由层只做序列化与鉴权。
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from .contract import (
    XTDATA_BAR_FIELDS,
    to_qmt_code,
    to_qmt_period,
    to_dividend_type,
)

# A 股交易时间按东八区
_CN_TZ = timezone(timedelta(hours=8))


def _ms_to_datetime(ms: int) -> datetime:
    """毫秒时间戳 -> 东八区 naive datetime（交易所本地时间）。

    Args:
        ms: Unix 毫秒时间戳，如 xtdata time 列的值。

    Returns:
        东八区对应的 naive datetime，不带 tzinfo。

    Example:
        >>> _ms_to_datetime(1704211200000)
        datetime.datetime(2024, 1, 2, 16, 0)
    """
    return datetime.fromtimestamp(int(ms) / 1000, tz=_CN_TZ).replace(tzinfo=None)


class XtdataClient:
    """对 xtquant.xtdata 的薄封装。注入 xtdata=None 时懒加载真模块。

    在 Windows 生产环境中不传 xtdata，首次调用 xt 属性时自动 import xtquant；
    在 Mac 单测环境中注入 FakeXtdata 实例，完全脱离 QMT 进程。

    Args:
        xtdata: 可注入的 xtdata 对象；None 时懒加载真 xtquant.xtdata。
        ratio_adjust: 为 True 时使用等比复权（front_ratio/back_ratio），默认 False。
    """

    def __init__(self, xtdata: Any = None, *, ratio_adjust: bool = False) -> None:
        self._xt = xtdata
        self._ratio = ratio_adjust

    @property
    def xt(self) -> Any:
        """懒加载真 xtquant.xtdata（Windows 上首次调用时）。

        Returns:
            已初始化的 xtdata 模块或注入的假对象。
        """
        if self._xt is None:
            from xtquant import xtdata  # type: ignore
            self._xt = xtdata
        return self._xt

    def get_bars(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: str,
        end: str,
        *,
        adjust_type: str = "hfq",
    ) -> list[dict]:
        """先 download 落地、再 get 读取，归一化为 BARS_COLUMNS 形状的 list[dict]。

        实现"两步取数"：先调 download_history_data2 确保本地缓存最新，
        再调 get_market_data_ex 读取并做列名映射与类型转换。

        Args:
            symbol: aitrade 股票代码，如 "600000"。
            exchange: aitrade 交易所标识，如 "SSE"/"SZSE"。
            interval: aitrade 内部周期，如 "d"/"1m"/"60m"。
            start: 起始时间，日线用 "YYYYMMDD"，分钟线用 14 位 "YYYYMMDDHHMMSS"。
            end: 结束时间，格式同 start。
            adjust_type: 复权口径，"none"/"qfq"/"hfq"，默认 "hfq"。

        Returns:
            list[dict]，每个 dict 含 BARS_COLUMNS 全部键，按时间升序排列。
            无数据（DataFrame 为空）时返回空列表 []。

        Example:
            >>> client = XtdataClient(xtdata=fake_xt)
            >>> rows = client.get_bars("600000", "SSE", "d", "20240101", "20240131", adjust_type="hfq")
            >>> rows[0]["close_price"]
            10.5
        """
        code = to_qmt_code(symbol, exchange)
        period = to_qmt_period(interval)
        dividend = to_dividend_type(adjust_type, ratio=self._ratio)

        self.xt.download_history_data2([code], period, start, end)
        data = self.xt.get_market_data_ex(
            XTDATA_BAR_FIELDS, [code], period=period, start_time=start,
            end_time=end, count=-1, dividend_type=dividend, fill_data=True,
        )
        df = data.get(code)
        if df is None or len(df) == 0:
            return []

        rows: list[dict] = []
        for rec in df.to_dict(orient="records"):
            rows.append({
                "symbol": symbol,
                "exchange": exchange,
                "datetime": _ms_to_datetime(rec["time"]),
                "interval": interval,
                "open_price": float(rec["open"]),
                "high_price": float(rec["high"]),
                "low_price": float(rec["low"]),
                "close_price": float(rec["close"]),
                "volume": float(rec.get("volume", 0.0)),
                "turnover": float(rec.get("amount", 0.0)),
                "open_interest": float(rec.get("openInterest", 0.0)),
                "adjust_type": adjust_type,
            })
        rows.sort(key=lambda r: r["datetime"])
        return rows

    def get_adj_factor(self, symbol: str, exchange: str,
                       start: str = "", end: str = "") -> list[dict]:
        """用 get_divid_factors 的 dr 累乘出后复权累积因子。

        每个除权日的复权因子 = 前一日累积因子 × 当日 dr，从 1.0 起单调递增。
        常用于将历史价格还原为后复权序列：price_hfq = price_raw × adj_factor。

        Args:
            symbol: aitrade 股票代码，如 "600000"。
            exchange: aitrade 交易所标识，如 "SSE"/"SZSE"。
            start: 起始日期，格式 "YYYYMMDD"；空字符串表示不限。
            end: 结束日期，格式 "YYYYMMDD"；空字符串表示不限。

        Returns:
            list[{'trade_date': 'YYYYMMDD', 'adj_factor': float}]，按除权日升序；
            adj_factor 从 1.0 起、每个除权日乘当日 dr，单调 >=1。无除权返回 []。

        Example:
            >>> client = XtdataClient(xtdata=fake_xt)
            >>> rows = client.get_adj_factor("600000", "SSE", "20240101", "20241231")
            >>> rows[0]
            {'trade_date': '20240110', 'adj_factor': 1.1}
        """
        code = to_qmt_code(symbol, exchange)
        df = self.xt.get_divid_factors(code, start, end)
        if df is None or len(df) == 0:
            return []

        out: list[dict] = []
        factor = 1.0
        for trade_date, rec in df.sort_index().iterrows():
            factor = round(factor * float(rec["dr"]), 6)
            out.append({"trade_date": str(trade_date), "adj_factor": factor})
        return out
