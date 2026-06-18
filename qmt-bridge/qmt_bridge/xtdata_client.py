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
    from_qmt_code,
    to_qmt_period,
    to_dividend_type,
    exchange_to_market,
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

    def get_contracts(self, *, include_bse: bool = False) -> list[dict]:
        """枚举沪深(可含北交所)A股并补详情，归一为 ContractInfo 形状的 list[dict]。

        先下载板块数据，再逐只查合约细节与类型，统一归一化为 aitrade 契约字段。
        产品类型根据 get_instrument_type 返回的 stock/index/fund 键推断。

        Args:
            include_bse: 为 True 时包含北交所（沪深京A股），默认 False（沪深A股）。

        Returns:
            list[dict]，每条含 symbol/exchange/name/product_type/size/pricetick/
            list_date/delist_date/extra 键。

        Example:
            >>> client = XtdataClient(xtdata=fake_xt)
            >>> rows = client.get_contracts(include_bse=False)
            >>> rows[0]["exchange"]
            'SSE'
        """
        self.xt.download_sector_data()
        sector = "沪深京A股" if include_bse else "沪深A股"
        codes = self.xt.get_stock_list_in_sector(sector) or []
        out: list[dict] = []
        for code in codes:
            symbol, exchange = from_qmt_code(code)
            detail = self.xt.get_instrument_detail(code, False) or {}
            types = self.xt.get_instrument_type(code) or {}
            product = "股票" if types.get("stock") else (
                "指数" if types.get("index") else ("基金" if types.get("fund") else ""))
            out.append({
                "symbol": symbol,
                "exchange": exchange,
                "name": detail.get("InstrumentName", ""),
                "product_type": product,
                "size": float(detail.get("VolumeMultiple", 1) or 1),
                "pricetick": float(detail.get("PriceTick", 0.01) or 0.01),
                "list_date": str(detail.get("OpenDate", "") or ""),
                "delist_date": str(detail.get("ExpireDate", "") or ""),
                "extra": {
                    "instrument_status": detail.get("InstrumentStatus"),
                    "is_trading": detail.get("IsTrading"),
                },
            })
        return out

    def get_trade_calendar(self, exchange: str, start: str, end: str) -> list[dict]:
        """交易日历，归一为 CalendarDay 形状的 list[dict]（仅含交易日，is_open=True）。

        先下载节假日数据，再查询指定市场的交易日列表并归一化。

        Args:
            exchange: aitrade 交易所标识，如 "SSE"/"SZSE"。
            start: 起始日期，格式 "YYYYMMDD"。
            end: 结束日期，格式 "YYYYMMDD"。

        Returns:
            list[dict]，每条含 date/exchange/is_open 键；is_open 恒为 True（只返回交易日）。

        Example:
            >>> client = XtdataClient(xtdata=fake_xt)
            >>> rows = client.get_trade_calendar("SSE", "20240101", "20240105")
            >>> rows[0]
            {'date': '20240102', 'exchange': 'SSE', 'is_open': True}
        """
        self.xt.download_holiday_data()
        market = exchange_to_market(exchange)
        days = self.xt.get_trading_calendar(market, start, end) or []
        return [{"date": str(d), "exchange": exchange, "is_open": True} for d in days]

    def get_fundamental(self, symbol: str, exchange: str, start: str, end: str) -> list[dict]:
        """财务数据，report_type 固定 announce_time 防未来函数。

        先 download 落地，再用 announce_time（按公告日）读取，展开所有表、
        所有行为扁平 list[dict]，字段中剔除 m_timetag/m_anntime（提升为顶层键）。

        Args:
            symbol: aitrade 股票代码，如 "600000"。
            exchange: aitrade 交易所标识，如 "SSE"/"SZSE"。
            start: 起始日期，格式 "YYYYMMDD"。
            end: 结束日期，格式 "YYYYMMDD"。

        Returns:
            list[dict]，每条形如
            {'symbol','exchange','table','report_period','ann_date','fields': {...}}；
            空表或 None 跳过。

        Example:
            >>> client = XtdataClient(xtdata=fake_xt)
            >>> rows = client.get_fundamental("600000", "SSE", "20230101", "20240401")
            >>> rows[0]["table"]
            'Balance'
        """
        code = to_qmt_code(symbol, exchange)
        self.xt.download_financial_data2([code], start_time=start, end_time=end)
        data = self.xt.get_financial_data([code], start_time=start, end_time=end,
                                          report_type="announce_time")
        tables = data.get(code, {})
        out: list[dict] = []
        for table_name, df in tables.items():
            if df is None or len(df) == 0:
                continue
            for rec in df.to_dict(orient="records"):
                fields = {k: v for k, v in rec.items() if k not in ("m_timetag", "m_anntime")}
                out.append({
                    "symbol": symbol,
                    "exchange": exchange,
                    "table": table_name,
                    "report_period": str(rec.get("m_timetag", "")),
                    "ann_date": str(rec.get("m_anntime", "")),
                    "fields": fields,
                })
        return out

    def is_connected(self) -> bool:
        """QMT/xtdata 连接是否在线（驱动 /health）。

        Returns:
            True 表示 QMT 客户端已连接；连接失败或抛异常均返回 False。

        Example:
            >>> client = XtdataClient(xtdata=fake_xt)
            >>> client.is_connected()
            False
        """
        try:
            return bool(self.xt.get_client().is_connected())
        except Exception:
            return False

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
