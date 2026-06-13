"""
Tushare / Tinyshare data source provider.

Fetches: contract lists, historical bars, trade calendars, fundamentals, adj factors.
Supports both tushare and tinyshare backends (switchable via TUSHARE_BACKEND env var).

Tushare permission requirements (credit-based):
  - 120 pts (free): stock list, daily bars, index daily, trade calendar, fundamentals
  - 2000 pts: minute bars, futures daily
  - 5000 pts: tick data
"""

from collections.abc import Callable
from datetime import datetime

from .base import BaseProvider
from .types import (
    DataCategory,
    ProviderStatus,
    ContractInfo,
    BarRecord,
    CalendarDay,
    FundamentalRecord,
)
from ..config import TUSHARE_TOKEN, TUSHARE_BACKEND

# Tushare exchange code -> vnpy exchange code
TS_EXCHANGE_MAP: dict[str, str] = {
    "SSE": "SSE",
    "SZSE": "SZSE",
    "BSE": "BSE",
    "CFFEX": "CFFEX",
    "CFX": "CFFEX",
    "SHFE": "SHFE",
    "SHF": "SHFE",
    "DCE": "DCE",
    "CZCE": "CZCE",
    "ZCE": "CZCE",
    "INE": "INE",
    "GFEX": "GFEX",
    "GFE": "GFEX",
}

# vnpy exchange -> Tushare suffix
VT_TO_TS_SUFFIX: dict[str, str] = {
    "SSE": "SH",
    "SZSE": "SZ",
    "CFFEX": "CFX",
    "SHFE": "SHF",
    "DCE": "DCE",
    "CZCE": "ZCE",
    "INE": "INE",
    "GFEX": "GFE",
}

# vnpy interval -> Tushare freq
INTERVAL_MAP: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "60min",
    "d": "D",
    "w": "W",
    "m": "M",
}


def _to_ts_code(symbol: str, exchange: str) -> str | None:
    """将项目内部合约代码转换为 Tushare 的 ts_code 格式（如 ``"600519.SH"``）。

    Args:
        symbol: 合约代码（不含交易所后缀），如 ``"600519"``。
        exchange: 项目标准交易所代码，如 ``"SSE"``。

    Returns:
        Tushare ts_code 字符串；不支持的交易所返回 None。
    """
    suffix = VT_TO_TS_SUFFIX.get(exchange)
    if not suffix:
        return None
    return f"{symbol}.{suffix}"


def _from_ts_code(ts_code: str) -> tuple[str, str] | None:
    """将 Tushare ts_code（如 ``"600519.SH"``）拆解为项目内部 (symbol, exchange)。

    Args:
        ts_code: Tushare 格式的合约代码，如 ``"000001.SZ"``。

    Returns:
        (symbol, vt_exchange) 元组，如 ``("000001", "SZSE")``；
        格式不合法或交易所后缀未知时返回 None。
    """
    parts = ts_code.split(".")
    if len(parts) != 2:
        return None
    symbol, suffix = parts
    for vt_ex, ts_suffix in VT_TO_TS_SUFFIX.items():
        if ts_suffix == suffix:
            return symbol, vt_ex
    return None


class TushareProvider(BaseProvider):
    """Tushare / Tinyshare 数据源，通过 TUSHARE_BACKEND 环境变量切换后端。

    支持：合约列表（股票/指数/期货/基金）、历史 K 线（日线/分钟线）、
    交易日历、基本面数据（PE/PB/市值）、复权因子。
    Tushare 权限要求：120 积分可用股票日线/日历/基本面；2000 积分可用分钟线。
    """

    name = "tushare"
    display_name = "Tushare 数据服务"
    description = "通过 Tushare/Tinyshare Pro API 获取合约列表、历史K线、交易日历、基本面数据"

    BACKEND_TUSHARE = "tushare"
    BACKEND_TINYSHARE = "tinyshare"

    def __init__(self) -> None:
        """初始化 Provider，从配置读取 token 与后端类型，不立即连接。"""
        self._pro = None
        self._ts = None
        self._inited = False
        self._token: str = TUSHARE_TOKEN
        self._backend: str = TUSHARE_BACKEND

    def _get_backend_module(self):
        """按 _backend 字段动态导入 tinyshare 或 tushare 模块。

        Returns:
            tinyshare 或 tushare 模块对象，供调用 pro_api/set_token/pro_bar。
        """
        if self._backend == self.BACKEND_TINYSHARE:
            import tinyshare as ts  # type: ignore
            return ts
        else:
            import tushare as ts  # type: ignore
            return ts

    def init(self, output: Callable = print) -> bool:
        """初始化 Tushare/Tinyshare Pro API 连接。

        Args:
            output: 日志输出函数，默认 print。

        Returns:
            True 表示连接成功；False 表示 token 为空或连接失败（软失败，不抛异常）。
        """
        if self._inited:
            return True

        if not self._token:
            output("TushareProvider: token 为空，无法初始化。请设置 TUSHARE_TOKEN 环境变量")
            return False

        try:
            self._ts = self._get_backend_module()
            try:
                self._ts.set_token(self._token)
            except Exception as e:
                output(f"TushareProvider: set_token 失败 ({e})，尝试直接使用 pro_api")
            self._pro = self._ts.pro_api(self._token)
            self._inited = True
            output(f"TushareProvider: 使用后端 [{self._backend}] 初始化成功")
            return True
        except Exception as e:
            output(f"TushareProvider: 初始化失败 - {e}")
            return False

    def _pro_bar(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        asset: str,
        freq: str,
    ):
        """封装 tushare/tinyshare 两种 pro_bar 调用差异（参数名不同）。

        tushare 使用 ``api=`` 参数；tinyshare 使用 ``pro_api_client=`` 参数。

        Args:
            ts_code: Tushare 合约代码，如 ``"600519.SH"``。
            start_date: 起始日期字符串，格式 YYYYMMDD。
            end_date: 截止日期字符串，格式 YYYYMMDD。
            asset: 资产类型（E=股票, I=指数, FT=期货, FD=基金）。
            freq: 行情频率，如 ``"D"``/``"W"``/``"60min"``。

        Returns:
            pandas DataFrame；失败时可能抛出异常。
        """
        if self._backend == self.BACKEND_TINYSHARE:
            return self._ts.pro_bar(
                ts_code=ts_code,
                pro_api_client=self._pro,
                start_date=start_date,
                end_date=end_date,
                asset=asset,
                freq=freq,
            )
        else:
            return self._ts.pro_bar(
                ts_code=ts_code,
                api=self._pro,
                start_date=start_date,
                end_date=end_date,
                asset=asset,
                freq=freq,
            )

    def _query_min_bar(
        self,
        ts_code: str,
        start: datetime,
        end: datetime,
        freq: str,
    ):
        """通过 stk_mins 接口查询分钟级 K 线（tinyshare 专用接口）。

        Args:
            ts_code: Tushare 合约代码。
            start: 起始时间（含）。
            end: 截止时间（含）。
            freq: 分钟频率，如 ``"1min"``/``"5min"``/``"60min"``。

        Returns:
            pandas DataFrame，包含 trade_time / open / high / low / close / vol 等列。
        """
        return self._pro.query(
            "stk_mins",
            ts_code=ts_code,
            start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
            freq=freq,
        )

    def get_status(self) -> ProviderStatus:
        """返回数据源当前可用状态，供上层判断能否取数。

        状态由初始化情况与 token 配置推导，不发起任何网络请求。

        Returns:
            未初始化且无 token 时返回 ``NOT_CONFIGURED``；未初始化但已配置
            token 时返回 ``UNAVAILABLE``；已成功初始化返回 ``AVAILABLE``。
        """
        if not self._inited:
            if not self._token:
                return ProviderStatus.NOT_CONFIGURED
            return ProviderStatus.UNAVAILABLE
        return ProviderStatus.AVAILABLE

    def get_supported_categories(self) -> list[DataCategory]:
        """声明本数据源支持的数据类别，供注册中心路由请求。

        Returns:
            固定包含合约、历史 K 线、交易日历、基本面、参考数据
            （含复权因子）五类的 DataCategory 列表。
        """
        return [
            DataCategory.CONTRACT,
            DataCategory.BAR_HISTORY,
            DataCategory.TRADE_CALENDAR,
            DataCategory.FUNDAMENTAL,
            DataCategory.REFERENCE,
        ]

    # ---- Contracts ----

    def get_contracts(
        self,
        product_type: str = "",
        exchange: str = "",
    ) -> list[ContractInfo] | None:
        """批量获取合约列表，按需覆盖股票/指数/期货/基金四类。

        product_type 为空时拉取全部四类并合并；否则仅拉取匹配类别。
        每类内部任一接口抛错都会导致整体返回 None（不做部分降级）。

        Args:
            product_type: 品种过滤，接受中英文别名（"股票"/"stock"、
                "指数"/"index"、"期货"/"futures"、"基金"/"fund"）；
                空字符串表示全部品种。
            exchange: 交易所过滤；空字符串表示全部交易所。

        Returns:
            合并后的 ContractInfo 列表；未初始化、请求异常或结果为空时返回 None。
        """
        if not self._inited:
            return None

        result: list[ContractInfo] = []

        try:
            if not product_type or product_type in ("股票", "stock"):
                result.extend(self._fetch_stock_contracts(exchange))
            if not product_type or product_type in ("指数", "index"):
                result.extend(self._fetch_index_contracts(exchange))
            if not product_type or product_type in ("期货", "futures"):
                result.extend(self._fetch_futures_contracts(exchange))
            if not product_type or product_type in ("基金", "fund"):
                result.extend(self._fetch_fund_contracts(exchange))
        except Exception:
            return None

        return result if result else None

    def get_contract(self, symbol: str, exchange: str) -> ContractInfo | None:
        """查询单只股票的合约基础信息（仅走 stock_basic 接口）。

        仅适用于股票；pricetick 固定填 0.01，product_type 固定填 "股票"。

        Args:
            symbol: 合约代码（不含交易所后缀），如 ``"600519"``。
            exchange: 项目标准交易所代码，如 ``"SSE"``。

        Returns:
            ContractInfo；未初始化、交易所不支持、查无此合约或请求异常时返回 None。
        """
        if not self._inited:
            return None

        ts_code = _to_ts_code(symbol, exchange)
        if not ts_code:
            return None

        try:
            df = self._pro.stock_basic(ts_code=ts_code, fields="ts_code,symbol,name,list_date")
            if df is not None and len(df) > 0:
                row = df.iloc[0]
                return ContractInfo(
                    symbol=symbol,
                    exchange=exchange,
                    name=row.get("name", ""),
                    product_type="股票",
                    pricetick=0.01,
                    list_date=str(row.get("list_date", "")),
                )
        except Exception:
            pass
        return None

    # ---- Historical bars ----

    def get_bar_history(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: datetime,
        end: datetime | None = None,
    ) -> list[BarRecord] | None:
        """获取单只标的的历史 K 线，自动按周期选择日/周线或分钟线接口。

        日线/周线走 pro_bar；其他周期走 stk_mins 分钟接口。资产类型由
        _detect_asset 自动推断。返回均为不复权价（adjust_type="none"），
        结果按时间升序排列，NaN 字段填 0。

        Args:
            symbol: 合约代码（不含交易所后缀），如 ``"600519"``。
            exchange: 项目标准交易所代码，如 ``"SSE"``。
            interval: K 线周期，需在 INTERVAL_MAP 内（"d"/"w"/"1m"/"5m"/
                "15m"/"30m"/"1h" 等）。
            start: 起始时间（含）。
            end: 截止时间（含）；None 表示取到当前时刻。

        Returns:
            按 datetime 升序排列的 BarRecord 列表；未初始化、交易所/周期
            不支持、请求异常或无数据时返回 None。
        """
        if not self._inited:
            return None

        ts_code = _to_ts_code(symbol, exchange)
        if not ts_code:
            return None

        ts_freq = INTERVAL_MAP.get(interval)
        if not ts_freq:
            return None

        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d") if end else datetime.now().strftime("%Y%m%d")
        asset = self._detect_asset(symbol, exchange)

        try:
            if interval in ("d", "w"):
                df = self._pro_bar(
                    ts_code=ts_code,
                    start_date=start_str,
                    end_date=end_str,
                    asset=asset,
                    freq=ts_freq,
                )
            else:
                df = self._query_min_bar(
                    ts_code=ts_code,
                    start=start,
                    end=end or datetime.now(),
                    freq=ts_freq,
                )
        except Exception:
            return None

        if df is None or len(df) == 0:
            return None

        df.fillna(0, inplace=True)
        records: list[BarRecord] = []

        for _, row in df.iterrows():
            if row.get("open") is None:
                continue

            if interval in ("d", "w"):
                dt_str = str(row.get("trade_date", ""))
                dt = datetime.strptime(dt_str, "%Y%m%d")
            else:
                dt_str = str(row.get("trade_time", ""))
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

            records.append(BarRecord(
                symbol=symbol,
                exchange=exchange,
                datetime=dt,
                interval=interval,
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                volume=float(row.get("vol", 0)),
                turnover=float(row.get("amount", 0)),
                open_interest=float(row.get("oi", 0)),
                adjust_type="none",  # pro_bar 未指定 adj，返回不复权价
            ))

        records.sort(key=lambda x: x.datetime)
        return records

    # ---- Trade calendar ----

    def get_trade_calendar(
        self,
        exchange: str,
        start: str,
        end: str,
    ) -> list[CalendarDay] | None:
        """获取指定交易所的交易日历（含开/休市标记与上一交易日）。

        会先把项目标准交易所代码反查为 Tushare 交易所代码再请求 trade_cal。

        Args:
            exchange: 项目标准交易所代码，如 ``"SSE"``。
            start: 起始日期字符串，格式 YYYYMMDD。
            end: 截止日期字符串，格式 YYYYMMDD。

        Returns:
            CalendarDay 列表（含 is_open 与 pre_trade_date）；未初始化、
            请求异常或无数据时返回 None。
        """
        if not self._inited:
            return None

        ts_exchange = exchange
        for ts_ex, vt_ex in TS_EXCHANGE_MAP.items():
            if vt_ex == exchange:
                ts_exchange = ts_ex
                break

        try:
            df = self._pro.trade_cal(
                exchange=ts_exchange,
                start_date=start,
                end_date=end,
            )
        except Exception:
            return None

        if df is None or len(df) == 0:
            return None

        return [
            CalendarDay(
                date=str(row["cal_date"]),
                exchange=exchange,
                is_open=bool(row["is_open"]),
                pre_trade_date=str(row.get("pretrade_date", "")),
            )
            for _, row in df.iterrows()
        ]

    # ---- Fundamentals ----

    def get_fundamental(
        self,
        symbol: str,
        exchange: str,
        start: str,
        end: str,
    ) -> list[FundamentalRecord] | None:
        """获取单只标的的每日基本面指标（PE/PB/PS/市值/换手率等）。

        数据来自 Tushare daily_basic 接口，各估值字段可能为 None（接口缺值时透传）。

        Args:
            symbol: 合约代码（不含交易所后缀），如 ``"600519"``。
            exchange: 项目标准交易所代码，如 ``"SSE"``。
            start: 起始日期字符串，格式 YYYYMMDD。
            end: 截止日期字符串，格式 YYYYMMDD。

        Returns:
            按交易日逐条的 FundamentalRecord 列表（估值字段缺失时为 None）；
            未初始化、交易所不支持、请求异常或无数据时返回 None。
        """
        if not self._inited:
            return None

        ts_code = _to_ts_code(symbol, exchange)
        if not ts_code:
            return None

        try:
            df = self._pro.daily_basic(
                ts_code=ts_code,
                start_date=start,
                end_date=end,
            )
        except Exception:
            return None

        if df is None or len(df) == 0:
            return None

        return [
            FundamentalRecord(
                symbol=symbol,
                exchange=exchange,
                trade_date=str(row["trade_date"]),
                pe=row.get("pe"),
                pe_ttm=row.get("pe_ttm"),
                pb=row.get("pb"),
                ps=row.get("ps"),
                total_mv=row.get("total_mv"),
                circ_mv=row.get("circ_mv"),
                turnover_rate=row.get("turnover_rate"),
                volume_ratio=row.get("volume_ratio"),
            )
            for _, row in df.iterrows()
        ]

    # ---- Adjustment factors ----

    def get_adj_factor(
        self,
        symbol: str,
        exchange: str,
        start: str = "",
        end: str = "",
    ) -> list[dict] | None:
        """获取单只标的的复权因子序列，用于前/后复权价计算。

        start/end 为空时不带日期参数，由 Tushare 返回该合约全部历史复权因子。

        Args:
            symbol: 合约代码（不含交易所后缀），如 ``"600519"``。
            exchange: 项目标准交易所代码，如 ``"SSE"``。
            start: 起始日期字符串（YYYYMMDD）；空字符串表示不限起始。
            end: 截止日期字符串（YYYYMMDD）；空字符串表示不限截止。

        Returns:
            形如 ``[{"trade_date": "20240101", "adj_factor": 1.23}, ...]`` 的列表；
            未初始化、交易所不支持、请求异常或无数据时返回 None。
        """
        if not self._inited:
            return None

        ts_code = _to_ts_code(symbol, exchange)
        if not ts_code:
            return None

        try:
            kwargs: dict = {"ts_code": ts_code}
            if start:
                kwargs["start_date"] = start
            if end:
                kwargs["end_date"] = end

            df = self._pro.adj_factor(**kwargs)
        except Exception:
            return None

        if df is None or len(df) == 0:
            return None

        return [
            {
                "trade_date": str(row["trade_date"]),
                "adj_factor": float(row["adj_factor"]),
            }
            for _, row in df.iterrows()
        ]

    # ---- Private helpers ----

    def _fetch_stock_contracts(self, exchange: str = "") -> list[ContractInfo]:
        """从 Tushare stock_basic 接口获取上市股票合约列表。

        Args:
            exchange: 交易所过滤；空字符串表示全部交易所。

        Returns:
            ContractInfo 列表（product_type="股票"）；请求失败或无数据返回空列表。
        """
        kwargs: dict = {"list_status": "L", "fields": "ts_code,symbol,name,list_date,delist_date,exchange"}
        if exchange:
            kwargs["exchange"] = exchange

        df = self._pro.stock_basic(**kwargs)
        if df is None or len(df) == 0:
            return []

        result = []
        for _, row in df.iterrows():
            parsed = _from_ts_code(row["ts_code"])
            if not parsed:
                continue
            sym, exc = parsed
            result.append(ContractInfo(
                symbol=sym,
                exchange=exc,
                name=row.get("name", ""),
                product_type="股票",
                pricetick=0.01,
                size=1,
                min_volume=100,
                list_date=str(row.get("list_date", "")),
                delist_date=str(row.get("delist_date", "")),
            ))
        return result

    def _fetch_index_contracts(self, exchange: str = "") -> list[ContractInfo]:
        """从 Tushare index_basic 接口获取指数合约列表。

        Args:
            exchange: 市场过滤（如 ``"SSE"``）；空字符串表示全部。

        Returns:
            ContractInfo 列表（product_type="指数"）；失败时返回空列表。
        """
        kwargs: dict = {"fields": "ts_code,name,market,publisher,category,base_date"}
        if exchange:
            kwargs["market"] = exchange

        df = self._pro.index_basic(**kwargs)
        if df is None or len(df) == 0:
            return []

        result = []
        for _, row in df.iterrows():
            parsed = _from_ts_code(row["ts_code"])
            if not parsed:
                continue
            sym, exc = parsed
            result.append(ContractInfo(
                symbol=sym,
                exchange=exc,
                name=row.get("name", ""),
                product_type="指数",
                pricetick=0.01,
                size=1,
            ))
        return result

    def _fetch_futures_contracts(self, exchange: str = "") -> list[ContractInfo]:
        """从 Tushare fut_basic 接口获取期货合约列表（按交易所逐一查询）。

        Args:
            exchange: 期货交易所代码（如 ``"SHFE"``/``"DCE"``）；空字符串获取全部。

        Returns:
            ContractInfo 列表（product_type="期货"，size 来自 multiplier 字段）；
            各交易所失败时跳过，不影响其他交易所。
        """
        exchanges = [exchange] if exchange else ["DCE", "SHFE", "CZCE", "CFFEX", "INE", "GFEX"]

        result = []
        for exc in exchanges:
            ts_exc = exc
            for ts_ex, vt_ex in TS_EXCHANGE_MAP.items():
                if vt_ex == exc:
                    ts_exc = ts_ex
                    break

            try:
                df = self._pro.fut_basic(
                    exchange=ts_exc,
                    fut_type="1",
                    fields="ts_code,symbol,exchange,name,multiplier,trade_unit,per_unit,list_date,delist_date",
                )
            except Exception:
                continue

            if df is None or len(df) == 0:
                continue

            for _, row in df.iterrows():
                parsed = _from_ts_code(row["ts_code"])
                if not parsed:
                    continue
                sym, vt_exc = parsed
                multiplier = row.get("multiplier")
                result.append(ContractInfo(
                    symbol=sym,
                    exchange=vt_exc,
                    name=row.get("name", ""),
                    product_type="期货",
                    size=float(multiplier) if multiplier else 1.0,
                    pricetick=0.01,
                    list_date=str(row.get("list_date", "")),
                    delist_date=str(row.get("delist_date", "")),
                ))
        return result

    def _fetch_fund_contracts(self, exchange: str = "") -> list[ContractInfo]:
        """从 Tushare fund_basic 接口获取场内基金合约列表（market="E"）。

        Args:
            exchange: 交易所过滤；空字符串表示全部。

        Returns:
            ContractInfo 列表（product_type="基金"）；失败时返回空列表。
        """
        try:
            df = self._pro.fund_basic(market="E", fields="ts_code,name,fund_type,found_date,list_date")
        except Exception:
            return []

        if df is None or len(df) == 0:
            return []

        result = []
        for _, row in df.iterrows():
            parsed = _from_ts_code(row["ts_code"])
            if not parsed:
                continue
            sym, exc = parsed
            if exchange and exc != exchange:
                continue
            result.append(ContractInfo(
                symbol=sym,
                exchange=exc,
                name=row.get("name", ""),
                product_type="基金",
                pricetick=0.001,
                size=1,
                min_volume=100,
                list_date=str(row.get("list_date", "")),
            ))
        return result

    def _detect_asset(self, symbol: str, exchange: str) -> str:
        """依据交易所与代码前缀推断 Tushare pro_bar 的 asset 参数。

        Args:
            symbol: 合约代码（不含交易所后缀）。
            exchange: 项目标准交易所代码。

        Returns:
            ``"E"``（股票）/``"I"``（指数）/``"FT"``（期货）/``"FD"``（基金）。
        """
        if exchange in ("CFFEX", "SHFE", "DCE", "CZCE", "INE", "GFEX"):
            return "FT"
        if exchange in ("SSE", "SZSE"):
            if symbol.startswith("5") and exchange == "SSE":
                return "FD"
            if symbol.startswith("1") and exchange == "SZSE":
                return "FD"
            if symbol.startswith("000") and exchange == "SSE":
                return "I"
            if symbol.startswith("399") and exchange == "SZSE":
                return "I"
            return "E"
        return "E"
