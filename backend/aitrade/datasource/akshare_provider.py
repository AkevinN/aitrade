"""
AKShare 数据源 Provider。

用于接入开源免费的 A 股历史 K 线数据（无需 token）。
底层通过 AKShare 调用东方财富接口：
  - stock_zh_a_hist          -> 日线 / 周线 / 月线
  - stock_zh_a_hist_min_em   -> 分钟线（1/5/15/30/60），其中 1 分钟仅支持近 5 个交易日
  - bond_zh_hs_cov_daily     -> 可转债日线（date/open/high/low/close/volume）

适配器职责：
  - 将项目内多种证券代码写法（000415.SZSE / sz000415 / 000415.SZ 等）统一为 AKShare 所需纯代码
  - 转债代码（110/111/113/118.SSE 与 123/127/128.SZSE）分派到 bond_zh_hs_cov_daily
  - 网络异常时自动重试，并将失败原因显式抛出，避免误报为「无数据」

字段与周期映射详见 docs/akshare对接参考.md。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime

from .base import BaseProvider
from .types import (
    BarRecord,
    ContractInfo,
    DataCategory,
    ProviderStatus,
)
from ..config import AKSHARE_ENABLED, AKSHARE_MAX_RETRIES, AKSHARE_RETRY_DELAY_SEC

logger = logging.getLogger(__name__)

# 东方财富 A 股接口支持的交易所
AK_SUPPORTED_EXCHANGES: set[str] = {"SSE", "SZSE", "BSE"}

# 转债代码前缀（6位代码的前3位）→ 交易所前缀（用于 bond_zh_hs_cov_daily 的 symbol 参数）
# SSE（上交所）转债：110xxx / 111xxx / 113xxx / 118xxx
# SZSE（深交所）转债：123xxx / 127xxx / 128xxx
_CB_SSE_PREFIXES: frozenset[str] = frozenset(["110", "111", "113", "118"])
_CB_SZSE_PREFIXES: frozenset[str] = frozenset(["123", "127", "128"])
# 所有转债代码前缀的集合（用于快速判断）
_CB_ALL_PREFIXES: frozenset[str] = _CB_SSE_PREFIXES | _CB_SZSE_PREFIXES

# 交易所别名 -> 项目标准写法
_EXCHANGE_ALIASES: dict[str, str] = {
    "SZ": "SZSE",
    "SZSE": "SZSE",
    "SH": "SSE",
    "SSE": "SSE",
    "BJ": "BSE",
    "BSE": "BSE",
}

# 前缀写法 -> 交易所
_PREFIX_EXCHANGES: dict[str, str] = {
    "sz": "SZSE",
    "sh": "SSE",
    "bj": "BSE",
}

# 项目内部周期 -> stock_zh_a_hist 的 period 参数
HIST_PERIOD_MAP: dict[str, str] = {
    "d": "daily",
    "w": "weekly",
    "m": "monthly",
}

# 项目标准交易所 -> 新浪 symbol 前缀
_SINA_EXCHANGE_PREFIX: dict[str, str] = {
    "SSE": "sh",
    "SZSE": "sz",
    "BSE": "bj",
}

# 项目内部周期 -> 分钟线 period 参数（东财/新浪通用）
MIN_PERIOD_MAP: dict[str, str] = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "60m": "60",
}


class AkshareProviderError(RuntimeError):
    """AKShare 数据源可恢复/可诊断错误，供上层展示给用户。"""


class AkshareProvider(BaseProvider):
    """AKShare 数据源：提供开源免费的 A 股历史 K 线（无需 token）。"""

    name = "akshare"
    display_name = "AKShare 数据服务"
    description = "通过 AKShare（东方财富）获取 A 股日/周/月线与分钟线历史行情，开源免费、无需 token"

    def __init__(self) -> None:
        self._ak = None
        self._inited = False
        self._enabled = AKSHARE_ENABLED

    def init(self, output: Callable = print) -> bool:
        """初始化 AKShare 模块（延迟导入，无需 token）。

        Args:
            output: 日志输出函数，默认 print。

        Returns:
            True 表示导入成功；False 表示已禁用或导入失败（软失败，不抛异常）。
        """
        if self._inited:
            return True

        if not self._enabled:
            output("AkshareProvider: 已通过 AKSHARE_ENABLED 禁用")
            return False

        try:
            import akshare as ak  # type: ignore

            self._ak = ak
            self._inited = True
            output("AkshareProvider: 初始化成功")
            return True
        except Exception as e:
            output(f"AkshareProvider: 初始化失败 - {e}")
            return False

    def get_status(self) -> ProviderStatus:
        """返回 AKShare 当前可用状态。

        Returns:
            NOT_CONFIGURED（已禁用）/ UNAVAILABLE（未初始化）/ AVAILABLE（就绪）。
        """
        if not self._enabled:
            return ProviderStatus.NOT_CONFIGURED
        if not self._inited:
            return ProviderStatus.UNAVAILABLE
        return ProviderStatus.AVAILABLE

    def get_supported_categories(self) -> list[DataCategory]:
        """返回 AKShare 支持的数据品类列表（合约信息与历史 K 线）。

        Returns:
            [DataCategory.CONTRACT, DataCategory.BAR_HISTORY]
        """
        return [DataCategory.CONTRACT, DataCategory.BAR_HISTORY]

    # ---- 合约信息 ----

    def get_contracts(
        self,
        product_type: str = "",
        exchange: str = "",
    ) -> list[ContractInfo] | None:
        self._ensure_ready()
        if product_type and product_type not in ("股票", "stock"):
            return None

        normalized_exchange = self._normalize_exchange(exchange) if exchange else ""
        if normalized_exchange and normalized_exchange not in AK_SUPPORTED_EXCHANGES:
            return None

        df = self._fetch_contract_frame()
        if df is None or len(df) == 0:
            return None

        result: list[ContractInfo] = []
        for row in df.to_dict("records"):
            raw_code = str(row.get("代码") or row.get("code") or row.get("symbol") or "").strip()
            if not raw_code:
                continue
            try:
                ak_symbol, canonical_symbol, contract_exchange = self._normalize_symbol_inputs(
                    raw_code,
                    normalized_exchange,
                )
            except AkshareProviderError:
                continue
            if normalized_exchange and contract_exchange != normalized_exchange:
                continue
            result.append(
                ContractInfo(
                    symbol=canonical_symbol,
                    exchange=contract_exchange,
                    name=str(row.get("名称") or row.get("name") or ""),
                    product_type="股票",
                    size=1,
                    pricetick=0.01,
                    min_volume=100,
                )
            )
            _ = ak_symbol
        return result if result else None

    def get_contract(self, symbol: str, exchange: str) -> ContractInfo | None:
        ak_symbol, canonical_symbol, canonical_exchange = self._normalize_symbol_inputs(symbol, exchange)
        contracts = self.get_contracts(product_type="股票", exchange=canonical_exchange)
        if not contracts:
            return None
        for contract in contracts:
            if contract.symbol == canonical_symbol:
                return contract
        _ = ak_symbol
        return None

    # ---- 历史 K 线 ----

    def get_bar_history(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: datetime,
        end: datetime | None = None,
    ) -> list[BarRecord] | None:
        self._ensure_ready()
        ak_symbol, canonical_symbol, canonical_exchange = self._normalize_symbol_inputs(symbol, exchange)
        end = end or datetime.now()

        # 转债代码（前3位在 _CB_ALL_PREFIXES 中）分派到专用接口
        if self._is_cb_symbol(ak_symbol):
            if interval not in HIST_PERIOD_MAP:
                raise AkshareProviderError(
                    f"AKShare 转债行情仅支持日线（d/w/m）周期，不支持 {interval}"
                )
            records = self._fetch_cb_daily(ak_symbol, canonical_symbol, canonical_exchange, interval, start, end)
            return records

        if interval in HIST_PERIOD_MAP:
            df = self._fetch_daily(ak_symbol, interval, start, end)
            empty_hint = (
                f"AKShare 在 {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')} "
                f"未返回 {canonical_symbol}.{canonical_exchange} 的 {interval} 数据"
            )
        elif interval in MIN_PERIOD_MAP:
            source, df = self._fetch_minute(ak_symbol, canonical_exchange, interval, start, end)
            empty_hint = (
                f"AKShare 在 {start.strftime('%Y-%m-%d %H:%M:%S')} ~ "
                f"{end.strftime('%Y-%m-%d %H:%M:%S')} 未返回 "
                f"{canonical_symbol}.{canonical_exchange} 的 {interval} 数据"
            )
            if interval == "1m":
                empty_hint += "（1 分钟线历史深度有限，建议优先使用 5m 及以上周期）"
        else:
            raise AkshareProviderError(f"AKShare 不支持周期 {interval}")

        if df is None:
            raise AkshareProviderError(
                f"AKShare 请求 {canonical_symbol}.{canonical_exchange} 失败，请稍后重试"
            )
        if len(df) == 0:
            raise AkshareProviderError(empty_hint)

        minute_source = source if interval in MIN_PERIOD_MAP else "em"
        records = self._to_bar_records(
            df,
            canonical_symbol,
            canonical_exchange,
            interval,
            start=start,
            end=end,
            source=minute_source,
        )
        if not records:
            raise AkshareProviderError(
                f"AKShare 返回了数据但无法解析为 K 线: {canonical_symbol}.{canonical_exchange}"
            )
        return records

    # ---- 内部辅助方法 ----

    @staticmethod
    def _is_cb_symbol(ak_symbol: str) -> bool:
        """判断 6 位纯数字代码是否属于可转债（前3位在已知转债前缀集合内）。"""
        return len(ak_symbol) >= 3 and ak_symbol[:3] in _CB_ALL_PREFIXES

    def _fetch_cb_daily(
        self,
        ak_symbol: str,
        canonical_symbol: str,
        canonical_exchange: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[BarRecord]:
        """调用 bond_zh_hs_cov_daily 拉取转债日线，映射到 BarRecord 列表。

        bond_zh_hs_cov_daily 的 symbol 参数格式为 "sh113050" / "sz128093"。
        列映射：date→datetime / open/high/low/close / volume（turnover 无字段置 0）。
        陷阱：未上市新债会抛 KeyError('date')，捕获后抛 AkshareProviderError 友好提示。

        Args:
            ak_symbol: 6 位纯数字代码（如 "113050"）。
            canonical_symbol: 项目标准代码（同 ak_symbol）。
            canonical_exchange: 项目标准交易所（SSE 或 SZSE）。
            interval: 行情周期（d/w/m，当前 bond_zh_hs_cov_daily 仅有日线，其余视作日线）。
            start: 起始时间（含）。
            end: 截止时间（含）。

        Returns:
            非空 BarRecord 列表，按 datetime 升序。

        Raises:
            AkshareProviderError: 网络失败、该转债无历史行情（可能未上市）或无法解析。
        """
        # 组装 AKShare bond_zh_hs_cov_daily 所需的 symbol 前缀
        prefix = "sh" if canonical_exchange == "SSE" else "sz"
        bond_symbol = f"{prefix}{ak_symbol}"

        def _fetch():
            try:
                return self._ak.bond_zh_hs_cov_daily(symbol=bond_symbol)
            except KeyError as exc:
                # 未上市新债抛 KeyError('date')，转为友好错误
                if "date" in str(exc).lower():
                    raise AkshareProviderError(
                        f"该转债 {canonical_symbol}.{canonical_exchange} 无历史行情（可能未上市或尚未交易）"
                    ) from exc
                raise

        try:
            df = self._call_with_retry(f"转债日线({bond_symbol})", _fetch)
        except AkshareProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AkshareProviderError(
                f"AKShare 请求转债 {canonical_symbol}.{canonical_exchange} 失败，请稍后重试"
            ) from exc

        if df is None or len(df) == 0:
            raise AkshareProviderError(
                f"AKShare 在 {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')} "
                f"未返回 {canonical_symbol}.{canonical_exchange} 的转债行情"
            )

        # 列映射：bond_zh_hs_cov_daily 返回 date/open/high/low/close/volume
        records: list[BarRecord] = []
        for row in df.to_dict("records"):
            raw_dt = row.get("date")
            if raw_dt is None:
                continue
            dt = self._parse_datetime(raw_dt, "%Y-%m-%d")
            if dt is None:
                continue
            if dt < start:
                continue
            if dt > end:
                continue
            try:
                records.append(
                    BarRecord(
                        symbol=canonical_symbol,
                        exchange=canonical_exchange,
                        datetime=dt,
                        interval=interval,
                        open_price=float(row.get("open") or 0.0),
                        high_price=float(row.get("high") or 0.0),
                        low_price=float(row.get("low") or 0.0),
                        close_price=float(row.get("close") or 0.0),
                        volume=float(row.get("volume") or 0.0),
                        turnover=0.0,  # bond_zh_hs_cov_daily 无成交额字段
                        open_interest=0.0,
                        adjust_type="none",
                    )
                )
            except (TypeError, ValueError):
                continue

        if not records:
            raise AkshareProviderError(
                f"AKShare 返回了转债数据但无法解析为 K 线: {canonical_symbol}.{canonical_exchange}"
            )

        records.sort(key=lambda x: x.datetime)
        return records

    def _ensure_ready(self) -> None:
        """检查 AKShare 是否可用，否则抛出 AkshareProviderError。

        Raises:
            AkshareProviderError: Provider 已禁用或未完成初始化时抛出。
        """
        if not self._enabled:
            raise AkshareProviderError("AKShare 已通过 AKSHARE_ENABLED 禁用")
        if not self._inited:
            raise AkshareProviderError("AKShare 未初始化，请确认已安装 akshare 依赖")

    @staticmethod
    def _normalize_exchange(exchange: str) -> str:
        """将交易所别名标准化为项目内部写法（如 ``"SH"`` → ``"SSE"``）。

        Args:
            exchange: 原始交易所字符串（支持 SH/SSE/SZ/SZSE/BJ/BSE 等写法）。

        Returns:
            项目标准交易所代码（大写）；未知别名原样返回大写。
        """
        return _EXCHANGE_ALIASES.get((exchange or "").strip().upper(), (exchange or "").strip().upper())

    def _normalize_symbol_inputs(self, symbol: str, exchange: str) -> tuple[str, str, str]:
        """将多种输入格式统一为 (akshare纯6位代码, 项目标准代码, 项目标准交易所)。

        支持格式：
        - 带交易所的 vt_symbol：``"000415.SZSE"`` / ``"000415.SZ"``
        - 前缀写法：``"sz000415"`` / ``"SH600000"``
        - 纯数字代码（需 exchange 参数或可由前缀推断）

        Args:
            symbol: 原始合约代码，支持多种格式。
            exchange: 交易所字符串（可为空，会在函数内推断）。

        Returns:
            (ak_symbol, canonical_symbol, canonical_exchange) 三元组。

        Raises:
            AkshareProviderError: 无法识别代码、交易所不匹配或不支持时抛出。
        """
        raw_symbol = (symbol or "").strip()
        raw_exchange = self._normalize_exchange(exchange)

        if not raw_symbol:
            raise AkshareProviderError("证券代码不能为空")

        # symbol 字段直接传入 vt_symbol：000415.SZSE / 000415.SZ
        if "." in raw_symbol:
            sym_part, exch_part = raw_symbol.rsplit(".", 1)
            raw_symbol = sym_part.strip()
            if not raw_exchange:
                raw_exchange = self._normalize_exchange(exch_part)

        # 前缀写法：sz000415 / SH600000
        lower = raw_symbol.lower()
        if len(lower) >= 8 and lower[:2] in _PREFIX_EXCHANGES:
            raw_symbol = lower[2:8]
            if not raw_exchange:
                raw_exchange = _PREFIX_EXCHANGES[lower[:2]]

        digits = "".join(ch for ch in raw_symbol if ch.isdigit())
        if not digits:
            raise AkshareProviderError(f"无法识别证券代码: {symbol}")

        if len(digits) > 6:
            digits = digits[-6:]
        ak_symbol = digits.zfill(6)

        guessed_exchange = self._guess_exchange(ak_symbol)
        if not raw_exchange:
            if not guessed_exchange:
                raise AkshareProviderError(f"无法根据代码 {ak_symbol} 推断交易所，请使用 000415.SZSE 格式")
            raw_exchange = guessed_exchange

        if raw_exchange not in AK_SUPPORTED_EXCHANGES:
            raise AkshareProviderError(
                f"AKShare 暂不支持交易所 {raw_exchange}，仅支持 SSE/SZSE/BSE"
            )

        if guessed_exchange and raw_exchange != guessed_exchange:
            raise AkshareProviderError(
                f"证券代码 {ak_symbol} 与交易所 {raw_exchange} 不匹配，建议使用 {ak_symbol}.{guessed_exchange}"
            )

        return ak_symbol, ak_symbol, raw_exchange

    def _call_with_retry(self, label: str, fetcher: Callable[[], object]):
        """带重试逻辑的 AKShare 接口调用封装。

        最多重试 AKSHARE_MAX_RETRIES 次，每次失败后等待 AKSHARE_RETRY_DELAY_SEC × attempt 秒。
        AkshareProviderError 不重试（业务级错误），直接透传。

        Args:
            label: 调用标签，用于告警日志中标识失败来源（如 ``"日线(000001)"``）。
            fetcher: 无参可调用对象，封装实际 akshare API 调用。

        Returns:
            fetcher() 的返回值（pandas DataFrame）。

        Raises:
            AkshareProviderError: 全部重试耗尽后仍失败时抛出，含重试次数说明。
        """
        last_error: Exception | None = None
        attempts = max(AKSHARE_MAX_RETRIES, 1)
        for attempt in range(1, attempts + 1):
            try:
                return fetcher()
            except AkshareProviderError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                logger.warning(
                    "AkshareProvider: %s 第 %s/%s 次失败: %s",
                    label,
                    attempt,
                    attempts,
                    exc,
                )
                time.sleep(AKSHARE_RETRY_DELAY_SEC * attempt)

        message = str(last_error) if last_error else "未知错误"
        raise AkshareProviderError(f"AKShare {label} 连接失败（已重试 {attempts} 次）: {message}") from last_error

    def _fetch_contract_frame(self):
        """获取 A 股合约列表 DataFrame（优先使用 stock_info_a_code_name）。

        Returns:
            pandas DataFrame（含代码与名称列）；失败时返回 None（软失败）。
        """
        def _fetch():
            if hasattr(self._ak, "stock_info_a_code_name"):
                return self._ak.stock_info_a_code_name()
            return self._ak.stock_zh_a_spot_em()

        try:
            return self._call_with_retry("合约列表", _fetch)
        except AkshareProviderError:
            return None

    def _fetch_daily(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ):
        """通过 stock_zh_a_hist 拉取 A 股日/周/月线数据。

        Args:
            symbol: 6 位纯数字代码，如 ``"600519"``。
            interval: 周期标识（d/w/m），对应 HIST_PERIOD_MAP。
            start: 起始日期（含）。
            end: 截止日期（含）。

        Returns:
            pandas DataFrame；失败时抛出 AkshareProviderError（经 _call_with_retry 包裹）。
        """
        return self._call_with_retry(
            f"日线({symbol})",
            lambda: self._ak.stock_zh_a_hist(
                symbol=symbol,
                period=HIST_PERIOD_MAP[interval],
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="",
            ),
        )

    def _fetch_minute(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> tuple[str, object]:
        """拉取分钟级 K 线：优先东财，失败时降级新浪。

        东财（stock_zh_a_hist_min_em）支持按日期区间拉取；
        新浪（stock_zh_a_minute）历史深度有限（约近 5 个交易日），稳定性略好。

        Args:
            symbol: 6 位纯数字代码。
            exchange: 项目标准交易所代码（用于构造新浪代码前缀）。
            interval: 分钟周期标识（1m/5m/15m/30m/1h/60m），对应 MIN_PERIOD_MAP。
            start: 起始时间（含）。
            end: 截止时间（含）。

        Returns:
            (source, DataFrame) 元组，source 为 ``"em"``（东财）或 ``"sina"``（新浪）。

        Raises:
            AkshareProviderError: 两个数据源均失败时抛出。
        """
        period = MIN_PERIOD_MAP[interval]
        # 优先东财（可按日期区间拉取）；失败时降级新浪（稳定性更好，但历史深度有限）。
        try:
            df = self._call_with_retry(
                f"分钟线-东财({symbol})",
                lambda: self._ak.stock_zh_a_hist_min_em(
                    symbol=symbol,
                    start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                    end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                    period=period,
                    adjust="",
                ),
            )
            return "em", df
        except AkshareProviderError as em_error:
            logger.warning("AkshareProvider: 东财分钟线不可用，降级新浪接口: %s", em_error)

        sina_symbol = self._to_sina_symbol(symbol, exchange)
        df = self._call_with_retry(
            f"分钟线-新浪({symbol})",
            lambda: self._ak.stock_zh_a_minute(
                symbol=sina_symbol,
                period=period,
                adjust="",
            ),
        )
        return "sina", df

    @staticmethod
    def _to_sina_symbol(symbol: str, exchange: str) -> str:
        """将 6 位代码转换为新浪接口所需的带前缀格式（如 ``"sh600519"``）。

        Args:
            symbol: 6 位纯数字代码。
            exchange: 项目标准交易所代码（SSE/SZSE/BSE）。

        Returns:
            新浪格式代码字符串，如 ``"sh600519"``/``"sz000001"``。

        Raises:
            AkshareProviderError: 未知交易所时抛出。
        """
        prefix = _SINA_EXCHANGE_PREFIX.get(exchange)
        if not prefix:
            raise AkshareProviderError(f"无法构造新浪代码，未知交易所: {exchange}")
        return f"{prefix}{symbol}"

    def _to_bar_records(
        self,
        df,
        symbol: str,
        exchange: str,
        interval: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        source: str = "em",
    ) -> list[BarRecord] | None:
        """将 AKShare 返回的 DataFrame 行转换为 BarRecord 列表。

        同时支持东财（source="em"）与新浪（source="sina"）两种列名格式，
        以及日线（列名为中文）和分钟线（列名为"时间"或"day"）两种情况。
        解析失败的行被静默跳过（dtype 转换异常不中断）。

        Args:
            df: AKShare 接口返回的 pandas DataFrame。
            symbol: 项目标准代码。
            exchange: 项目标准交易所代码。
            interval: K 线周期，用于区分日线/分钟线及设置 BarRecord.interval。
            start: 起始时间过滤（含）；None 表示不限。
            end: 截止时间过滤（含）；None 表示不限。
            source: 数据来源标识，``"em"``（东财默认）或 ``"sina"``（新浪降级）。

        Returns:
            按 datetime 升序排列的 BarRecord 列表；解析后无有效记录则返回 None。
        """
        is_minute = interval in MIN_PERIOD_MAP
        records: list[BarRecord] = []

        for row in df.to_dict("records"):
            if source == "sina" and is_minute:
                raw_dt = row.get("day")
                dt_fmt = "%Y-%m-%d %H:%M:%S"
                open_price = row.get("open")
                high_price = row.get("high")
                low_price = row.get("low")
                close_price = row.get("close")
                volume = row.get("volume")
                turnover = row.get("amount")
            else:
                dt_col = "时间" if is_minute else "日期"
                raw_dt = row.get(dt_col)
                dt_fmt = "%Y-%m-%d %H:%M:%S" if is_minute else "%Y-%m-%d"
                open_price = row.get("开盘")
                high_price = row.get("最高")
                low_price = row.get("最低")
                close_price = row.get("收盘")
                volume = row.get("成交量")
                turnover = row.get("成交额")

            if raw_dt is None:
                continue
            dt = self._parse_datetime(raw_dt, dt_fmt)
            if dt is None:
                continue
            if start and dt < start:
                continue
            if end and dt > end:
                continue
            try:
                records.append(
                    BarRecord(
                        symbol=symbol,
                        exchange=exchange,
                        datetime=dt,
                        interval=interval,
                        open_price=float(open_price or 0.0),
                        high_price=float(high_price or 0.0),
                        low_price=float(low_price or 0.0),
                        close_price=float(close_price or 0.0),
                        volume=float(volume or 0.0),
                        turnover=float(turnover or 0.0),
                        open_interest=0.0,
                        adjust_type="none",  # AKShare 当前以 adjust="" 拉取，不复权
                    )
                )
            except (TypeError, ValueError):
                continue

        if not records:
            return None
        records.sort(key=lambda x: x.datetime)
        return records

    @staticmethod
    def _parse_datetime(value, fmt: str) -> datetime | None:
        """将各种格式的日期/时间值解析为 datetime 对象。

        按优先顺序尝试 fmt → 常见格式（%Y-%m-%d %H:%M:%S / %Y-%m-%d / %Y%m%d），
        任一成功即返回；全部失败则返回 None（静默失败，由调用方跳过该行）。

        Args:
            value: 待解析的值，可为 datetime（直接返回）或字符串。
            fmt: 优先尝试的日期格式字符串。

        Returns:
            解析后的 datetime；无法解析时返回 None。
        """
        if isinstance(value, datetime):
            return value
        text = str(value).strip()
        if not text:
            return None
        for candidate in (fmt, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(text, candidate)
            except ValueError:
                continue
        return None

    @staticmethod
    def _guess_exchange(symbol: str) -> str | None:
        """根据代码首位数字推断 A 股交易所（规则性推断，不适用全部品种）。

        推断规则（A 股现行编码惯例）：
        - 6 开头 → SSE（上交所）；
        - 0 或 3 开头 → SZSE（深交所）；
        - 4/8/9 开头 → BSE（北交所）。

        Args:
            symbol: 6 位纯数字代码。

        Returns:
            ``"SSE"`` / ``"SZSE"`` / ``"BSE"``；无法判定时返回 None。
        """
        if symbol.startswith("6"):
            return "SSE"
        if symbol.startswith(("0", "3")):
            return "SZSE"
        if symbol.startswith(("4", "8", "9")):
            return "BSE"
        return None
