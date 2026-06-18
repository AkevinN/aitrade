"""QmtBridgeProvider：经 REST 调用 Windows 上的 qmt-bridge 服务取 QMT 数据。

Mac 端永不 import xtquant。懒加载 httpx；无数据返回 None（让 manager 回退），
HTTP/网络错误 raise（provider_name 锁定时原样上抛，区分"真错"vs"无数据"）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

import polars as pl

from .base import BaseProvider
from .types import BarRecord, DataCategory, ProviderStatus

# QMT 周期代码映射（前端/代码 -> 桥接服务标准名）
_PERIOD_MAP: dict[str, str] = {
    "d": "d",
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "60m": "1h",
    "w": "w",
}


def _fmt_date(d: datetime) -> str:
    """把 datetime 格式化为 ``'YYYYMMDD'`` 字符串，供桥接 REST 接口使用。

    Args:
        d: 任意 datetime 对象，只取日期部分。

    Returns:
        形如 ``'20240101'`` 的 8 位日期字符串。

    Example:
        >>> _fmt_date(datetime(2024, 1, 2))
        '20240102'
    """
    return d.strftime("%Y%m%d")


class QmtBridgeProvider(BaseProvider):
    """通过 REST 桥在 Mac 端使用 QMT 数据的数据源。

    本 Provider 不依赖 xtquant，仅通过 HTTP 调用运行在 Windows 侧的
    qmt-bridge 服务（Wave A）来获取历史 K 线等数据。

    语义约定：
    - 无数据时返回 ``None``（让 DataSourceManager 回退到下一个 Provider）。
    - HTTP/网络错误时直接 ``raise``（当调用方显式指定 qmt 时可感知真实错误）。

    Attributes:
        name: Provider 注册名，固定为 ``"qmt"``。
        display_name: 面向用户的显示名。
        description: Provider 说明文本。
    """

    name = "qmt"
    display_name = "QMT 数据桥"
    description = "经 Windows 上的 qmt-bridge 服务使用 QMT/xtdata 数据"

    def __init__(self, url: str = "", token: str = "") -> None:
        """初始化 QmtBridgeProvider 配置（不建立连接）。

        Args:
            url: qmt-bridge 服务地址，如 ``"http://192.168.1.100:58610"``；
                空字符串表示未配置，``get_status()`` 将返回 NOT_CONFIGURED。
            token: Bearer 认证 token；与桥接服务配置保持一致。
        """
        self._url = url
        self._token = token
        self._http: Any = None
        self._inited = False

    def init(self, output: Callable = print) -> bool:
        """懒加载 httpx 并 GET /health 验证桥接服务是否在线且 QMT 已连接。

        Args:
            output: 日志输出函数，默认 ``print``；可替换为 ``logger.info``。

        Returns:
            ``True`` 表示桥接服务在线且 QMT 已连接，``False`` 表示任何软失败
            （未配置 URL、服务不可达、QMT 连接断开等）。

        Example:
            >>> p = QmtBridgeProvider(url="http://win:58610", token="secret")
            >>> p.init()   # 会打印连接状态
            False  # 若 Windows 侧未运行服务
        """
        if not self._url:
            output("[qmt] 未配置 QMT_BRIDGE_URL，跳过")
            return False
        try:
            import httpx  # 懒加载，避免无 httpx 环境下 import 报错

            self._http = httpx.Client(
                base_url=self._url,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=60.0,
            )
            resp = self._http.get("/health")
            resp.raise_for_status()
            self._inited = bool(resp.json().get("connected"))
            output(f"[qmt] 桥连接 {'正常' if self._inited else '在线但 QMT 未连'}")
            return self._inited
        except Exception as exc:
            output(f"[qmt] 初始化失败: {exc}")
            self._inited = False
            return False

    def get_status(self) -> ProviderStatus:
        """返回当前连接状态。

        Returns:
            - ``NOT_CONFIGURED``：URL 为空，尚未配置桥接地址。
            - ``AVAILABLE``：已成功 init 且 QMT 连接正常。
            - ``UNAVAILABLE``：URL 已配置但尚未 init 或 init 失败。
        """
        if not self._url:
            return ProviderStatus.NOT_CONFIGURED
        return ProviderStatus.AVAILABLE if self._inited else ProviderStatus.UNAVAILABLE

    def get_supported_categories(self) -> list[DataCategory]:
        """返回本 Provider 支持的数据品类列表。

        Returns:
            包含 BAR_HISTORY / CONTRACT / TRADE_CALENDAR / REFERENCE / FUNDAMENTAL
            的 DataCategory 列表。
        """
        return [
            DataCategory.BAR_HISTORY,
            DataCategory.CONTRACT,
            DataCategory.TRADE_CALENDAR,
            DataCategory.REFERENCE,
            DataCategory.FUNDAMENTAL,
        ]

    def get_bar_history(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: datetime,
        end: Optional[datetime] = None,
        adjust_type: str = "hfq",
    ) -> Optional[list[BarRecord]]:
        """向 qmt-bridge 请求历史 K 线，解码 Arrow IPC Stream 返回 BarRecord 列表。

        Args:
            symbol: 合约代码（不含交易所后缀），如 ``"600000"``。
            exchange: 交易所代码，如 ``"SSE"``/``"SZSE"``。
            interval: K 线周期，如 ``"d"``/``"1m"``/``"30m"``；
                ``"60m"`` 会自动映射为 ``"1h"``。
            start: 起始时间（含）；只取日期部分传给桥接服务。
            end: 截止时间（含）；``None`` 时传空字符串，由桥接服务取当日。
            adjust_type: 复权口径，``"hfq"`` 后复权 / ``"qfq"`` 前复权 /
                ``"none"`` 不复权，默认后复权。

        Returns:
            按 datetime 升序排列的 BarRecord 列表；若响应为空 DataFrame
            则返回 ``None``（让 manager 回退到下一个 Provider）。

        Raises:
            RuntimeError: 桥接服务返回 4xx/5xx HTTP 状态码时抛出（由
                ``raise_for_status()`` 触发），让显式指定 qmt 的调用方感知真实错误。
            Exception: httpx 网络异常（超时/连接拒绝等）原样上抛。

        Example:
            >>> bars = provider.get_bar_history("600000", "SSE", "d",
            ...     datetime(2024, 1, 1), datetime(2024, 1, 31))
            >>> bars[0].close_price
            10.5
        """
        body = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": _PERIOD_MAP.get(interval, interval),
            "start": _fmt_date(start),
            "end": _fmt_date(end) if end else "",
            "adjust_type": adjust_type,
        }
        resp = self._http.post("/bars", json=body)
        resp.raise_for_status()
        df = pl.read_ipc_stream(resp.content)
        if df.height == 0:
            return None
        return [BarRecord(**row) for row in df.iter_rows(named=True)]
