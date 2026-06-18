"""qmt-bridge FastAPI 应用：把 XtdataClient 投影成 REST 端点。

取数经单线程串行锁保护（MiniQmt 单连接，禁并发 download）。
所有数据路由需 bearer token 鉴权；/health 开放无需认证。
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import Depends, FastAPI, Query, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from . import config
from .auth import make_token_guard
from .serialize import bars_to_ipc
from .xtdata_client import XtdataClient


class BarsRequest(BaseModel):
    """K 线查询请求体。

    Attributes:
        symbol: 股票代码，如 "600000"。
        exchange: aitrade 交易所标识，如 "SSE"/"SZSE"。
        interval: K 线周期，如 "d"/"1m"。
        start: 起始日期字符串，格式 "YYYYMMDD"。
        end: 结束日期字符串，格式 "YYYYMMDD"。
        adjust_type: 复权口径，默认 "hfq"（后复权）。
    """

    symbol: str
    exchange: str
    interval: str
    start: str
    end: str
    adjust_type: str = "hfq"


def create_app(client: Any = None, token: str | None = None) -> FastAPI:
    """构造 FastAPI 应用实例。

    client/token 不传时从 config + 真 xtquant 装配（Windows 生产环境用）。
    传入 client 与 token 可注入假客户端以供测试（不触碰真实 QMT 连接）。

    Args:
        client: 实现了 XtdataClient 接口的客户端对象；None 时自动构建
            真实 XtdataClient（需 Windows + xtquant）。
        token: Bearer 鉴权 token；None 时读取 config.BRIDGE_TOKEN。

    Returns:
        已注册全部路由的 FastAPI 应用实例，可直接用 uvicorn 伺服或
        用 TestClient 驱动测试。

    Example:
        >>> from fastapi.testclient import TestClient
        >>> app = create_app(client=FakeClient(), token="test-token")
        >>> c = TestClient(app)
        >>> c.get("/health").json()["connected"]
        True
    """
    app = FastAPI(title="qmt-bridge", version="0.1.0")
    app.state.client = client if client is not None else XtdataClient(ratio_adjust=config.RATIO_ADJUST)
    app.state.lock = threading.Lock()

    expected = token if token is not None else config.BRIDGE_TOKEN
    _guard = make_token_guard(expected)
    bearer = HTTPBearer()

    def require_token(cred: HTTPAuthorizationCredentials = Depends(bearer)) -> None:
        """校验 Bearer token，不匹配时由 _guard 抛 401。

        Args:
            cred: FastAPI HTTPBearer 解析出的凭据对象。

        Raises:
            HTTPException: token 不匹配时，status_code=401。
        """
        _guard(cred)

    @app.get("/health")
    def health() -> dict:
        """健康检查端点（无需鉴权）。

        Returns:
            dict 含 connected（是否连接 QMT）和 version（服务版本）。
        """
        return {
            "connected": bool(app.state.client.is_connected()),
            "version": app.version,
        }

    @app.post("/bars", dependencies=[Depends(require_token)])
    def bars(req: BarsRequest) -> Response:
        """查询 K 线数据，返回 Arrow IPC stream 字节流（鉴权必需）。

        取数经串行锁保护，禁止并发 download（MiniQmt 单连接限制）。

        Args:
            req: K 线查询参数，见 BarsRequest。

        Returns:
            content-type 为 ``application/vnd.apache.arrow.stream`` 的响应，
            body 为 zstd 压缩的 Arrow IPC stream。
        """
        with app.state.lock:
            rows = app.state.client.get_bars(
                req.symbol, req.exchange, req.interval, req.start, req.end,
                adjust_type=req.adjust_type,
            )
        return Response(
            content=bars_to_ipc(rows),
            media_type="application/vnd.apache.arrow.stream",
        )

    @app.get("/contracts", dependencies=[Depends(require_token)])
    def contracts(include_bse: bool = Query(default=False)) -> list[dict]:
        """查询全量合约列表（JSON，鉴权必需）。

        Args:
            include_bse: 是否包含北交所合约，默认 False。

        Returns:
            合约信息 dict 列表，每条含 symbol/exchange/name 等字段。
        """
        with app.state.lock:
            return app.state.client.get_contracts(include_bse=include_bse)

    @app.get("/trading_calendar", dependencies=[Depends(require_token)])
    def trading_calendar(
        exchange: str = Query(...),
        start: str = Query(...),
        end: str = Query(...),
    ) -> list[dict]:
        """查询交易日历（JSON，鉴权必需）。

        Args:
            exchange: aitrade 交易所标识，如 "SSE"。
            start: 起始日期字符串，格式 "YYYYMMDD"。
            end: 结束日期字符串，格式 "YYYYMMDD"。

        Returns:
            交易日信息 dict 列表，每条含 date/exchange/is_open 字段。
        """
        with app.state.lock:
            return app.state.client.get_trade_calendar(exchange, start, end)

    @app.get("/adj_factor", dependencies=[Depends(require_token)])
    def adj_factor(
        symbol: str = Query(...),
        exchange: str = Query(...),
        start: str = Query(default=""),
        end: str = Query(default=""),
    ) -> list[dict]:
        """查询复权因子序列（JSON，鉴权必需）。

        Args:
            symbol: 股票代码，如 "600000"。
            exchange: aitrade 交易所标识，如 "SSE"。
            start: 起始日期字符串，默认为空字符串（取全部历史）。
            end: 结束日期字符串，默认为空字符串（取至最新）。

        Returns:
            复权因子 dict 列表，每条含 trade_date/adj_factor 字段。
        """
        with app.state.lock:
            return app.state.client.get_adj_factor(symbol, exchange, start, end)

    @app.get("/fundamental", dependencies=[Depends(require_token)])
    def fundamental(
        symbol: str = Query(...),
        exchange: str = Query(...),
        start: str = Query(...),
        end: str = Query(...),
    ) -> list[dict]:
        """查询财务数据（JSON，鉴权必需）。

        Args:
            symbol: 股票代码，如 "600000"。
            exchange: aitrade 交易所标识，如 "SSE"。
            start: 起始日期字符串，格式 "YYYYMMDD"。
            end: 结束日期字符串，格式 "YYYYMMDD"。

        Returns:
            财务数据 dict 列表，每条含 symbol/exchange/table/report_period 等字段。
        """
        with app.state.lock:
            return app.state.client.get_fundamental(symbol, exchange, start, end)

    return app


app = create_app()
