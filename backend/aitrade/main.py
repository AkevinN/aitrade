"""
FastAPI 应用主入口 — aitrade 后端服务。

整合所有 API 路由、WebSocket 端点和数据源初始化。
"""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    API_CORS_ORIGINS,
    API_HOST,
    API_PORT,
    QMT_BRIDGE_TOKEN,
    QMT_BRIDGE_URL,
    SCHEDULER_ENABLED,
    SCHEDULER_LOCK_PATH,
    SCHEDULER_TICK_SECONDS,
)
from .datasource import AkshareProvider, MockProvider, QmtBridgeProvider, TushareProvider, datasource_manager
from .api import alpha_router, cnn_router, live_router, status_router, strategy_router
from .api.live import build_plan_scheduler, register_scheduler
from .api.ws import ws_manager
from .live.single_instance import SingleInstanceLock

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _configure_logging() -> None:
    """配置统一的后端 logging（R7.1 / R7.2）。

    - 日志级别经 AITRADE_LOG_LEVEL 环境变量控制，默认 INFO。
    - basicConfig 以 force=False（温和模式）运行：uvicorn 已配置 root handler 时为 no-op，
      避免破坏 uvicorn 控制台输出。
    - 若设置了 AITRADE_LOG_FILE，为 root logger 追加 RotatingFileHandler
      （单文件 20MB、保留 5 份；幂等：同路径 handler 已存在不重复加）。
    """
    level_str = os.getenv("AITRADE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    # force=False：uvicorn 已有 root handler 时为 no-op（温和，不破坏控制台输出）
    logging.basicConfig(level=level, format=_LOG_FORMAT)

    log_file = os.getenv("AITRADE_LOG_FILE")
    if log_file:
        root = logging.getLogger()
        # 幂等：同路径 RotatingFileHandler 已存在则跳过
        existing_paths = {
            getattr(h, "baseFilename", None)
            for h in root.handlers
            if isinstance(h, RotatingFileHandler)
        }
        abs_path = os.path.abspath(log_file)
        if abs_path not in existing_paths:
            fh = RotatingFileHandler(
                abs_path,
                maxBytes=20 * 1024 * 1024,  # 20 MB
                backupCount=5,
                encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter(_LOG_FORMAT))
            root.addHandler(fh)


_configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI 应用生命周期管理：启动时拉起依赖、关闭时优雅清理。

    作为 FastAPI 的 lifespan 上下文使用，yield 之前为 Startup 阶段、之后为 Shutdown 阶段。

    Startup:
      - 注册并初始化数据源提供者（tushare → akshare → mock 降级链）
      - 若 SCHEDULER_ENABLED，启动进程内交易计划调度器（PlanScheduler）；经单实例锁
        防同机多进程并发触发，锁被占用时不注册为运行中实例。

    Shutdown:
      - 停止调度器并释放单实例锁（仅当本进程成功启动了调度器时）。

    Args:
        app: 当前 FastAPI 应用实例，由框架在启动时传入（本函数不直接读取其属性）。

    Yields:
        None。yield 处即应用就绪、开始处理请求；控制权交回框架直至关闭。
    """
    qmt_provider = QmtBridgeProvider(url=QMT_BRIDGE_URL, token=QMT_BRIDGE_TOKEN)
    datasource_manager.register(qmt_provider, priority=-10)

    tushare_provider = TushareProvider()
    datasource_manager.register(tushare_provider, priority=0)

    akshare_provider = AkshareProvider()
    datasource_manager.register(akshare_provider, priority=10)

    mock_provider = MockProvider()
    datasource_manager.register(mock_provider, priority=100)

    datasource_manager.init_all()

    # 交易计划自动调度器：随应用启停；单实例锁防同机多进程并发触发（Req 4.1 / 7.2）。
    scheduler = None
    if SCHEDULER_ENABLED:
        scheduler = build_plan_scheduler(tick_seconds=SCHEDULER_TICK_SECONDS)
        lock = SingleInstanceLock(SCHEDULER_LOCK_PATH)
        started = scheduler.start(lock=lock)
        logger.info("PlanScheduler started=%s (tick=%ss)", started, SCHEDULER_TICK_SECONDS)
        if not started:
            scheduler = None  # 锁被占用：不注册为运行中实例
        register_scheduler(scheduler)

    yield

    if scheduler is not None:
        scheduler.stop()
        register_scheduler(None)


def create_app(history_store=None) -> FastAPI:
    """创建并装配 FastAPI 应用实例（CORS、各业务路由、WebSocket、根路由）。

    应用工厂：模块底部以默认参数调用一次得到全局 app；测试可重复调用以注入桩件。

    Args:
        history_store: 可选的 TaskHistoryStore 实例，挂到 app.state.history_store
            供 alpha 路由使用（主要用于测试注入）。None 时按默认路径
            TASK_HISTORY_PATH 构造一个实例。

    Returns:
        装配完成、已注册全部路由与中间件的 FastAPI 实例。
    """
    from .task.history import TaskHistoryStore as _TaskHistoryStore
    from .config import TASK_HISTORY_PATH as _TASK_HISTORY_PATH

    app = FastAPI(
        title="Aitrade Backend API",
        description="AI 量化交易研究平台后端 API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # 注入 history_store 到 app.state，供 alpha 路由使用（R2.3）
    app.state.history_store = history_store or _TaskHistoryStore(_TASK_HISTORY_PATH)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=API_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(status_router)
    app.include_router(alpha_router)
    app.include_router(cnn_router)
    app.include_router(live_router)
    app.include_router(strategy_router)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket 连接端点：建立连接后循环接收客户端消息并交由 ws_manager 处理。

        用于实时事件推送（如任务进度、行情）。连接断开或处理出错时统一注销该连接。

        Args:
            websocket: FastAPI 注入的 WebSocket 连接对象。
        """
        await ws_manager.connect(websocket)
        try:
            while True:
                text: str = await websocket.receive_text()
                await ws_manager.handle_client_message(websocket, text)
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception:
            ws_manager.disconnect(websocket)

    @app.get("/")
    async def root() -> dict[str, str]:
        """根路由健康探针：返回服务名、运行状态与 API 文档地址。

        Returns:
            形如 {"name": ..., "status": "running", "docs": "/docs"} 的 dict，
            供探活 / 快速确认服务在线。
        """
        return {
            "name": "Aitrade Backend",
            "status": "running",
            "docs": "/docs",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "aitrade.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
