"""
FastAPI 应用主入口 — aitrade 后端服务。

整合所有 API 路由、WebSocket 端点和数据源初始化。
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    API_CORS_ORIGINS,
    API_HOST,
    API_PORT,
    SCHEDULER_ENABLED,
    SCHEDULER_LOCK_PATH,
    SCHEDULER_TICK_SECONDS,
)
from .datasource import AkshareProvider, MockProvider, TushareProvider, datasource_manager
from .api import alpha_router, cnn_router, live_router, status_router
from .api.live import build_plan_scheduler, register_scheduler
from .api.ws import ws_manager
from .live.single_instance import SingleInstanceLock

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    应用生命周期管理。

    Startup:
      - 注册并初始化数据源提供者（tushare → akshare → mock 降级链）
      - 启动进程内交易计划调度器（PlanScheduler），经单实例锁防并发触发

    Shutdown:
      - 停止调度器并释放单实例锁
    """
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


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""
    app = FastAPI(
        title="Aitrade Backend API",
        description="AI 量化交易研究平台后端 API",
        version="1.0.0",
        lifespan=lifespan,
    )

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

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket 连接端点，用于实时事件推送。"""
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
