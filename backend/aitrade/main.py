"""
FastAPI 应用主入口 — aitrade 后端服务。

整合所有 API 路由、WebSocket 端点和数据源初始化。
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import API_CORS_ORIGINS, API_HOST, API_PORT
from .datasource import MockProvider, TushareProvider, datasource_manager
from .api import alpha_router, cnn_router, status_router
from .api.ws import ws_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    应用生命周期管理。

    Startup:
      - 注册并初始化数据源提供者（tushare → mock 降级链）

    Shutdown:
      - 保留清理钩子（当前无持久化连接需关闭）
    """
    tushare_provider = TushareProvider()
    datasource_manager.register(tushare_provider, priority=0)

    mock_provider = MockProvider()
    datasource_manager.register(mock_provider, priority=100)

    datasource_manager.init_all()

    yield


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
