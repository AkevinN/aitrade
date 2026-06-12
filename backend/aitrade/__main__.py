"""CLI 入口：python -m aitrade 以生产模式启动后端服务。"""
import uvicorn
from .config import API_HOST, API_PORT

def main():
    """以生产模式启动 aitrade 后端服务（reload=False）。

    读取 ``API_HOST`` / ``API_PORT`` 配置，调用 uvicorn 启动 FastAPI 应用。
    开发调试时建议改用 ``uvicorn aitrade.main:app --reload``，
    本函数供生产环境与 Docker 使用。
    """
    uvicorn.run(
        "aitrade.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()
