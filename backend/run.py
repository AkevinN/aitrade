"""Backend startup script for aitrade."""
import uvicorn
from aitrade.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "aitrade.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
