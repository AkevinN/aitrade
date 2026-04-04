"""CLI entry point: python -m aitrade"""
import uvicorn
from .config import API_HOST, API_PORT

def main():
    uvicorn.run(
        "aitrade.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()
