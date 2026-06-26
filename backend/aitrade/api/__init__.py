"""
API package — exposes all FastAPI routers.
"""

from .alpha import router as alpha_router
from .cnn import router as cnn_router
from .live import router as live_router
from .status import router as status_router
from .strategy import router as strategy_router
from .t0 import router as t0_router

__all__ = [
    "alpha_router",
    "cnn_router",
    "live_router",
    "status_router",
    "strategy_router",
    "t0_router",
]
