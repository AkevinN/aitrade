"""
System status API — GET /api/status.
"""

from fastapi import APIRouter

from ..config import ALPHA_LAB_PATH, TUSHARE_TOKEN
from ..datasource import datasource_manager

router = APIRouter(prefix="/api", tags=["System"])


@router.get("/status")
async def get_status() -> dict:
    """
    Return system-wide status for health checks and UI readiness.

    Returns:
        dict: System status
            - version: API server version
            - torch_available: PyTorch installed
            - torch_device: cuda/cpu
            - data_path: AlphaLab path
            - tushare_token_set: Whether TUSHARE_TOKEN is configured
            - providers: Registered data source providers
    """
    torch_available = False
    torch_device = "N/A"
    try:
        import torch
        torch_available = True
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    providers_info = []
    for info in datasource_manager.get_all_providers_info():
        providers_info.append({
            "name": info.name,
            "priority": info.priority,
            "status": info.status,
            "description": info.description,
        })

    return {
        "version": "1.0.0",
        "torch_available": torch_available,
        "torch_device": torch_device,
        "data_path": str(ALPHA_LAB_PATH),
        "tushare_token_set": bool(TUSHARE_TOKEN),
        "providers": providers_info,
    }
