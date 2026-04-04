"""
CNN 卷积神经网络量化预测 API 路由。

提供 CNN 模型的训练、查询、删除功能，
复用现有 TaskManager 实现异步任务管理。
"""

from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException

from ..models import CNNTrainRequest, TaskType
from ..task import task_manager

router = APIRouter(prefix="/api/cnn", tags=["CNN量化预测"])


# =============================================================================
# Torch check helpers
# =============================================================================

def _check_torch() -> bool:
    """检查 PyTorch 是否可用。"""
    try:
        import torch
        return True
    except ImportError:
        return False


def _get_device() -> str:
    """返回 PyTorch 可用的设备（cuda/cpu）。"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "N/A"


# =============================================================================
# 状态检查
# =============================================================================

@router.get("/status")
async def get_cnn_status() -> dict:
    """检查 CNN 功能是否可用（PyTorch 是否安装）。"""
    torch_ok = _check_torch()
    return {
        "torch_installed": torch_ok,
        "device": _get_device() if torch_ok else "N/A",
    }


# =============================================================================
# 训练管理
# =============================================================================

@router.post("/train")
async def start_cnn_train(req: CNNTrainRequest) -> dict:
    """
    启动 CNN 训练任务。

    通过 TaskManager 在后台线程执行，立即返回 task_id。
    前端可通过 /api/alpha/tasks/{task_id} 轮询进度。
    """
    if not _check_torch():
        raise HTTPException(400, "PyTorch 未安装，请先执行: pip install torch")

    if req.start >= req.end:
        raise HTTPException(400, "开始日期必须早于结束日期")

    task_id = task_manager.create_task(
        TaskType.CNN_TRAIN,
        params={"name": req.name, "symbols": req.vt_symbols},
    )

    def _run(on_progress: Optional[Callable[[float, str], None]] = None) -> dict[str, Any]:
        from ..cnn import train_cnn_model
        return train_cnn_model(
            name=req.name,
            vt_symbols=req.vt_symbols,
            start=req.start,
            end=req.end,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            lookback=req.lookback,
            dropout=req.dropout,
            train_ratio=req.train_ratio,
            on_progress=on_progress,
        )

    task_manager.run_async(task_id, _run, on_progress=True)

    return {"task_id": task_id, "name": req.name}


# =============================================================================
# 模型管理
# =============================================================================

@router.get("/models")
async def list_models() -> list[dict]:
    """列出已保存的 CNN 模型。"""
    from ..cnn import list_cnn_models
    return list_cnn_models()


@router.get("/models/{name}")
async def get_model_detail(name: str) -> dict:
    """获取模型详情（含训练历史）。"""
    from ..cnn import get_cnn_model_detail
    try:
        return get_cnn_model_detail(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.delete("/models/{name}")
async def delete_model(name: str) -> dict:
    """删除 CNN 模型。"""
    from ..cnn import delete_cnn_model
    ok = delete_cnn_model(name)
    if not ok:
        raise HTTPException(404, f"模型不存在: {name}")
    return {"deleted": name}
