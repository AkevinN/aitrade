"""
CNN 模型持久化 — 模型保存、加载、列表、删除。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import CNN_MODEL_PATH

# 模型存储目录
CNN_MODEL_DIR: Path = CNN_MODEL_PATH
CNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def save_cnn_model(
    name: str,
    save_data: dict[str, Any],
    history: list[dict],
) -> tuple[Path, Path]:
    """
    保存 CNN 模型 checkpoint 和训练历史。

    Returns:
        (model_path, history_path)
    """
    import torch

    model_path = CNN_MODEL_DIR / f"{name}.pt"
    torch.save(save_data, str(model_path))

    history_path = CNN_MODEL_DIR / f"{name}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logger.info(f"CNN 模型已保存: {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return model_path, history_path


def list_cnn_models() -> list[dict]:
    """列出已保存的 CNN 模型"""
    models: list[dict] = []
    for f in CNN_MODEL_DIR.glob("*.pt"):
        name = f.stem
        models.append({
            "name": name,
            "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    return models


def get_cnn_model_detail(name: str) -> dict:
    """获取模型详情"""
    import torch

    model_path = CNN_MODEL_DIR / f"{name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {name}")

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    history_path = CNN_MODEL_DIR / f"{name}_history.json"
    history: list[dict] = []
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)

    return {
        "name": name,
        "train_config": checkpoint.get("train_config", {}),
        "model_config": checkpoint.get("model_config", {}),
        "best_epoch": checkpoint.get("best_epoch", 0),
        "best_val_loss": checkpoint.get("best_val_loss", 0.0),
        "history": history,
    }


def delete_cnn_model(name: str) -> bool:
    """删除 CNN 模型"""
    model_path = CNN_MODEL_DIR / f"{name}.pt"
    history_path = CNN_MODEL_DIR / f"{name}_history.json"
    deleted = False
    if model_path.exists():
        model_path.unlink()
        deleted = True
    if history_path.exists():
        history_path.unlink()
    return deleted
