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
    import torch

    models: list[dict] = []
    for f in CNN_MODEL_DIR.glob("*.pt"):
        name = f.stem
        checkpoint: dict[str, Any] = {}
        try:
            checkpoint = torch.load(str(f), map_location="cpu", weights_only=False)
        except Exception:
            checkpoint = {}
        raw_groups = checkpoint.get("train_config", {}).get("observation_groups", [])
        cleaned_groups = []
        for g in raw_groups:
            g = dict(g)
            role = str(g.get("role", "custom"))
            if role.startswith("ObservationRole."):
                role = role.split(".", 1)[1].lower()
            g["role"] = role
            cleaned_groups.append(g)
        models.append({
            "name": name,
            "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
            "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            "best_epoch": checkpoint.get("best_epoch", 0),
            "best_val_loss": checkpoint.get("best_val_loss", 0.0),
            "target_symbol": checkpoint.get("train_config", {}).get("target_symbol", ""),
            "input_data_kind": checkpoint.get("train_config", {}).get("input_data_kind", "bar"),
            "input_interval": checkpoint.get("train_config", {}).get("input_interval", "d"),
            "objective": checkpoint.get("train_config", {}).get("objective", "classification"),
            "group_count": checkpoint.get("model_config", {}).get("group_count", 1),
            "observation_groups": cleaned_groups,
        })
    return models


def checkpoint_input_interval(model_path: Path) -> str:
    """读 checkpoint 文件的训练输入周期（`train_config.input_interval`，默认 "d"）。

    供 API 层做「间隔锁定」校验（计划/手动决策的 bar_freq 必须与模型训练间隔一致）。
    文件不存在抛 `FileNotFoundError`；不可读（损坏/非 checkpoint）由 torch 抛原始异常。
    """
    import torch

    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path.stem}")
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    return str(checkpoint.get("train_config", {}).get("input_interval", "d"))


def model_input_interval(name: str) -> str:
    """按模型名读训练输入周期（默认模型库目录）。"""
    return checkpoint_input_interval(CNN_MODEL_DIR / f"{name}.pt")


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
        "created_at": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
        "train_config": checkpoint.get("train_config", {}),
        "model_config": checkpoint.get("model_config", {}),
        "normalization": checkpoint.get("normalization", {}),
        "dataset_info": checkpoint.get("dataset_info", {}),
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
