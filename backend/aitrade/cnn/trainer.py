"""
CNN 训练器 — 封装完整训练流程。

功能：数据加载 → 特征计算 → 数据集构建 → 模型训练 → 早停 → 保存
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .model import build_dataset, create_market_cnn
from .storage import CNN_MODEL_DIR, save_cnn_model

logger = logging.getLogger(__name__)


def train_cnn_model(
    name: str,
    vt_symbols: list[str],
    start: date,
    end: date,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    lookback: int = 30,
    dropout: float = 0.5,
    train_ratio: float = 0.7,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """
    CNN 模型训练入口 — 供 TaskManager 调用。

    完整流程: 数据加载 → 特征计算 → 数据集构建 → 模型训练 → 保存
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if not torch.cuda.is_available():
        logger.info("CUDA 不可用，将使用 CPU 训练")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. 构建数据集
    X, y, info = build_dataset(vt_symbols, start, end, lookback, on_progress)
    n = len(X)

    if n < 50:
        raise ValueError(f"样本数不足: {n}，需至少50个样本，请扩大日期范围或添加更多股票")

    # 2. 划分数据集（时间序列顺序，不shuffle）
    n_train = int(n * train_ratio)
    n_val = n - n_train
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size, shuffle=False,
    )

    if on_progress:
        on_progress(60, f"数据划分: 训练={n_train}, 验证={n_val}, 设备={device}")

    # 3. 初始化模型
    C, T, W = X.shape[1], X.shape[2], X.shape[3]
    model = create_market_cnn(C, T, W, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    total_params = sum(p.numel() for p in model.parameters())
    if on_progress:
        on_progress(62, f"模型参数: {total_params:,}, 开始训练...")

    # 4. 训练循环
    best_val_loss = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    patience_counter = 0
    patience = max(10, epochs // 5)
    history: list[dict] = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_correct += ((pred > 0.5).float() == yb).sum().item()
            train_total += len(xb)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(xb)
                val_correct += ((pred > 0.5).float() == yb).sum().item()
                val_total += len(xb)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(lr_now, 8),
        })

        # 早停
        if val_loss < best_val_loss - 0.0001:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # 进度回调
        if on_progress:
            pct = 62 + 35 * epoch / epochs
            on_progress(
                pct,
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_acc={train_acc:.1%} val_acc={val_acc:.1%}",
            )

        if patience_counter >= patience:
            if on_progress:
                on_progress(pct, f"早停触发: {patience_counter}轮无改善, 最佳epoch={best_epoch}")
            break

    elapsed = time.time() - start_time

    # 5. 恢复最佳状态并保存
    if best_state:
        model.load_state_dict(best_state)

    save_data = {
        "model_state_dict": model.state_dict(),
        "model_config": {"in_channels": C, "time_steps": T, "width": W, "dropout": dropout},
        "train_config": {
            "symbols": vt_symbols,
            "start": str(start),
            "end": str(end),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
            "lookback": lookback,
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

    model_path, history_path = save_cnn_model(name, save_data, history)

    if on_progress:
        on_progress(100, f"训练完成 | 最佳Epoch={best_epoch} | val_loss={best_val_loss:.5f} | 耗时={elapsed:.0f}s")

    return {
        "name": name,
        "model_path": str(model_path),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_acc": round(history[best_epoch - 1]["val_acc"], 4) if best_epoch > 0 else 0.0,
        "total_params": total_params,
        "train_samples": n_train,
        "val_samples": n_val,
        "elapsed_seconds": round(elapsed, 1),
        "history": history,
    }
