"""
CNN module — 多尺度卷积神经网络量化预测模型。

功能：
- 从 K 线数据构建多资产张量
- 定义多尺度卷积网络（MarketCNN）
- 训练 / 预测 / 模型保存加载
"""

from .model import build_dataset, _compute_features, create_market_cnn, check_torch_available
from .trainer import train_cnn_model
from .storage import (
    CNN_MODEL_DIR,
    list_cnn_models,
    get_cnn_model_detail,
    delete_cnn_model,
)

__all__ = [
    "create_market_cnn",
    "build_dataset",
    "_compute_features",
    "train_cnn_model",
    "check_torch_available",
    "CNN_MODEL_DIR",
    "list_cnn_models",
    "get_cnn_model_detail",
    "delete_cnn_model",
]
