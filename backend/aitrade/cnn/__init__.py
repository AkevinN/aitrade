"""
CNN module — 多尺度卷积神经网络量化预测模型。

功能：
- 从 K 线数据构建多资产张量
- 定义多尺度卷积网络（MarketCNN）
- 训练 / 预测 / 模型保存加载
- 推理生成信号 / 回测策略
"""

from .model import (
    build_dataset,
    check_torch_available,
    create_market_cnn,
    normalize_observation_groups,
)
from .trainer import train_cnn_model
from .storage import (
    CNN_MODEL_DIR,
    list_cnn_models,
    get_cnn_model_detail,
    delete_cnn_model,
    model_input_interval,
)
from .architecture import describe_cnn_architecture
from .predictor import predict_cnn_signals
from .strategy import CNNSignalStrategy

__all__ = [
    "create_market_cnn",
    "build_dataset",
    "train_cnn_model",
    "check_torch_available",
    "normalize_observation_groups",
    "CNN_MODEL_DIR",
    "list_cnn_models",
    "get_cnn_model_detail",
    "delete_cnn_model",
    "model_input_interval",
    "describe_cnn_architecture",
    "predict_cnn_signals",
    "CNNSignalStrategy",
]

