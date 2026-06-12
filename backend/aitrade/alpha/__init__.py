"""
Alpha 因子量化研究模块。

从 vnpy/alpha/ 提取并适配，提供以下核心接口：

- ``AlphaLab``：本地行情数据持久化与研究工件管理（K 线/Tick/数据集/模型/信号）。
- ``AlphaDataset``：因子截面数据集，封装训练/验证分割与 segment 索引。
- ``AlphaModel``：因子模型训练与预测封装。
- ``logger``：模块统一 loguru 日志实例。
- ``normalize_vt_symbol``（经 AlphaLab 重导出）：证券代码规范化工具函数。
"""

from .logger import logger
from .lab import AlphaLab
from .dataset import AlphaDataset, Segment, to_datetime
from .model import AlphaModel

__all__ = [
    "logger",
    "AlphaLab",
    "AlphaDataset",
    "Segment",
    "to_datetime",
    "AlphaModel",
]
