"""
Pydantic data models for aitrade API.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Async task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """Async task type."""
    DATA_DOWNLOAD = "data_download"
    DATASET_CREATE = "dataset_create"
    MODEL_TRAIN = "model_train"
    SIGNAL_GENERATE = "signal_generate"
    BACKTEST_RUN = "backtest_run"
    CNN_TRAIN = "cnn_train"


class TaskModel(BaseModel):
    """Async task state model."""
    task_id: str
    type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: Optional[dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Alpha research request models
# =============================================================================

class DataDownloadRequest(BaseModel):
    """K-line data download request."""
    vt_symbols: list[str] = Field(description="合约列表，如 ['000001.SZSE', '600000.SSE']")
    start: date = Field(description="开始日期")
    end: date = Field(description="结束日期")
    interval: str = Field(default="d", description="K线周期: d=日线, m=分钟")


class DatasetCreateRequest(BaseModel):
    """Dataset creation request."""
    name: str = Field(description="数据集名称")
    vt_symbols: list[str] = Field(description="合约列表")
    start: date = Field(description="训练开始日期")
    end: date = Field(description="测试结束日期")
    train_end: date = Field(description="训练结束日期")
    valid_end: Optional[date] = Field(default=None, description="验证结束日期")
    features: list[str] = Field(default=["alpha158"], description="特征集: alpha158, alpha101")
    label_period: int = Field(default=3, description="预测标签周期（天）")


class ModelTrainRequest(BaseModel):
    """Model training request."""
    name: str = Field(description="模型名称")
    dataset: str = Field(description="数据集名称")
    model_type: str = Field(default="lgb", description="模型类型: lgb, mlp, lasso")
    params: dict[str, Any] = Field(default_factory=dict, description="模型参数")


class SignalGenerateRequest(BaseModel):
    """Signal generation request."""
    name: str = Field(description="信号名称")
    model: str = Field(description="模型名称")
    vt_symbols: list[str] = Field(description="合约列表")
    start: date = Field(description="开始日期")
    end: date = Field(description="结束日期")


class BacktestRunRequest(BaseModel):
    """Strategy backtest request."""
    name: str = Field(description="回测名称")
    signal: str = Field(description="信号名称")
    capital: float = Field(default=1_000_000, description="初始资金")
    start: date = Field(description="回测开始日期")
    end: date = Field(description="回测结束日期")
    benchmark: Optional[str] = Field(default=None, description="基准合约")


# =============================================================================
# CNN request models
# =============================================================================

class CNNTrainRequest(BaseModel):
    """CNN model training request."""
    name: str = Field(description="模型名称")
    vt_symbols: list[str] = Field(description="股票列表")
    start: date = Field(description="训练开始日期")
    end: date = Field(description="训练结束日期")
    epochs: int = Field(default=50, description="训练轮数")
    batch_size: int = Field(default=32, description="批大小")
    learning_rate: float = Field(default=0.001, description="学习率")
    lookback: int = Field(default=30, description="回看窗口（天）")
    dropout: float = Field(default=0.5, description="Dropout率")
    train_ratio: float = Field(default=0.7, description="训练集比例")


# =============================================================================
# Response models
# =============================================================================

class DatasetInfo(BaseModel):
    """Dataset information."""
    name: str
    created_at: Optional[str] = None
    size_kb: float = 0.0


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    model_type: str = ""
    created_at: Optional[str] = None
    size_kb: float = 0.0


class SignalInfo(BaseModel):
    """Signal information."""
    name: str
    created_at: Optional[str] = None
    size_kb: float = 0.0


class ContractSetting(BaseModel):
    """Contract trading settings."""
    vt_symbol: str
    long_rate: float = 0.0
    short_rate: float = 0.0
    size: float = 1.0
    pricetick: float = 0.01


class SystemStatus(BaseModel):
    """System status response."""
    version: str
    torch_available: bool
    torch_device: str
    data_path: str
    tushare_token_set: bool
    providers: list[dict] = Field(default_factory=list)


class BarDataInfo(BaseModel):
    """Downloaded bar data info."""
    vt_symbol: str
    interval: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    count: int = 0
    size_kb: float = 0.0
