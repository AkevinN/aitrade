"""
aitrade API 请求/响应模型。

定义所有 API 层使用的 Pydantic v2 模型：任务模型（TaskModel/TaskStatus/TaskType）、
Alpha 研究请求（数据下载/数据集/模型训练/信号/回测）、CNN 相关请求，以及各种响应模型。
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """异步任务状态枚举。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """异步任务类型枚举（单一事实来源，API 层与 TaskManager 共用）。"""
    DATA_DOWNLOAD = "data_download"
    DATA_IMPORT = "data_import"
    DATA_AGGREGATE = "data_aggregate"
    DATASET_CREATE = "dataset_create"
    MODEL_TRAIN = "model_train"
    SIGNAL_GENERATE = "signal_generate"
    BACKTEST_RUN = "backtest_run"
    CNN_TRAIN = "cnn_train"
    CNN_BACKTEST = "cnn_backtest"
    CNN_PREDICT = "cnn_predict"
    CNN_WF_EVALUATE = "cnn_wf_evaluate"
    CNN_CANDIDATE_TRAIN = "cnn_candidate_train"
    CNN_PROMOTE = "cnn_promote"
    CNN_ROLLBACK = "cnn_rollback"
    CNN_DRIFT_CHECK = "cnn_drift_check"
    CNN_GOVERNANCE_REPLAY = "cnn_governance_replay"
    SCHEME_BACKTEST = "scheme_backtest"
    LIVE_DECISION = "live_decision"
    STRATEGY_BACKTEST = "strategy_backtest"
    STRATEGY_SWEEP = "strategy_sweep"
    STRATEGY_WALKFORWARD = "strategy_walkforward"
    LIVE_REBALANCE = "live_rebalance"


class TaskModel(BaseModel):
    """异步任务状态模型（内存存储 + API 响应共用）。

    所有字段带默认值以保证既有消费者零回归；started_at/finished_at/duration_ms 为
    task-scheduler-observability R1 新增字段，旧版 JSON 兼容读取（None 为默认）。
    """
    task_id: str
    type: TaskType
    title: str = ""
    entity_type: str = ""
    entity_name: str = ""
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: Optional[dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    # --- 任务执行记录增强（task-scheduler-observability R1）---
    # 全部带默认值，保证既有消费者零回归。
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    error_traceback: str = ""       # 失败堆栈，截断至 8000 字符
    params: dict[str, Any] = Field(default_factory=dict)  # 创建参数摘要（已脱敏）


# =============================================================================
# Alpha research request models
# =============================================================================

class DataDownloadRequest(BaseModel):
    """Raw market data download request."""
    vt_symbols: list[str] = Field(description="合约列表，如 ['000001.SZSE', '600000.SSE']")
    start: date = Field(description="开始日期")
    end: date = Field(description="结束日期")
    data_kind: str = Field(default="bar", description="原始数据类型: bar | tick")
    source_interval: Optional[str] = Field(default=None, description="原始K线周期，如 d/1m/5m")
    interval: Optional[str] = Field(default=None, description="兼容旧字段，等价于 source_interval")
    provider: Optional[str] = Field(default=None, description="指定数据源，如 tushare/akshare；为空则自动选择")
    asset_class: Literal["stock", "etf", "cbond"] = Field(default="stock", description="品种类型：stock=A股股票；etf=交易所交易基金；cbond=可转债")


class DataAggregateRequest(BaseModel):
    """Local aggregation request."""
    vt_symbols: list[str] = Field(description="需要聚合的合约列表")
    start: date = Field(description="开始日期")
    end: date = Field(description="结束日期")
    source_kind: str = Field(default="bar", description="来源类型: bar | tick")
    source_interval: Optional[str] = Field(default=None, description="来源周期，如 1m")
    target_interval: str = Field(description="目标周期，如 5m/10m/30m")
    session_profile: str = Field(default="cn_equity", description="交易时段规则")


class DataResourceMergeRequest(BaseModel):
    """Merge pending upload batches into an official raw resource."""
    kind: Literal["raw_bar", "raw_tick"] = Field(description="目标原始资源类型")
    keys: list[str] = Field(description="上传批次 key 列表")


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


class CNNBacktestRequest(BaseModel):
    """CNN model backtest request."""
    name: str = Field(description="回测名称")
    model: str = Field(description="CNN 模型名称")
    capital: float = Field(default=1_000_000, description="初始资金")
    start: date = Field(description="回测开始日期")
    end: date = Field(description="回测结束日期")
    buy_threshold: float = Field(default=0.6, description="买入阈值 (0~1)")
    sell_threshold: float = Field(default=0.4, description="卖出阈值 (0~1)")
    commission_rate: float = Field(default=0.0003, ge=0, lt=0.1, description="单边佣金率（默认万3）")
    stamp_duty: float = Field(default=0.0005, ge=0, lt=0.1, description="卖出印花税率（默认0.5‰，A股2023-08起现行）")
    slippage: float = Field(default=0.0005, ge=0, lt=0.1, description="每笔成交不利滑点率（默认5bp）")
    price_add: float = Field(default=0.002, ge=0, lt=0.1, description="限价单价格缓冲/市价化挂单（默认20bp）")
    exit_mode: Literal["threshold", "fixed_hold", "oco", "auto"] = Field(
        default="threshold",
        description="出场模式：threshold=概率阈值平仓；fixed_hold=固定持有；oco=止盈止损；auto=按模型 label 自动推导对齐的固定持有出场",
    )
    hold_days: int = Field(default=1, ge=1, le=60, description="fixed_hold/oco 的固定/最大持有交易日数（auto 模式下由 label 覆盖）")
    take_profit: float = Field(default=0.0, ge=0, lt=1.0, description="oco 止盈幅度（0.02=+2%），0=不启用")
    stop_loss: float = Field(default=0.0, ge=0, lt=1.0, description="oco 止损幅度（0.03=-3%），0=不启用")
    t_plus1: bool = Field(default=False, description="是否启用 T+1 卖出限制（当日买入不可当日卖出）")


class CNNPredictRequest(BaseModel):
    """CNN model inference request — 推理生成信号并保存到信号库。"""
    name: str = Field(description="信号名称（保存后可在 Alpha 回测中复用）")
    model: str = Field(description="CNN 模型名称")
    start: date = Field(description="推理开始日期")
    end: date = Field(description="推理结束日期")


# =============================================================================
# CNN request models
# =============================================================================

class CNNTrainRequest(BaseModel):
    """CNN model training request."""
    name: str = Field(description="模型名称")
    start: date = Field(description="训练开始日期")
    end: date = Field(description="训练结束日期")
    vt_symbols: list[str] = Field(default_factory=list, description="兼容旧请求的股票列表")
    target_symbol: Optional[str] = Field(default=None, description="预测目标股票，默认取 vt_symbols[0]")
    input_data_kind: str = Field(default="bar", description="输入数据类型: bar | tick")
    input_interval: str = Field(default="d", description="输入周期，如 d/1m/5m/10m")
    label_spec: "LabelSpec" = Field(default_factory=lambda: LabelSpec(mode="next_bar"))
    observation_groups: list["ObservationGroup"] = Field(default_factory=list, description="语义观测分组")
    epochs: int = Field(default=50, description="训练轮数")
    batch_size: int = Field(default=32, description="批大小")
    learning_rate: float = Field(default=0.001, description="学习率")
    lookback: int = Field(default=30, description="回看窗口（天）")
    dropout: float = Field(default=0.5, description="Dropout率")
    train_ratio: float = Field(default=0.7, description="训练集比例")
    loss_weighting: Literal["none", "magnitude"] = Field(
        default="none",
        description="损失加权：none=普通BCE；magnitude=按|未来收益|加权，让大波动样本主导梯度（仅分类）",
    )
    objective: Literal["classification", "regression"] = Field(
        default="classification",
        description="预测目标：classification=方向二分类(输出概率)；regression=直接预测涨跌幅(输出预测收益)",
    )


class ObservationRole(str, Enum):
    TARGET = "target"
    MARKET = "market"
    SECTOR = "sector"
    LEADERS = "leaders"
    CUSTOM = "custom"


class ObservationGroup(BaseModel):
    """Semantic observation group for CNN input."""
    role: ObservationRole = Field(description="分组角色")
    name: str = Field(description="分组展示名称")
    symbols: list[str] = Field(default_factory=list, description="该分组内的证券列表")


class LabelMode(str, Enum):
    NEXT_BAR = "next_bar"
    HORIZON_BARS = "horizon_bars"
    SESSION_CLOSE = "session_close"
    NEXT_SESSION_CLOSE = "next_session_close"
    OCO = "oco"


class LabelSpec(BaseModel):
    """Unified CNN label definition."""
    mode: LabelMode = Field(default=LabelMode.NEXT_BAR, description="标签模式")
    horizon: Optional[int] = Field(default=None, description="仅 horizon_bars 模式需要指定 horizon")
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "最小有效波动阈值（收益率，0.005 = 0.5%）。"
            "|未来收益| ≤ 阈值的样本视为噪声，按 neutral_policy 处理；"
            "0 表示关闭去噪、保持旧的 future_return>0 二分类行为"
        ),
    )
    neutral_policy: Literal["drop", "negative"] = Field(
        default="drop",
        description="中性样本（|未来收益|≤阈值）处理：drop=丢弃（默认），negative=并入下跌类",
    )
    price_ref: Literal["close", "next_open", "next_close", "next_vwap"] = Field(
        default="close",
        description=(
            "标签收益的计价口径（与回测撮合成交价一一对应）：\n"
            "- close=收盘到收盘(旧/研究口径，实盘吃不到，回测仍按次开盘近似)；\n"
            "- next_open=次开盘到次开盘，对齐「T 收盘出信号、T+1 开盘成交」；\n"
            "- next_close=次收盘到次收盘，对齐「T+1 收盘价(MOC)成交」；\n"
            "- next_vwap=次日均价到次日均价，对齐「T+1 全天均价(VWAP=成交额/成交量)成交」。"
        ),
    )
    # OCO（三重障碍）路径依赖标签专用字段，仅 mode=oco 时生效（与 cnn.dataset 字段名一致）。
    take_profit: Optional[float] = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="oco 模式止盈幅度（收益率，0.03=+3%）；mode=oco 时必填且 > 0",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="oco 模式止损幅度（收益率，0.02=-2%）；mode=oco 时必填且 > 0",
    )
    max_hold: Optional[int] = Field(
        default=None,
        ge=1,
        description="oco 模式最大持有 bar 数（到期未触发按时间止损平仓）；缺省回退到 horizon 或 10",
    )
    stop_first: bool = Field(
        default=True,
        description="oco 模式同根 bar 止盈止损都触发、日内先后未知时是否保守假设止损先到（默认 True，与回测一致）",
    )


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
    # 可选字段：不传则不写入 JSON（保持 JSON 干净）
    stamp_duty: float | None = None      # 卖出印花税率（A 股 2023-08 起默认 0.0005）
    slippage: float | None = None        # 每笔成交不利滑点率
    limit_ratio: float | None = None     # 单边涨跌停比例（None = 无限制，如转债）
    t_plus1: bool | None = None          # T+1 卖出限制（引擎层消费为下一任务）


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


class RelocateBarIntervalRequest(BaseModel):
    """更正原始 K 线资源的存储周期。"""
    interval: str = Field(description="目标周期，如 d/1m/5m/15m/30m/60m")


CNNTrainRequest.model_rebuild()
