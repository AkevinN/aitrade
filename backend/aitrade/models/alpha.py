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
    """原始行情下载请求体。

    描述从外部数据源拉取裸行情（K 线或 tick）所需的合约范围、时间区间与数据源选择。
    各字段的语义/取值范围见对应 Field 的 description；其中 interval 为兼容旧版的别名，
    等价于 source_interval，二者择一即可。
    """
    vt_symbols: list[str] = Field(description="合约列表，如 ['000001.SZSE', '600000.SSE']")
    start: date = Field(description="开始日期")
    end: date = Field(description="结束日期")
    data_kind: str = Field(default="bar", description="原始数据类型: bar | tick")
    source_interval: Optional[str] = Field(default=None, description="原始K线周期，如 d/1m/5m")
    interval: Optional[str] = Field(default=None, description="兼容旧字段，等价于 source_interval")
    provider: Optional[str] = Field(default=None, description="指定数据源，如 tushare/akshare；为空则自动选择")
    asset_class: Literal["stock", "etf", "cbond"] = Field(default="stock", description="品种类型：stock=A股股票；etf=交易所交易基金；cbond=可转债")


class DataAggregateRequest(BaseModel):
    """本地行情聚合请求体。

    把已落地的低周期数据（如 1m）按目标周期重采样为高周期 K 线（如 5m/30m）。
    source_kind/source_interval 描述来源数据，target_interval 为目标周期，
    session_profile 指定交易时段切分规则；各字段语义见对应 Field 的 description。
    """
    vt_symbols: list[str] = Field(description="需要聚合的合约列表")
    start: date = Field(description="开始日期")
    end: date = Field(description="结束日期")
    source_kind: str = Field(default="bar", description="来源类型: bar | tick")
    source_interval: Optional[str] = Field(default=None, description="来源周期，如 1m")
    target_interval: str = Field(description="目标周期，如 5m/10m/30m")
    session_profile: str = Field(default="cn_equity", description="交易时段规则")


class DataResourceMergeRequest(BaseModel):
    """把待合并的上传批次并入正式原始资源的请求体。

    用于将多个暂存的上传批次（按 keys 标识）合并落地到指定类型的正式原始资源中。
    kind 指定目标资源类型（裸 K 线或裸 tick），keys 为待合并批次的标识列表。
    """
    kind: Literal["raw_bar", "raw_tick"] = Field(description="目标原始资源类型")
    keys: list[str] = Field(description="上传批次 key 列表")


class DatasetCreateRequest(BaseModel):
    """传统因子数据集创建请求体。

    描述构造一个训练/验证/测试三段切分的因子数据集所需的合约、时间区间、特征集与标签设定。
    时间轴语义：start..train_end 为训练段，train_end..valid_end 为验证段（valid_end 缺省
    则无独立验证段），其后至 end 为测试段。其余字段语义见对应 Field 的 description。
    """
    name: str = Field(description="数据集名称")
    vt_symbols: list[str] = Field(description="合约列表")
    start: date = Field(description="训练开始日期")
    end: date = Field(description="测试结束日期")
    train_end: date = Field(description="训练结束日期")
    valid_end: Optional[date] = Field(default=None, description="验证结束日期")
    features: list[str] = Field(default=["alpha158"], description="特征集: alpha158, alpha101")
    label_period: int = Field(default=3, description="预测标签周期（天）")


class ModelTrainRequest(BaseModel):
    """传统机器学习模型训练请求体。

    在已创建的因子数据集上训练一个指定类型的模型（如 LightGBM/MLP/Lasso）。
    params 为透传给底层模型的超参数字典，键值随 model_type 不同而异；
    各字段语义见对应 Field 的 description。
    """
    name: str = Field(description="模型名称")
    dataset: str = Field(description="数据集名称")
    model_type: str = Field(default="lgb", description="模型类型: lgb, mlp, lasso")
    params: dict[str, Any] = Field(default_factory=dict, description="模型参数")


class SignalGenerateRequest(BaseModel):
    """传统模型信号生成请求体。

    用已训练模型在指定合约与时间区间上推理，产出可供回测复用的信号并以 name 落库。
    各字段语义见对应 Field 的 description。
    """
    name: str = Field(description="信号名称")
    model: str = Field(description="模型名称")
    vt_symbols: list[str] = Field(description="合约列表")
    start: date = Field(description="开始日期")
    end: date = Field(description="结束日期")


class BacktestRunRequest(BaseModel):
    """传统信号策略回测请求体。

    基于已生成的信号在指定资金与时间区间上运行回测，benchmark 为可选基准合约
    （缺省则不计算超额收益）。各字段语义见对应 Field 的 description。
    """
    name: str = Field(description="回测名称")
    signal: str = Field(description="信号名称")
    capital: float = Field(default=1_000_000, description="初始资金")
    start: date = Field(description="回测开始日期")
    end: date = Field(description="回测结束日期")
    benchmark: Optional[str] = Field(default=None, description="基准合约")


class CNNBacktestRequest(BaseModel):
    """CNN 模型回测请求体，同时服务 classification / regression / path_class 三种 objective。

    统一封装回测所需的模型名称、资金、日期区间、出场策略及交易成本参数。
    ``veto_threshold`` 仅在 objective='path_class' 时生效：当「先触止损」类概率
    prob_sl 达到该阈值时否决买入信号；其他 objective 下该字段无意义，保留默认值 1.0
    即可保持与旧版行为完全兼容。
    """
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
    veto_threshold: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "path_class 专用：先触止损概率 prob_sl >= 该值时否决买入；"
            "默认 1.0 等效关闭否决（仅 prob_sl 饱和为 1.0 的极端行会被否决，属可接受边界）。"
            "对单标量模型无效。"
        ),
    )


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
    objective: Literal["classification", "regression", "path_class"] = Field(
        default="classification",
        description=(
            "预测目标：classification=方向二分类（输出概率）；"
            "regression=直接预测涨跌幅（输出预测收益）；"
            "path_class=路径形态四分类（先触止盈/先触止损/到期小涨/到期小跌，需配合 label_spec.mode='oco'）"
        ),
    )


class ObservationRole(str, Enum):
    """CNN 观测分组的语义角色枚举。

    标记一组证券在模型输入中扮演的角色，用于按语义组织多通道输入。

    Attributes:
        TARGET: 预测目标本身。
        MARKET: 大盘/市场指数等宏观参照。
        SECTOR: 目标所属行业/板块。
        LEADERS: 板块龙头或强相关个股。
        CUSTOM: 自定义分组，语义由调用方约定。
    """

    TARGET = "target"
    MARKET = "market"
    SECTOR = "sector"
    LEADERS = "leaders"
    CUSTOM = "custom"


class ObservationGroup(BaseModel):
    """CNN 输入的语义观测分组。

    把一组证券按其在模型中的语义角色（见 ObservationRole）聚合，供 CNN 构造多通道输入。
    role 指定该组角色，name 为展示名称，symbols 为组内证券列表（可为空）。
    """
    role: ObservationRole = Field(description="分组角色")
    name: str = Field(description="分组展示名称")
    symbols: list[str] = Field(default_factory=list, description="该分组内的证券列表")


class LabelMode(str, Enum):
    """CNN 标签的取值口径枚举。

    决定 LabelSpec 如何从未来若干 bar 推导监督标签。

    Attributes:
        NEXT_BAR: 以下一根 bar 的收益作为标签。
        HORIZON_BARS: 以未来 horizon 根 bar 的累计收益作为标签（需指定 horizon）。
        SESSION_CLOSE: 以当日收盘为锚的日内/到收盘收益。
        NEXT_SESSION_CLOSE: 以下一交易日收盘为锚的收益。
        OCO: 三重障碍（止盈/止损/到期）路径依赖标签，配合 take_profit/stop_loss/max_hold 使用。
    """

    NEXT_BAR = "next_bar"
    HORIZON_BARS = "horizon_bars"
    SESSION_CLOSE = "session_close"
    NEXT_SESSION_CLOSE = "next_session_close"
    OCO = "oco"


class LabelSpec(BaseModel):
    """CNN 统一标签定义。

    集中描述如何由未来行情推导监督标签：mode 选择口径（见 LabelMode），threshold/neutral_policy
    控制小波动去噪，price_ref 决定收益计价口径并与回测撮合成交价对齐；take_profit/stop_loss/
    max_hold/stop_first 仅在 mode=oco 时生效（其中 take_profit、stop_loss 必填且 > 0）。
    各字段的取值范围与单位见对应 Field 的 description。
    """
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
    """数据集列表项信息。

    用于数据集列表/详情接口的响应条目。

    Attributes:
        name: 数据集名称。
        created_at: 创建时间字符串（ISO 格式）；未知时为 None。
        size_kb: 落盘体积，单位 KB；缺省 0.0。
    """

    name: str
    created_at: Optional[str] = None
    size_kb: float = 0.0


class ModelInfo(BaseModel):
    """模型列表项信息。

    用于模型列表/详情接口的响应条目。

    Attributes:
        name: 模型名称。
        model_type: 模型类型（如 lgb/mlp/lasso）；未知时为空串。
        created_at: 创建时间字符串（ISO 格式）；未知时为 None。
        size_kb: 落盘体积，单位 KB；缺省 0.0。
    """

    name: str
    model_type: str = ""
    created_at: Optional[str] = None
    size_kb: float = 0.0


class SignalInfo(BaseModel):
    """信号列表项信息。

    用于信号列表/详情接口的响应条目。

    Attributes:
        name: 信号名称。
        created_at: 创建时间字符串（ISO 格式）；未知时为 None。
        size_kb: 落盘体积，单位 KB；缺省 0.0。
    """

    name: str
    created_at: Optional[str] = None
    size_kb: float = 0.0


class ContractSetting(BaseModel):
    """单合约的交易撮合参数设置。

    描述回测/实盘引擎对某一合约的费率、合约乘数、最小变动价位及交易限制。
    带 None 默认的可选字段在序列化时不写入 JSON（保持配置文件干净），
    None 即表示「沿用引擎默认/不施加该项限制」。

    Attributes:
        vt_symbol: 合约代码，如 "000001.SZSE"。
        long_rate: 买入方向佣金率，单边；缺省 0.0。
        short_rate: 卖出方向佣金率，单边；缺省 0.0。
        size: 合约乘数；股票/基金通常为 1.0。
        pricetick: 最小变动价位，单位元；缺省 0.01。
        stamp_duty: 卖出印花税率；None 表示用引擎默认（A 股 2023-08 起为 0.0005）。
        slippage: 每笔成交不利滑点率；None 表示沿用引擎默认。
        limit_ratio: 单边涨跌停比例（如 0.1 表示 10%）；None 表示无限制（如可转债）。
        t_plus1: 是否启用 T+1 卖出限制；None 表示沿用引擎默认。
    """

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
    """系统状态响应体。

    汇报后端运行环境的关键能力与配置，供前端探活/诊断展示。

    Attributes:
        version: 后端版本号。
        torch_available: PyTorch 是否可用（决定 CNN 相关功能是否可跑）。
        torch_device: PyTorch 计算设备，如 "cpu"/"cuda"/"mps"。
        data_path: 本地数据根目录绝对路径。
        tushare_token_set: 是否已配置 Tushare token。
        providers: 已注册数据源的描述列表，每项为一个数据源的元信息字典；缺省空列表。
    """

    version: str
    torch_available: bool
    torch_device: str
    data_path: str
    tushare_token_set: bool
    providers: list[dict] = Field(default_factory=list)


class BarDataInfo(BaseModel):
    """已下载 K 线数据的概览信息。

    描述某合约某周期落地数据的覆盖区间与体量，供数据资源列表展示。

    Attributes:
        vt_symbol: 合约代码。
        interval: K 线周期，如 d/1m/5m。
        start_date: 数据起始日期字符串；无数据时为 None。
        end_date: 数据结束日期字符串；无数据时为 None。
        count: K 线根数；缺省 0。
        size_kb: 落盘体积，单位 KB；缺省 0.0。
    """

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
