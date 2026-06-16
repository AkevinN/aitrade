"""
Alpha AI 投研模块 REST API。

本模块提供 Alpha AI 投研系统的 HTTP API 接口，路由前缀为 /api/alpha。
所有长时间运行的操作（数据下载、模型训练、回测等）都通过异步任务机制处理，
避免阻塞 HTTP 请求。

API 分类：
1. 模块状态：/status
2. 任务管理：/tasks
3. 数据下载：/data/download
4. 数据集管理：/datasets
5. 模型管理：/models
6. 信号管理：/signals
7. 策略回测：/backtest
8. 合约配置：/contracts
9. K线数据：/bar-data

版本历史：
- v1.0 (2026-04-03): 适配 aitrade 后端，移除 vnpy 依赖
"""

import logging
from datetime import date, datetime
from typing import Any

import polars as pl
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from .. import __version__
from ..alpha.lab_utils import normalize_vt_symbol
from ..config import ALPHA_LAB_PATH
from ..models import (
    TaskType,
    DataAggregateRequest,
    DataDownloadRequest,
    DataResourceMergeRequest,
    DatasetCreateRequest,
    ModelTrainRequest,
    RelocateBarIntervalRequest,
    SignalGenerateRequest,
    BacktestRunRequest,
    SymbolProfileRequest,
)
from ..profiling import Profiler, ProfileStore
from ..task import task_manager

from . import alpha_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alpha", tags=["Alpha AI"])


# =============================================================================
# Helpers
# =============================================================================

def _check_alpha_installed() -> bool:
    """探测 Alpha 投研模块是否可用。

    通过尝试导入 AlphaLab 判断依赖是否就位，各路由在动手前用它决定是返回
    503/空结果还是继续执行。导入失败不抛错，仅返回 False。

    Returns:
        AlphaLab 可导入返回 True；缺少依赖（ImportError）返回 False。
    """
    try:
        from ..alpha import AlphaLab
        return True
    except ImportError:
        return False


def _get_alpha_lab():
    """构造一个绑定到默认数据目录的 AlphaLab 实例。

    每次调用都新建实例，数据根目录固定为配置项 ALPHA_LAB_PATH。调用前应先用
    _check_alpha_installed() 确认模块可用，否则此处导入会抛 ImportError。

    Returns:
        指向 ALPHA_LAB_PATH 的 AlphaLab 实例，封装数据集/模型/信号/K线等全部本地资源访问。
    """
    from ..alpha import AlphaLab
    return AlphaLab(ALPHA_LAB_PATH)


# 业务逻辑已迁移至 alpha_service.py：
#   _build_feature_dataset / _apply_default_preprocessing / _pick_bar_provider


def _normalize_market_interval(interval: str | None) -> str:
    """把请求里的周期写法归一化为引擎内部统一的周期代码。

    兼容前端/旧接口的多种别名（如 "daily"→"d"、"minute"/"m"/"1m"→"1m"、
    "1h"→"60m"、"weekly"→"w"），大小写与首尾空白都会被忽略。无法识别的取值
    原样返回（小写去空白后），交由各路由自行做白名单校验。

    Args:
        interval: 原始周期写法；None 或空串按空字符串处理。

    Returns:
        归一化后的周期代码（如 "d"/"1m"/"5m"/"60m"/"w"）；未命中映射表时返回清洗后的原值。
    """
    raw = (interval or "").strip().lower()
    mapping = {
        "daily": "d",
        "d": "d",
        "minute": "1m",
        "m": "1m",
        "1m": "1m",
        "5m": "5m",
        "10m": "10m",
        "15m": "15m",
        "30m": "30m",
        "60m": "60m",
        "1h": "60m",
        "weekly": "w",
        "w": "w",
    }
    return mapping.get(raw, raw)


def _normalize_symbol_list(vt_symbols: list[str]) -> list[str]:
    """逐个归一化合约代码并去重，保留首次出现的顺序。

    跳过空字符串项，对每个非空项调用 normalize_vt_symbol 标准化后用 dict.fromkeys
    去重（去重在归一化之后，所以等价别名会被合并）。

    Args:
        vt_symbols: 原始合约代码列表，可能含空串或重复项。

    Returns:
        归一化、去重且保序的合约代码列表；输入为空或全为空串时返回空列表。
    """
    return list(dict.fromkeys(normalize_vt_symbol(item) for item in vt_symbols if item))


def _normalize_optional_symbol(vt_symbol: str | None) -> str | None:
    """归一化可选的单个合约代码，保留"未提供"语义。

    用于 benchmark 等可空字段：有值时标准化，None 或空串则原样透传，不会把"未指定"
    误转成具体合约。

    Args:
        vt_symbol: 合约代码；None 或空串表示未指定。

    Returns:
        归一化后的合约代码；输入为 None/空串时按原值返回（保持假值语义）。
    """
    return normalize_vt_symbol(vt_symbol) if vt_symbol else vt_symbol


# _load_required_local_bar_df 已迁移至 alpha_service.py


# =============================================================================
# Services (inlined, no separate files)
# =============================================================================

# _download_bar_data 已迁移至 alpha_service.py


# _aggregate_data 已迁移至 alpha_service.py


# _create_dataset 已迁移至 alpha_service.py


# _train_model 已迁移至 alpha_service.py


# _generate_signal 已迁移至 alpha_service.py


# _run_backtest 已迁移至 alpha_service.py


# =============================================================================
# 模块状态
# =============================================================================

@router.get("/status")
async def get_alpha_status() -> dict:
    """获取 Alpha 模块状态，供前端判断功能是否可用。

    探测依赖是否就位，并附带版本号与数据目录信息。

    Returns:
        状态字典，含字段：
            - installed: 模块是否已安装（依赖可导入）。
            - version: 后端版本号。
            - lab_path: 数据目录路径（字符串）。
            - lab_exists: 数据目录是否存在；未安装时恒为 False。
    """
    installed = _check_alpha_installed()
    return {
        "installed": installed,
        "version": __version__,
        "lab_path": str(ALPHA_LAB_PATH),
        "lab_exists": ALPHA_LAB_PATH.exists() if installed else False,
    }


# =============================================================================
# 任务管理
# =============================================================================

@router.get("/tasks")
async def list_tasks(
    request: Request,
    status: str | None = None,
    task_type: str | None = None,
    include_history: bool = False,
    limit: int = 200,
    history_days: int = 90,
) -> list[dict]:
    """获取任务状态列表（R2.3）。

    默认行为（无参数）与现状完全一致：返回内存中所有任务，按 updated_at 倒序，上限 200。
    include_history=True 时再叠加磁盘归档历史，同 task_id 以内存版本为准。

    Args:
        request:         FastAPI 请求对象，用于从 app.state.history_store 取历史存储；缺失时回退到 task_manager 内置 store。
        status:          按状态过滤（completed/failed/running/pending）；None 表示不过滤。
        task_type:       按任务类型过滤（data_download/model_train 等）；None 表示不过滤。
        include_history: True 时合并归档历史（同 task_id 以内存为准），默认 False。
        limit:           最多返回条数，默认 200。
        history_days:    include_history=True 时回看天数，默认 90，传入值会被夹到 [1, 365]。

    Returns:
        任务字典列表，每项为 Task.model_dump(mode="json") 的结果（datetime 已序列化为 ISO 字符串），
        按 updated_at 倒序、长度不超过 limit；无任务时返回空列表。
    """
    # 内存任务
    mem_tasks = task_manager.get_all_tasks()

    # 过滤内存任务
    if status is not None:
        mem_tasks = [t for t in mem_tasks if t.status.value == status]
    if task_type is not None:
        mem_tasks = [t for t in mem_tasks if t.type.value == task_type]

    # 按 updated_at 倒序
    mem_tasks = sorted(mem_tasks, key=lambda t: t.updated_at, reverse=True)
    # model_dump(mode="json") 将所有 datetime 序列化为 ISO 字符串（T 分隔），
    # 确保与历史侧字符串格式一致，排序键可直接字符串比较
    mem_dicts = [t.model_dump(mode="json") for t in mem_tasks]

    if not include_history:
        return mem_dicts[:limit]

    # 合并历史（R2.3）：同 task_id 以内存为准
    mem_ids = {t.task_id for t in mem_tasks}

    # 获取 history_store（从 app.state 注入，或从全局 task_manager 取）
    hist_store = getattr(getattr(request, "app", None), "state", None)
    hist_store = getattr(hist_store, "history_store", None)
    if hist_store is None:
        # 回退到 task_manager 内置 store
        hist_store = task_manager._history_store

    today = date.today()
    # 回看 history_days 天（上限 365），避免全量扫描
    history_days = max(1, min(history_days, 365))
    from datetime import timedelta
    hist_start = today - timedelta(days=history_days - 1)

    hist_records = hist_store.query(
        status=status,
        task_type=task_type,
        start=hist_start,
        end=today,
        limit=None,
    )

    # 历史中去掉已在内存的 task_id
    extra_hist = [r for r in hist_records if r.get("task_id") not in mem_ids]

    # 合并后按 updated_at 字符串倒序截断
    # mem_dicts 已经过 model_dump(mode="json")，updated_at 为 ISO "T" 格式字符串，
    # 与 extra_hist（从 JSONL 读出，同为 ISO "T" 格式）可直接字符串比较
    combined = mem_dicts + extra_hist
    combined.sort(key=lambda d: d.get("updated_at") or "", reverse=True)
    return combined[:limit]


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict:
    """按 task_id 查询单个任务的当前状态。

    Args:
        task_id: 任务唯一标识，由 create_task 返回。

    Returns:
        该任务的 model_dump() 字典（含状态、进度、结果等字段）。

    Raises:
        HTTPException: task_id 在内存任务表中不存在时抛 404。
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, f"任务 {task_id} 不存在")
    return task.model_dump()


# =============================================================================
# 数据下载
# =============================================================================

@router.post("/data/download")
async def start_data_download(req: DataDownloadRequest) -> dict:
    """提交原始市场数据下载任务并立即返回（异步执行）。

    入队前会归一化合约代码、校验数据类型与下载周期，校验通过才创建后台任务。

    Args:
        req: 下载请求；vt_symbols 为合约列表，data_kind 当前仅支持 "bar"，
            source_interval/interval 指定下载周期（归一化后须属于 d/1m/5m/15m/30m/60m/w）。

    Returns:
        含 task_id（用于后续轮询任务状态）与提示文案 message 的字典。

    Raises:
        HTTPException: Alpha 未安装时 503；data_kind 非 "bar" 或周期不在白名单时 400。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    req = req.model_copy(update={"vt_symbols": _normalize_symbol_list(req.vt_symbols)})
    if req.data_kind != "bar":
        raise HTTPException(400, "当前版本仅支持原始K线下载，历史 Tick 请通过导入功能准备")

    source_interval = _normalize_market_interval(req.source_interval or req.interval or "d")
    if source_interval not in {"d", "1m", "5m", "15m", "30m", "60m", "w"}:
        raise HTTPException(400, "source_interval 仅支持 d/1m/5m/15m/30m/60m/w")

    task_id = task_manager.create_task(
        TaskType.DATA_DOWNLOAD,
        req.model_dump(),
        title=f"下载 {source_interval} 原始K线",
        entity_type="data",
        entity_name=",".join(req.vt_symbols[:3]),
    )

    def execute(on_progress=None):
        """后台任务体：执行原始K线下载，并通过 on_progress 回调上报进度。"""
        return alpha_service._download_bar_data(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)

    return {"task_id": task_id, "message": "下载任务已启动"}


@router.post("/data/aggregate")
async def start_data_aggregate(req: DataAggregateRequest) -> dict:
    """提交本地数据聚合任务并立即返回（异步执行）。

    把已有的细粒度 K 线聚合为更粗的派生周期。入队前归一化合约代码并校验日期区间。

    Args:
        req: 聚合请求；vt_symbols 为合约列表，target_interval 为目标派生周期，
            start/end 为聚合日期区间（含端点）。

    Returns:
        含 task_id 与提示文案 message 的字典。

    Raises:
        HTTPException: Alpha 未安装时 503；start 晚于 end 时 400。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    req = req.model_copy(update={"vt_symbols": _normalize_symbol_list(req.vt_symbols)})
    if req.start > req.end:
        raise HTTPException(400, "开始日期不能晚于结束日期")

    task_id = task_manager.create_task(
        TaskType.DATA_AGGREGATE,
        req.model_dump(),
        title=f"聚合 {req.target_interval} 派生K线",
        entity_type="data",
        entity_name=",".join(req.vt_symbols[:3]),
    )

    def execute(on_progress=None):
        """后台任务体：执行本地数据聚合，并通过 on_progress 回调上报进度。"""
        return alpha_service._aggregate_data(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "聚合任务已启动"}


# =============================================================================
# 标的画像（只读诊断）
# =============================================================================

@router.post("/profiling")
async def create_symbol_profile(payload: dict[str, Any]) -> Any:
    """同步生成标的画像（只读诊断），不进入任务队列。

    手动校验请求体而非用 FastAPI 自动绑定，以便把校验失败转成 400 友好提示；
    通过后归一化周期与合约代码并调用 Profiler 即时计算返回。

    Args:
        payload: 原始请求体，按 SymbolProfileRequest 校验。关键字段：vt_symbol（主标的）、
            interval（周期，归一化后须属于 d/1m/5m/10m/15m/30m/60m/w）、as_of（诊断截止日）、
            lookback_days（回看天数）、observation_symbols（参照标的列表）、
            with_suggestion（是否附建议）、persist（是否持久化产物）。

    Returns:
        Profiler.profile 的画像结果（结构由画像引擎决定）。

    Raises:
        HTTPException: Alpha 未安装时 503；请求体校验失败或周期非法时 400。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    try:
        req = SymbolProfileRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(400, f"画像参数非法: {exc.errors()[0]['msg']}")

    interval = _normalize_market_interval(req.interval)
    if interval not in {"d", "1m", "5m", "10m", "15m", "30m", "60m", "w"}:
        raise HTTPException(400, "interval 仅支持 d/1m/5m/10m/15m/30m/60m/w")

    lab = _get_alpha_lab()
    profiler = Profiler(lab)
    return profiler.profile(
        vt_symbol=normalize_vt_symbol(req.vt_symbol),
        interval=interval,
        as_of=req.as_of,
        lookback_days=req.lookback_days,
        observation_symbols=_normalize_symbol_list(req.observation_symbols),
        with_suggestion=req.with_suggestion,
        persist=req.persist,
    )


@router.get("/profiling/artifacts")
async def list_symbol_profile_artifacts() -> list[str]:
    """列出全部已持久化的画像产物 id。

    Returns:
        画像产物 id 列表；无产物时返回空列表。

    Raises:
        HTTPException: Alpha 未安装时 503。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    return ProfileStore().list_ids()


@router.get("/profiling/{artifact_id}")
async def get_symbol_profile_artifact(artifact_id: str) -> Any:
    """按 id 读取一份已持久化的画像产物。

    Args:
        artifact_id: 画像产物 id，来自 list_symbol_profile_artifacts。

    Returns:
        ProfileStore.load 返回的画像产物内容。

    Raises:
        HTTPException: Alpha 未安装时 503；产物文件不存在时 404。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    store = ProfileStore()
    try:
        return store.load(artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


# =============================================================================
# 数据集管理
# =============================================================================

@router.get("/datasets")
async def list_datasets() -> list[str]:
    """列出全部数据集名称（升序）。

    Returns:
        数据集名称列表，按字母升序；Alpha 未安装时返回空列表（不报错）。
    """
    if not _check_alpha_installed():
        return []
    lab = _get_alpha_lab()
    return sorted(lab.list_all_datasets())


@router.get("/datasets/{name}")
async def get_dataset(name: str) -> dict:
    """获取单个数据集的概要信息。

    样本数优先取 learn_df 行数，其次回退到 df 行数，二者皆无则为 0。

    Args:
        name: 数据集名称。

    Returns:
        含字段 name、feature_count（特征表达式数量）、sample_count（样本行数）、
        label_expression（标签表达式，无则为空串）的字典。

    Raises:
        HTTPException: Alpha 未安装时 503；数据集不存在时 404。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    lab = _get_alpha_lab()
    dataset = lab.load_dataset(name)
    if dataset is None:
        raise HTTPException(404, f"数据集 {name} 不存在")

    sample_count = 0
    if hasattr(dataset, "learn_df") and dataset.learn_df is not None:
        sample_count = len(dataset.learn_df)
    elif hasattr(dataset, "df") and dataset.df is not None:
        sample_count = len(dataset.df)

    return {
        "name": name,
        "feature_count": len(dataset.feature_expressions),
        "sample_count": sample_count,
        "label_expression": dataset.label_expression or "",
    }


@router.post("/datasets/create")
async def start_create_dataset(req: DatasetCreateRequest) -> dict:
    """提交数据集创建任务并立即返回（异步执行）。

    入队前归一化合约代码，并要求至少选择一个特征库。

    Args:
        req: 数据集创建请求；vt_symbols 为合约列表，features 为特征库选择（不可为空），
            其余字段（标签、区间等）按请求模型定义。

    Returns:
        含 task_id 与提示文案 message 的字典。

    Raises:
        HTTPException: Alpha 未安装时 503；features 为空时 400。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    req = req.model_copy(update={"vt_symbols": _normalize_symbol_list(req.vt_symbols)})
    if not req.features:
        raise HTTPException(400, "至少需要选择一个特征库")

    task_id = task_manager.create_task(TaskType.DATASET_CREATE, req.model_dump())

    def execute(on_progress=None):
        """后台任务体：执行数据集创建，并通过 on_progress 回调上报进度。"""
        return alpha_service._create_dataset(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)

    return {"task_id": task_id, "message": "数据集创建任务已启动"}


@router.delete("/datasets/{name}")
async def remove_dataset(name: str) -> dict:
    """删除指定数据集。

    Args:
        name: 待删除的数据集名称。

    Returns:
        删除成功时返回 {"success": True, "message": ...}。

    Raises:
        HTTPException: Alpha 未安装时 503；数据集不存在（删除返回假值）时 404。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    lab = _get_alpha_lab()
    result = lab.remove_dataset(name)
    if result:
        return {"success": True, "message": f"数据集 {name} 已删除"}
    raise HTTPException(404, f"数据集 {name} 不存在")


# =============================================================================
# 模型管理
# =============================================================================

@router.get("/models")
async def list_models() -> list[str]:
    """列出全部模型名称（升序）。

    Returns:
        模型名称列表，按字母升序；Alpha 未安装时返回空列表（不报错）。
    """
    if not _check_alpha_installed():
        return []
    lab = _get_alpha_lab()
    return sorted(lab.list_all_models())


@router.get("/models/{name}")
async def get_model(name: str) -> dict:
    """获取单个模型的概要信息。

    Args:
        name: 模型名称。

    Returns:
        含字段 name 与 model_type（模型类名）的字典。

    Raises:
        HTTPException: Alpha 未安装时 503；模型不存在时 404。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    lab = _get_alpha_lab()
    model = lab.load_model(name)
    if model is None:
        raise HTTPException(404, f"模型 {name} 不存在")

    return {
        "name": name,
        "model_type": type(model).__name__,
    }


@router.post("/models/train")
async def start_train_model(req: ModelTrainRequest) -> dict:
    """提交模型训练任务并立即返回（异步执行）。

    Args:
        req: 模型训练请求，按 ModelTrainRequest 定义（数据集、模型类型、超参等）。

    Returns:
        含 task_id 与提示文案 message 的字典。

    Raises:
        HTTPException: Alpha 未安装时 503。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    task_id = task_manager.create_task(TaskType.MODEL_TRAIN, req.model_dump())

    def execute(on_progress=None):
        """后台任务体：执行模型训练，并通过 on_progress 回调上报进度。"""
        return alpha_service._train_model(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)

    return {"task_id": task_id, "message": "模型训练任务已启动"}


@router.delete("/models/{name}")
async def remove_model(name: str) -> dict:
    """删除指定模型。

    Args:
        name: 待删除的模型名称。

    Returns:
        删除成功时返回 {"success": True, "message": ...}。

    Raises:
        HTTPException: Alpha 未安装时 503；模型不存在（删除返回假值）时 404。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    lab = _get_alpha_lab()
    result = lab.remove_model(name)
    if result:
        return {"success": True, "message": f"模型 {name} 已删除"}
    raise HTTPException(404, f"模型 {name} 不存在")


# =============================================================================
# 信号管理
# =============================================================================

@router.get("/signals")
async def list_signals() -> list[str]:
    """列出全部信号名称（升序）。

    Returns:
        信号名称列表，按字母升序；Alpha 未安装时返回空列表（不报错）。
    """
    if not _check_alpha_installed():
        return []
    lab = _get_alpha_lab()
    return sorted(lab.list_all_signals())


@router.get("/signals/{name}")
async def get_signal(name: str) -> dict:
    """获取单个信号的概要信息与末尾预览。

    预览取信号表最后 100 行，每个单元格转成字符串（None 保留为 None）；预览过程若出错
    则静默跳过、preview 留空，不影响主体信息返回。

    Args:
        name: 信号名称。

    Returns:
        含字段 name、row_count（总行数）、columns（列名列表）、
        preview（末尾最多 100 行的字符串化记录列表）的字典。

    Raises:
        HTTPException: Alpha 未安装时 503；信号不存在时 404。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    lab = _get_alpha_lab()
    signal_df = lab.load_signal(name)
    if signal_df is None:
        raise HTTPException(404, f"信号 {name} 不存在")

    preview = []
    try:
        for row in signal_df.tail(100).iter_rows(named=True):
            record = {k: str(v) if v is not None else None for k, v in row.items()}
            preview.append(record)
    except Exception:
        pass

    return {
        "name": name,
        "row_count": len(signal_df),
        "columns": list(signal_df.columns),
        "preview": preview,
    }


@router.post("/signals/generate")
async def start_generate_signal(req: SignalGenerateRequest) -> dict:
    """提交信号生成任务并立即返回（异步执行）。

    入队前归一化合约代码。

    Args:
        req: 信号生成请求；vt_symbols 为合约列表，其余字段（模型、区间等）按请求模型定义。

    Returns:
        含 task_id 与提示文案 message 的字典。

    Raises:
        HTTPException: Alpha 未安装时 503。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    req = req.model_copy(update={"vt_symbols": _normalize_symbol_list(req.vt_symbols)})

    task_id = task_manager.create_task(TaskType.SIGNAL_GENERATE, req.model_dump())

    def execute(on_progress=None):
        """后台任务体：执行信号生成，并通过 on_progress 回调上报进度。"""
        return alpha_service._generate_signal(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)

    return {"task_id": task_id, "message": "信号生成任务已启动"}


@router.delete("/signals/{name}")
async def remove_signal(name: str) -> dict:
    """删除指定信号。

    Args:
        name: 待删除的信号名称。

    Returns:
        删除成功时返回 {"success": True, "message": ...}。

    Raises:
        HTTPException: Alpha 未安装时 503；信号不存在（删除返回假值）时 404。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    lab = _get_alpha_lab()
    result = lab.remove_signal(name)
    if result:
        return {"success": True, "message": f"信号 {name} 已删除"}
    raise HTTPException(404, f"信号 {name} 不存在")


# =============================================================================
# 策略回测
# =============================================================================

@router.post("/backtest/run")
async def start_backtest(req: BacktestRunRequest) -> dict:
    """提交策略回测任务并立即返回（异步执行）。

    入队前归一化可选的 benchmark 合约（未指定则保持为空，不会被误转成具体合约）。

    Args:
        req: 回测请求；benchmark 为可选基准合约，其余字段（信号、区间、费率等）按请求模型定义。

    Returns:
        含 task_id 与提示文案 message 的字典。

    Raises:
        HTTPException: Alpha 未安装时 503。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    req = req.model_copy(update={"benchmark": _normalize_optional_symbol(req.benchmark)})

    task_id = task_manager.create_task(TaskType.BACKTEST_RUN, req.model_dump())

    def execute(on_progress=None):
        """后台任务体：执行策略回测，并通过 on_progress 回调上报进度。"""
        return alpha_service._run_backtest(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)

    return {"task_id": task_id, "message": "回测任务已启动"}


# =============================================================================
# 合约配置
# =============================================================================

@router.get("/contracts")
async def get_contract_settings() -> dict:
    """获取全部合约的回测/交易参数配置。

    Returns:
        合约配置字典（key 为 vt_symbol，value 为费率/合约乘数等参数）；
        Alpha 未安装时返回空字典（不报错）。
    """
    if not _check_alpha_installed():
        return {}
    lab = _get_alpha_lab()
    return lab.load_contract_settings()


@router.post("/contracts")
async def add_contract_setting(
    vt_symbol: str,
    long_rate: float = 0.0001,
    short_rate: float = 0.0001,
    size: float = 1,
    pricetick: float = 0.01,
    stamp_duty: float | None = None,
    slippage: float | None = None,
    limit_ratio: float | None = None,
    t_plus1: bool | None = None,
) -> dict:
    """添加或更新单个合约的回测/交易参数配置。

    可选字段 stamp_duty / slippage / limit_ratio / t_plus1 传值时写入 JSON；
    不传（None）则忽略，保持已有配置不被意外清空。
    t_plus1 字段本任务只打通写入与存储，引擎消费在下一任务实现。

    Args:
        vt_symbol: 合约代码，写入前会归一化。
        long_rate: 多头手续费率（成交额占比），默认 0.0001。
        short_rate: 空头手续费率（成交额占比），默认 0.0001。
        size: 合约乘数（每点价值），默认 1。
        pricetick: 最小价格变动单位，默认 0.01。
        stamp_duty: 印花税率；None 表示不更新该字段。
        slippage: 滑点（按价格单位计）；None 表示不更新该字段。
        limit_ratio: 涨跌停限制比例；None 表示不更新该字段。
        t_plus1: 是否启用 T+1 交易规则；None 表示不更新该字段。

    Returns:
        保存成功时返回 {"success": True, "message": ...}。

    Raises:
        HTTPException: Alpha 未安装时 503。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    vt_symbol = normalize_vt_symbol(vt_symbol)
    lab = _get_alpha_lab()
    lab.add_contract_setting(
        vt_symbol, long_rate, short_rate, size, pricetick,
        stamp_duty=stamp_duty,
        slippage=slippage,
        limit_ratio=limit_ratio,
        t_plus1=t_plus1,
    )
    return {"success": True, "message": f"合约 {vt_symbol} 配置已保存"}


# =============================================================================
# 统一数据资源
# =============================================================================

@router.get("/data/resources")
async def get_data_resources() -> dict[str, Any]:
    """获取统一的数据资源清单：原始 K 线、历史 Tick 与派生周期。

    Returns:
        含 raw_bars / raw_ticks / derived_bars 三个列表的字典；
        Alpha 未安装时返回三者皆空的字典（不报错）。
    """
    if not _check_alpha_installed():
        return {"raw_bars": [], "raw_ticks": [], "derived_bars": []}

    lab = _get_alpha_lab()
    return lab.list_data_resources()


@router.get("/data/resources/{kind}/{key}")
async def get_data_resource_detail(
    kind: str,
    key: str,
    limit: int = 100,
    before: str | None = None,
) -> dict[str, Any]:
    """查看单个数据资源的详情与分页预览。

    Args:
        kind: 资源类别（如 raw_bar/raw_tick/derived_bar），由底层 lab 校验。
        key: 资源主键，定位具体文件。
        limit: 预览返回条数，默认 100。
        before: 游标时间（ISO 字符串），只取早于该时间的数据；None 表示从最新开始。

    Returns:
        资源详情字典，含预览数据与分页游标（具体结构由底层 lab 决定）。

    Raises:
        HTTPException: Alpha 未安装时 503；资源不存在时 404；参数非法时 400。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    lab = _get_alpha_lab()
    try:
        return lab.get_data_resource_detail(kind, key, limit=limit, before=before)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.delete("/data/resources/{kind}/{key}")
async def delete_data_resource(kind: str, key: str) -> dict[str, Any]:
    """删除单个原始或派生数据资源。

    Args:
        kind: 资源类别（如 raw_bar/derived_bar 等）。
        key: 资源主键，定位待删除文件。

    Returns:
        删除成功时返回 {"success": True, "message": ...}。

    Raises:
        HTTPException: Alpha 未安装时 503；资源不存在（删除返回假值）时 404。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    lab = _get_alpha_lab()
    if lab.delete_data_resource(kind, key):
        return {"success": True, "message": f"已删除 {kind}/{key}"}
    raise HTTPException(404, f"资源不存在: {kind}/{key}")


@router.patch("/data/resources/raw_bar/{key}/interval")
async def relocate_raw_bar_interval(key: str, req: RelocateBarIntervalRequest) -> dict[str, Any]:
    """更正原始 K 线资源的存储周期，把文件移动到对应周期目录。

    用于修正误归类的原始 K 线（如把当成日线存的分钟线挪到正确目录）。

    Args:
        key: 原始 K 线资源主键。
        req: 含目标 interval 的请求；interval 归一化后须属于 d/1m/5m/15m/30m/60m。

    Returns:
        迁移结果字典（含新位置等信息，结构由底层 lab 决定）。

    Raises:
        HTTPException: Alpha 未安装时 503；interval 不在白名单时 400；
            资源不存在时 404；底层校验失败时 400。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    interval = _normalize_market_interval(req.interval)
    if interval not in {"d", "1m", "5m", "15m", "30m", "60m"}:
        raise HTTPException(400, "interval 仅支持 d/1m/5m/15m/30m/60m")

    lab = _get_alpha_lab()
    try:
        return lab.relocate_raw_bar_interval(key, interval)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post("/data/resources/merge/preview")
async def preview_data_resource_merge(req: DataResourceMergeRequest) -> dict[str, Any]:
    """预检若干上传批次能否手动合并为正式原始资源（不落盘）。

    Args:
        req: 合并请求；kind 为资源类别，keys 为待合并的批次主键列表。

    Returns:
        预检结果字典（是否可合并、冲突/重叠提示等，结构由底层 lab 决定）。

    Raises:
        HTTPException: Alpha 未安装时 503；批次不可预检（如类别不一致）时 400。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    lab = _get_alpha_lab()
    try:
        return lab.preview_merge_import_batches(kind=req.kind, keys=req.keys)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post("/data/resources/merge")
async def merge_data_resource_batches(req: DataResourceMergeRequest) -> dict[str, Any]:
    """把若干上传批次合并并写入正式原始资源（落盘）。

    Args:
        req: 合并请求；kind 为资源类别，keys 为待合并的批次主键列表。

    Returns:
        合并结果字典（success=True 及产物信息）。

    Raises:
        HTTPException: Alpha 未安装时 503；底层校验失败（ValueError）时 400；
            合并结果 success 为假值时按其 reason 返回 400。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    lab = _get_alpha_lab()
    try:
        result = lab.merge_import_batches(kind=req.kind, keys=req.keys)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not result.get("success"):
        raise HTTPException(400, result.get("reason") or "批次不可合并")
    return result


# =============================================================================
# K线数据列表（兼容旧前端）
# =============================================================================

@router.get("/bar-data", deprecated=True)
async def get_bar_data_list() -> dict:
    """[已废弃] 获取已有日线和 1 分钟原始K线列表（兼容旧前端）。

    请改用 `GET /api/alpha/data/resources` 获取完整的原始/派生/批次资源列表。

    Returns:
        形如 {"daily": [...], "minute": [...]} 的字典：daily 收录周期为 "d" 的
        原始K线、minute 收录周期为 "1m" 的原始K线，其余周期不纳入。每个列表项
        为一只合约的概要 {"vt_symbol", "row_count", "start", "end",
        "file_size_kb"}。Alpha 模块未安装时返回 {"daily": [], "minute": []}；
        无对应周期数据时相应列表为空。
    """
    if not _check_alpha_installed():
        return {"daily": [], "minute": []}

    lab = _get_alpha_lab()
    resources = lab.list_data_resources()
    result: dict[str, list[dict[str, Any]]] = {"daily": [], "minute": []}

    for item in resources["raw_bars"]:
        summary = {
            "vt_symbol": item["vt_symbol"],
            "row_count": item["row_count"],
            "start": item["start"],
            "end": item["end"],
            "file_size_kb": item["file_size_kb"],
        }
        if item["interval"] == "d":
            result["daily"].append(summary)
        elif item["interval"] == "1m":
            result["minute"].append(summary)

    return result


# =============================================================================
# CSV 导入（新 Tick 路由 + 旧 bar 路由）
# =============================================================================

async def _preview_csv_upload(
    file: UploadFile = File(...),
    field_mapping: str | None = None,
) -> tuple[bytes, dict[str, str] | None]:
    """校验上传的 CSV 文件并解码自定义字段映射，供预览/导入路由共用。

    校验文件名后缀为 .csv、内容非空；field_mapping 若提供须为合法 JSON 对象。

    Args:
        file: 上传的文件对象（FastAPI UploadFile）。
        field_mapping: 自定义字段映射的 JSON 字符串；None 表示使用默认映射。

    Returns:
        二元组 (contents, custom_mapping)：contents 为读取到的文件字节，
        custom_mapping 为解析后的映射字典（未提供时为 None）。

    Raises:
        HTTPException: Alpha 未安装时 503；非 .csv 后缀、文件为空、
            或 field_mapping 不是合法 JSON 对象时 400。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "仅支持 CSV 文件")

    contents: bytes = await file.read()
    if not contents:
        raise HTTPException(400, "CSV 文件为空")

    custom_mapping: dict[str, str] | None = None
    if field_mapping:
        try:
            import json
            custom_mapping = json.loads(field_mapping)
        except Exception:
            raise HTTPException(400, "field_mapping 格式错误，需为 JSON 对象")

    return contents, custom_mapping


@router.post("/bar-data/import/preview")
async def preview_csv_import(
    file: UploadFile = File(...),
    field_mapping: str | None = None,
) -> dict[str, Any]:
    """预览 bar（K 线）CSV 文件的解析结果（兼容旧前端），不落盘。

    Args:
        file: 上传的 CSV 文件。
        field_mapping: 自定义字段映射的 JSON 字符串；None 表示使用默认映射。

    Returns:
        预览结果字典（识别到的列、样例行、推断周期等，结构由底层 lab 决定）。

    Raises:
        HTTPException: 上传校验失败时 400/503（见 _preview_csv_upload）；
            CSV 解析失败时 400（友好提示）。
    """
    contents, custom_mapping = await _preview_csv_upload(file, field_mapping)
    lab = _get_alpha_lab()
    try:
        return lab.preview_csv(contents, custom_mapping, data_kind="bar")
    except HTTPException:
        raise
    except Exception as exc:
        # CSV 内容由用户上传，解析失败属可预期的客户端错误，返回 400 友好提示。
        logger.warning("bar CSV 预览失败: %s", exc)
        raise HTTPException(400, "K线 CSV 解析失败，请检查文件编码与列格式")


@router.post("/ticks/import/preview")
async def preview_tick_csv_import(
    file: UploadFile = File(...),
    field_mapping: str | None = None,
) -> dict[str, Any]:
    """预览历史 Tick CSV 文件的解析结果，不落盘。

    Args:
        file: 上传的 CSV 文件。
        field_mapping: 自定义字段映射的 JSON 字符串；None 表示使用默认映射。

    Returns:
        预览结果字典（识别到的列、样例行等，结构由底层 lab 决定）。

    Raises:
        HTTPException: 上传校验失败时 400/503（见 _preview_csv_upload）；
            CSV 解析失败时 400（友好提示）。
    """
    contents, custom_mapping = await _preview_csv_upload(file, field_mapping)
    lab = _get_alpha_lab()
    try:
        return lab.preview_csv(contents, custom_mapping, data_kind="tick")
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("tick CSV 预览失败: %s", exc)
        raise HTTPException(400, "Tick CSV 解析失败，请检查文件编码与列格式")


@router.post("/bar-data/import")
async def import_csv_data(
    interval: str = Form(default="d", description="支持 d/1m/5m/15m/30m/60m"),
    import_mode: str = Form(default="merge", description="merge=追加, replace=替换"),
    save_mode: str = Form(default="batch", description="batch=保存为待合并批次, official=写入正式资源"),
    field_mapping: str | None = Form(default=None, description="自定义字段映射 JSON"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """把上传的 bar（K 线）CSV 落盘为批次或正式原始资源（同步执行）。

    Args:
        interval: 周期，归一化后须属于 d/1m/5m/15m/30m/60m，默认 "d"。
        import_mode: "merge" 追加 / "replace" 替换，默认 "merge"。
        save_mode: "batch" 存为待合并批次 / "official" 直接写入正式资源，默认 "batch"。
        field_mapping: 自定义字段映射的 JSON 字符串；None 表示使用默认映射。
        file: 上传的 CSV 文件。

    Returns:
        导入结果字典（含 success、imported_count 等，结构由底层 lab 决定）。

    Raises:
        HTTPException: Alpha 未安装时 503；import_mode/save_mode/interval 非法或
            上传校验失败时 400；底层导入异常时 500；导入条数为 0 且失败时 400。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    if import_mode not in {"merge", "replace"}:
        raise HTTPException(400, "import_mode 参数必须是 merge 或 replace")
    if save_mode not in {"official", "batch"}:
        raise HTTPException(400, "save_mode 参数必须是 official 或 batch")

    contents, custom_mapping = await _preview_csv_upload(file, field_mapping)
    interval = _normalize_market_interval(interval)
    if interval not in {"d", "1m", "5m", "15m", "30m", "60m"}:
        raise HTTPException(400, "interval 仅支持 d/1m/5m/15m/30m/60m")

    lab = _get_alpha_lab()
    try:
        result = lab.import_csv(
            csv_content=contents,
            data_kind="bar",
            interval=interval,
            import_mode=import_mode,
            save_mode=save_mode,
            file_name=file.filename,
            custom_mapping=custom_mapping,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("bar CSV 导入失败")
        raise HTTPException(500, "K线 CSV 导入失败，请稍后重试")
    # 服务端二次校验：缺少必填字段（不依赖前端拦截），直接 400。
    if not result.get("success") and result.get("imported_count", 0) == 0:
        raise HTTPException(400, result.get("message") or "K线 CSV 导入失败")
    return result


@router.post("/ticks/import")
async def import_tick_csv_data(
    import_mode: str = Form(default="merge", description="merge=追加, replace=替换"),
    save_mode: str = Form(default="batch", description="batch=保存为待合并批次, official=写入正式资源"),
    field_mapping: str | None = Form(default=None, description="自定义字段映射 JSON"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """把上传的历史 Tick CSV 落盘为批次或正式原始资源（同步执行）。

    周期固定为 "tick"，不接受 interval 入参。

    Args:
        import_mode: "merge" 追加 / "replace" 替换，默认 "merge"。
        save_mode: "batch" 存为待合并批次 / "official" 直接写入正式资源，默认 "batch"。
        field_mapping: 自定义字段映射的 JSON 字符串；None 表示使用默认映射。
        file: 上传的 CSV 文件。

    Returns:
        导入结果字典（含 success、imported_count 等，结构由底层 lab 决定）。

    Raises:
        HTTPException: Alpha 未安装时 503；import_mode/save_mode 非法或上传校验失败时 400；
            底层导入异常时 500；导入条数为 0 且失败时 400。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    if import_mode not in {"merge", "replace"}:
        raise HTTPException(400, "import_mode 参数必须是 merge 或 replace")
    if save_mode not in {"official", "batch"}:
        raise HTTPException(400, "save_mode 参数必须是 official 或 batch")

    contents, custom_mapping = await _preview_csv_upload(file, field_mapping)

    lab = _get_alpha_lab()
    try:
        result: dict[str, Any] = lab.import_csv(
            csv_content=contents,
            data_kind="tick",
            interval="tick",
            import_mode=import_mode,
            save_mode=save_mode,
            file_name=file.filename,
            custom_mapping=custom_mapping,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("tick CSV 导入失败")
        raise HTTPException(500, "Tick CSV 导入失败，请稍后重试")
    if not result.get("success") and result.get("imported_count", 0) == 0:
        raise HTTPException(400, result.get("message") or "Tick CSV 导入失败")
    return result


# =============================================================================
# K线数据动态路由 (必须在 CSV 导入路由之后)
# =============================================================================

@router.delete("/bar-data/{interval}/{vt_symbol}", deprecated=True)
async def delete_bar_data(interval: str, vt_symbol: str) -> dict:
    """[已废弃] 删除单个合约的 K 线数据文件。

    请改用 `DELETE /api/alpha/data/resources/{kind}/{key}`。
    仅识别 daily 与 minute 两类目录：interval=="daily" 落在日线目录，其余一律按分钟目录处理。

    Args:
        interval: 周期目录选择，"daily" 取日线目录，其它值取分钟目录。
        vt_symbol: 合约代码，删除前会归一化。

    Returns:
        删除成功时返回 {"success": True, "message": ...}。

    Raises:
        HTTPException: Alpha 未安装时 503；目标 parquet 文件不存在时 404。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    vt_symbol = normalize_vt_symbol(vt_symbol)
    lab = _get_alpha_lab()
    folder = lab.daily_path if interval == "daily" else lab.minute_path
    file_path = folder.joinpath(f"{vt_symbol}.parquet")
    if not file_path.exists():
        raise HTTPException(404, f"数据文件 {vt_symbol} 不存在")
    file_path.unlink()
    return {"success": True, "message": f"{interval}/{vt_symbol} 数据已删除"}


@router.get("/bar-data/{interval}/{vt_symbol}", deprecated=True)
async def get_bar_data_detail(
    interval: str,
    vt_symbol: str,
    limit: int = 0,
    before: str | None = None,
) -> dict:
    """
    [已废弃] 获取单个合约K线数据详情。请改用
    `GET /api/alpha/data/resources/{kind}/{key}`。

    Args:
        interval: 周期（daily/minute）
        vt_symbol: 合约标识
        limit: 返回条数限制（默认0=返回全部；>0=返回最近N条）
        before: 游标时间（ISO格式），仅返回早于该时间的数据

    Returns:
        dict: 数据详情，含 preview、loaded_count、has_more、next_before

    Raises:
        HTTPException: Alpha 模块未安装时 503；interval 不在支持白名单
            {d/1m/5m/10m/15m/30m/60m/w} 内时 400；limit 小于 0 时 400；
            before 不是合法 ISO 时间字符串时 400；该合约对应周期的数据文件
            不存在（加载结果为空）时 404。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    vt_symbol = normalize_vt_symbol(vt_symbol)

    canonical_interval = _normalize_market_interval(interval)
    supported_intervals = {"d", "1m", "5m", "10m", "15m", "30m", "60m", "w"}
    if canonical_interval not in supported_intervals:
        raise HTTPException(
            400,
            f"interval 参数不支持: {interval}，"
            f"支持 daily/d/1m/5m/15m/30m/60m 等周期",
        )

    if limit < 0:
        raise HTTPException(400, "limit 参数不能小于0")

    before_dt: datetime | None = None
    if before:
        try:
            before_dt = datetime.fromisoformat(before.replace("Z", "+00:00"))
            if before_dt.tzinfo is not None:
                before_dt = before_dt.replace(tzinfo=None)
        except ValueError:
            raise HTTPException(400, "before 参数格式错误，需为 ISO 时间字符串")

    import polars as pl

    lab = _get_alpha_lab()
    df = lab.load_bar_frame_any_range(
        vt_symbol,
        canonical_interval,
        include_derived=True,
    )
    if df is None or df.is_empty():
        raise HTTPException(
            404,
            f"数据文件 {vt_symbol} ({canonical_interval}) 不存在",
        )

    filtered_df = df
    if before_dt is not None:
        try:
            filtered_df = filtered_df.filter(pl.col("datetime") < before_dt)
        except Exception:
            filtered_df = filtered_df.filter(pl.col("datetime").cast(pl.Utf8) < before)

    if limit > 0:
        has_more = len(filtered_df) > limit
        data_df = filtered_df.tail(limit)
    else:
        has_more = False
        data_df = filtered_df

    next_before: str | None = None
    if has_more and len(data_df) > 0:
        first_dt = data_df["datetime"][0]
        next_before = first_dt.isoformat() if hasattr(first_dt, "isoformat") else str(first_dt)

    preview = []
    for row in data_df.iter_rows(named=True):
        record = {}
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                record[k] = v.isoformat()
            elif v is None:
                record[k] = None
            else:
                record[k] = float(v) if isinstance(v, (int, float)) else str(v)
        preview.append(record)

    return {
        "vt_symbol": vt_symbol,
        "interval": canonical_interval,
        "row_count": len(df),
        "start": str(df["datetime"].min()),
        "end": str(df["datetime"].max()),
        "columns": list(df.columns),
        "preview": preview,
        "loaded_count": len(data_df),
        "has_more": has_more,
        "next_before": next_before,
    }
