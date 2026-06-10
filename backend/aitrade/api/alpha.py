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
from datetime import datetime
from typing import Any

import polars as pl
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
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
    """检查 Alpha 模块是否已安装。"""
    try:
        from ..alpha import AlphaLab
        return True
    except ImportError:
        return False


def _get_alpha_lab():
    """获取 AlphaLab 实例。"""
    from ..alpha import AlphaLab
    return AlphaLab(ALPHA_LAB_PATH)


# 业务逻辑已迁移至 alpha_service.py：
#   _build_feature_dataset / _apply_default_preprocessing / _pick_bar_provider


def _normalize_market_interval(interval: str | None) -> str:
    """Normalize request intervals used by market data endpoints."""
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
    return list(dict.fromkeys(normalize_vt_symbol(item) for item in vt_symbols if item))


def _normalize_optional_symbol(vt_symbol: str | None) -> str | None:
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
    """
    获取 Alpha 模块状态。

    返回模块安装状态和数据目录信息，用于前端判断功能可用性。

    Returns:
        dict: 状态信息
            - installed: 模块是否已安装
            - lab_path: 数据目录路径
            - lab_exists: 数据目录是否存在
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
async def list_tasks() -> list[dict]:
    """获取所有任务状态列表。"""
    tasks = task_manager.get_all_tasks()
    return [task.model_dump() for task in tasks]


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict:
    """获取单个任务状态。"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, f"任务 {task_id} 不存在")
    return task.model_dump()


# =============================================================================
# 数据下载
# =============================================================================

@router.post("/data/download")
async def start_data_download(req: DataDownloadRequest) -> dict:
    """启动原始市场数据下载任务。"""
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
        return alpha_service._download_bar_data(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)

    return {"task_id": task_id, "message": "下载任务已启动"}


@router.post("/data/aggregate")
async def start_data_aggregate(req: DataAggregateRequest) -> dict:
    """启动本地数据聚合任务。"""
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
        return alpha_service._aggregate_data(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "聚合任务已启动"}


# =============================================================================
# 标的画像（只读诊断）
# =============================================================================

@router.post("/profiling")
async def create_symbol_profile(payload: dict[str, Any]) -> Any:
    """生成标的画像。该端点只读，不进入任务队列。"""
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
    """列出已持久化的画像产物 id。"""
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    return ProfileStore().list_ids()


@router.get("/profiling/{artifact_id}")
async def get_symbol_profile_artifact(artifact_id: str) -> Any:
    """读取已持久化的画像产物。"""
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
    """列出所有数据集名称。Alpha 未安装时返回空列表。"""
    if not _check_alpha_installed():
        return []
    lab = _get_alpha_lab()
    return sorted(lab.list_all_datasets())


@router.get("/datasets/{name}")
async def get_dataset(name: str) -> dict:
    """获取数据集详情。"""
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
    """启动数据集创建任务。"""
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    req = req.model_copy(update={"vt_symbols": _normalize_symbol_list(req.vt_symbols)})
    if not req.features:
        raise HTTPException(400, "至少需要选择一个特征库")

    task_id = task_manager.create_task(TaskType.DATASET_CREATE, req.model_dump())

    def execute(on_progress=None):
        return alpha_service._create_dataset(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)

    return {"task_id": task_id, "message": "数据集创建任务已启动"}


@router.delete("/datasets/{name}")
async def remove_dataset(name: str) -> dict:
    """删除数据集。"""
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
    """列出所有模型名称。Alpha 未安装时返回空列表。"""
    if not _check_alpha_installed():
        return []
    lab = _get_alpha_lab()
    return sorted(lab.list_all_models())


@router.get("/models/{name}")
async def get_model(name: str) -> dict:
    """获取模型详情。"""
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
    """启动模型训练任务。"""
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    task_id = task_manager.create_task(TaskType.MODEL_TRAIN, req.model_dump())

    def execute(on_progress=None):
        return alpha_service._train_model(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)

    return {"task_id": task_id, "message": "模型训练任务已启动"}


@router.delete("/models/{name}")
async def remove_model(name: str) -> dict:
    """删除模型。"""
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
    """列出所有信号名称。Alpha 未安装时返回空列表。"""
    if not _check_alpha_installed():
        return []
    lab = _get_alpha_lab()
    return sorted(lab.list_all_signals())


@router.get("/signals/{name}")
async def get_signal(name: str) -> dict:
    """获取信号详情。"""
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
    """启动信号生成任务。"""
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    req = req.model_copy(update={"vt_symbols": _normalize_symbol_list(req.vt_symbols)})

    task_id = task_manager.create_task(TaskType.SIGNAL_GENERATE, req.model_dump())

    def execute(on_progress=None):
        return alpha_service._generate_signal(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)

    return {"task_id": task_id, "message": "信号生成任务已启动"}


@router.delete("/signals/{name}")
async def remove_signal(name: str) -> dict:
    """删除信号。"""
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
    """启动回测任务。"""
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    req = req.model_copy(update={"benchmark": _normalize_optional_symbol(req.benchmark)})

    task_id = task_manager.create_task(TaskType.BACKTEST_RUN, req.model_dump())

    def execute(on_progress=None):
        return alpha_service._run_backtest(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)

    return {"task_id": task_id, "message": "回测任务已启动"}


# =============================================================================
# 合约配置
# =============================================================================

@router.get("/contracts")
async def get_contract_settings() -> dict:
    """获取合约配置。Alpha 未安装时返回空字典。"""
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
) -> dict:
    """添加合约配置。"""
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    vt_symbol = normalize_vt_symbol(vt_symbol)
    lab = _get_alpha_lab()
    lab.add_contract_setting(vt_symbol, long_rate, short_rate, size, pricetick)
    return {"success": True, "message": f"合约 {vt_symbol} 配置已保存"}


# =============================================================================
# 统一数据资源
# =============================================================================

@router.get("/data/resources")
async def get_data_resources() -> dict[str, Any]:
    """获取原始K线、历史Tick与派生周期资源列表。"""
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
    """查看单个数据资源详情与预览。"""
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
    """删除单个原始或派生数据资源。"""
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    lab = _get_alpha_lab()
    if lab.delete_data_resource(kind, key):
        return {"success": True, "message": f"已删除 {kind}/{key}"}
    raise HTTPException(404, f"资源不存在: {kind}/{key}")


@router.patch("/data/resources/raw_bar/{key}/interval")
async def relocate_raw_bar_interval(key: str, req: RelocateBarIntervalRequest) -> dict[str, Any]:
    """更正原始 K 线资源的存储周期（移动文件到对应目录）。"""
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
    """预检上传批次是否可以手动合并为正式原始资源。"""
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    lab = _get_alpha_lab()
    try:
        return lab.preview_merge_import_batches(kind=req.kind, keys=req.keys)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post("/data/resources/merge")
async def merge_data_resource_batches(req: DataResourceMergeRequest) -> dict[str, Any]:
    """合并上传批次，写入正式原始资源。"""
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
    """Validate CSV upload and decode custom mapping."""
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
    """预览 bar CSV 文件（兼容旧前端）。"""
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
    """预览历史 Tick CSV 文件。"""
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
    """执行 bar CSV 数据导入。"""
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
    """执行历史 Tick CSV 数据导入。"""
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
    """[已废弃] 删除单个合约的K线数据文件。

    请改用 `DELETE /api/alpha/data/resources/{kind}/{key}`。
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
