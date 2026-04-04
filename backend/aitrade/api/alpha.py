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

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import ALPHA_LAB_PATH
from ..datasource import datasource_manager
from ..datasource.types import DataCategory, ProviderStatus
from ..models import (
    TaskType,
    DataDownloadRequest,
    DatasetCreateRequest,
    ModelTrainRequest,
    SignalGenerateRequest,
    BacktestRunRequest,
)
from ..task import task_manager
from .ws import ws_manager


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


def _pick_bar_provider() -> str:
    """为 Alpha 研究数据选择最佳数据源（拒绝 mock）。"""
    for name in ("tushare", "gateway"):
        provider = datasource_manager.get_provider(name)
        if provider and provider.get_info(0).status == ProviderStatus.AVAILABLE:
            if DataCategory.BAR_HISTORY in provider.get_supported_categories():
                return name
    return ""


# =============================================================================
# Services (inlined, no separate files)
# =============================================================================

def _download_bar_data(
    req: DataDownloadRequest,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> dict[str, Any]:
    """从数据源下载K线数据，保存到 AlphaLab。"""
    lab = _get_alpha_lab()

    interval_map = {"d": "d", "m": "1m", "w": "w"}
    interval = interval_map.get(req.interval, "d")

    provider_name = _pick_bar_provider()
    if not provider_name:
        raise RuntimeError(
            "没有可用的真实数据源。请配置 Tushare token。"
            "Alpha 研究需要真实历史数据，不支持 Mock 数据。"
        )

    total = len(req.vt_symbols)
    success_count = 0
    failed_symbols = []

    for i, vt_symbol in enumerate(req.vt_symbols):
        try:
            symbol, exchange_str = vt_symbol.rsplit(".", 1)

            records = datasource_manager.get_bar_history(
                symbol=symbol,
                exchange=exchange_str,
                interval=interval,
                start=datetime.combine(req.start, datetime.min.time()),
                end=datetime.combine(req.end, datetime.max.time()),
                provider_name=provider_name,
            )

            if records:
                from ..alpha.lab import BarData
                bars = []
                for r in records:
                    bar = BarData(
                        symbol=symbol,
                        exchange=exchange_str,
                        datetime=r.datetime,
                        interval=interval,
                        open_price=r.open_price,
                        high_price=r.high_price,
                        low_price=r.low_price,
                        close_price=r.close_price,
                        volume=r.volume,
                        turnover=r.turnover,
                        open_interest=r.open_interest,
                    )
                    bars.append(bar)

                lab.save_bar_data(bars)
                success_count += 1
            else:
                failed_symbols.append(f"{vt_symbol}: 数据源无数据")
        except Exception as e:
            failed_symbols.append(f"{vt_symbol}: {str(e)}")

        if on_progress:
            progress = (i + 1) / total * 100
            on_progress(progress, f"已下载 {vt_symbol} ({i + 1}/{total})")

    return {
        "total": total,
        "success": success_count,
        "failed": len(failed_symbols),
        "failed_symbols": failed_symbols,
        "provider": provider_name,
    }


def _create_dataset(
    req: DatasetCreateRequest,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> dict[str, Any]:
    """创建数据集（含特征计算）。"""
    from ..alpha.dataset.datasets.alpha_158 import Alpha158

    lab = _get_alpha_lab()

    if req.start >= req.train_end:
        raise ValueError(f"数据起始日期({req.start})必须早于训练截止日期({req.train_end})")
    if req.train_end >= req.end:
        raise ValueError(f"训练截止日期({req.train_end})必须早于数据结束日期({req.end})")

    if on_progress:
        on_progress(10, "加载K线数据...")

    valid_end = req.valid_end or (req.train_end + timedelta(days=90))
    if valid_end >= req.end:
        valid_end = req.train_end + (req.end - req.train_end) // 2

    df = lab.load_bar_df(
        vt_symbols=req.vt_symbols,
        interval="d",
        start=req.start,
        end=req.end,
        extended_days=100
    )

    if df is None or df.is_empty():
        raise ValueError("无法加载K线数据，请先下载数据")

    if on_progress:
        on_progress(30, "初始化数据集...")

    train_period = (str(req.start), str(req.train_end))
    valid_period = (str(req.train_end + timedelta(days=1)), str(valid_end))
    test_period = (str(valid_end + timedelta(days=1)), str(req.end))

    dataset = Alpha158(
        df=df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period
    )

    if on_progress:
        on_progress(40, "设置标签...")

    dataset.set_label(f"ts_delay(close, -{req.label_period}) / ts_delay(close, -1) - 1")

    if on_progress:
        on_progress(50, "计算特征（可能需要几分钟）...")

    dataset.prepare_data()

    if on_progress:
        on_progress(80, "数据预处理...")

    dataset.process_data()

    if on_progress:
        on_progress(90, "保存数据集...")

    lab.save_dataset(req.name, dataset)

    return {
        "name": req.name,
        "feature_count": len(dataset.feature_expressions),
        "sample_count": len(dataset.df) if dataset.df is not None else 0
    }


def _train_model(
    req: ModelTrainRequest,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> dict[str, Any]:
    """训练机器学习模型。"""
    from ..alpha.model.models.lgb_model import LgbModel
    from ..alpha.model.models.mlp_model import MlpModel
    from ..alpha.model.models.lasso_model import LassoModel

    lab = _get_alpha_lab()

    if on_progress:
        on_progress(10, "加载数据集...")

    dataset = lab.load_dataset(req.dataset)
    if dataset is None:
        raise ValueError(f"数据集 {req.dataset} 不存在")

    if on_progress:
        on_progress(20, "初始化模型...")

    model_classes = {
        "lgb": LgbModel,
        "mlp": MlpModel,
        "lasso": LassoModel
    }

    model_class = model_classes.get(req.model_type, LgbModel)
    model = model_class(**req.params)

    if on_progress:
        on_progress(30, "开始训练...")

    try:
        model.fit(dataset)
    except Exception as e:
        error_msg = str(e)
        if "non empty" in error_msg or "empty" in error_msg.lower():
            raise ValueError(
                "训练数据为空。可能原因：\n"
                "1. 数据集的时间范围设置不合理（start >= train_end）\n"
                "2. 下载的K线数据量不足\n"
                "3. 特征计算后有效样本被过滤掉\n"
                "请检查数据集创建时的日期参数。"
            ) from e
        raise

    if on_progress:
        on_progress(90, "保存模型...")

    lab.save_model(req.name, model)

    return {
        "name": req.name,
        "model_type": req.model_type
    }


def _generate_signal(
    req: SignalGenerateRequest,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> dict[str, Any]:
    """使用训练好的模型生成交易信号。"""
    import polars as pl
    from ..alpha.dataset.datasets.alpha_158 import Alpha158
    from ..alpha.dataset import Segment

    lab = _get_alpha_lab()

    if on_progress:
        on_progress(10, "加载模型...")

    model = lab.load_model(req.model)
    if model is None:
        raise ValueError(f"模型 {req.model} 不存在")

    if on_progress:
        on_progress(20, "加载K线数据...")

    df = lab.load_bar_df(
        vt_symbols=req.vt_symbols,
        interval="d",
        start=req.start,
        end=req.end,
        extended_days=100
    )

    if df is None or df.is_empty():
        raise ValueError("无法加载K线数据")

    if on_progress:
        on_progress(40, "准备数据集...")

    train_period = (str(req.start), str(req.start))
    valid_period = (str(req.start), str(req.start))
    test_period = (str(req.start), str(req.end))

    dataset = Alpha158(
        df=df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period
    )

    if on_progress:
        on_progress(60, "计算特征...")

    dataset.prepare_data()

    if on_progress:
        on_progress(80, "生成预测...")

    predictions = model.predict(dataset, Segment.TEST)

    base_df = df.filter(
        (pl.col("datetime") >= datetime.combine(req.start, datetime.min.time())) &
        (pl.col("datetime") <= datetime.combine(req.end, datetime.max.time()))
    ).select(["datetime", "vt_symbol", "close"])

    n_pred = len(predictions)
    if n_pred < len(base_df):
        base_df = base_df.tail(n_pred)
    elif n_pred > len(base_df):
        predictions = predictions[-len(base_df):]

    signal_df = base_df.with_columns(
        pl.Series(name="signal", values=predictions)
    )

    if on_progress:
        on_progress(90, "保存信号...")

    lab.save_signal(req.name, signal_df)

    return {
        "name": req.name,
        "row_count": len(signal_df)
    }


def _run_backtest(
    req: BacktestRunRequest,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> dict[str, Any]:
    """基于信号运行策略回测。"""
    from ..alpha.strategy import BacktestingEngine

    lab = _get_alpha_lab()

    if on_progress:
        on_progress(10, "加载信号...")

    signal_df = lab.load_signal(req.signal)
    if signal_df is None:
        raise ValueError(f"信号 {req.signal} 不存在")

    vt_symbols = signal_df["vt_symbol"].unique().to_list()

    if on_progress:
        on_progress(20, "初始化回测引擎...")

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval="d",
        start=datetime.combine(req.start, datetime.min.time()),
        end=datetime.combine(req.end, datetime.max.time()),
        capital=req.capital
    )

    for vt_symbol in vt_symbols:
        if vt_symbol not in engine.sizes:
            engine.sizes[vt_symbol] = 1
            engine.priceticks[vt_symbol] = 0.01
            engine.long_rates[vt_symbol] = 0.0003
            engine.short_rates[vt_symbol] = 0.0003

    if on_progress:
        on_progress(40, "加载历史数据...")

    engine.load_data()

    if on_progress:
        on_progress(60, "运行回测...")

    engine.run_backtesting()

    if on_progress:
        on_progress(80, "计算结果...")

    result = engine.calculate_result()

    if result is None:
        return {
            "name": req.name,
            "statistics": {
                "start_date": str(req.start),
                "end_date": str(req.end),
                "total_days": 0,
                "profit_days": 0,
                "loss_days": 0,
                "capital": req.capital,
                "end_balance": req.capital,
                "total_return": 0,
                "annual_return": 0,
                "max_drawdown": 0,
                "max_ddpercent": 0,
                "sharpe_ratio": 0,
                "total_trade_count": 0,
                "total_net_pnl": 0,
                "total_commission": 0,
                "error": "回测期间无成交记录，策略未产生交易信号",
            },
        }

    statistics = engine.calculate_statistics()

    return {
        "name": req.name,
        "statistics": statistics
    }


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
    """启动K线数据下载任务。"""
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD, req.model_dump())

    def execute(on_progress=None):
        return _download_bar_data(req, on_progress)

    task_manager.run_async(task_id, execute, on_progress=True)

    return {"task_id": task_id, "message": "下载任务已启动"}


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

    return {
        "name": name,
        "feature_count": len(dataset.feature_expressions),
        "sample_count": len(dataset.df) if hasattr(dataset, "df") and dataset.df is not None else 0,
        "label_expression": dataset.label_expression or "",
    }


@router.post("/datasets/create")
async def start_create_dataset(req: DatasetCreateRequest) -> dict:
    """启动数据集创建任务。"""
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    task_id = task_manager.create_task(TaskType.DATASET_CREATE, req.model_dump())

    def execute(on_progress=None):
        return _create_dataset(req, on_progress)

    task_manager.run_async(task_id, execute, on_progress=True)

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
        return _train_model(req, on_progress)

    task_manager.run_async(task_id, execute, on_progress=True)

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

    task_id = task_manager.create_task(TaskType.SIGNAL_GENERATE, req.model_dump())

    def execute(on_progress=None):
        return _generate_signal(req, on_progress)

    task_manager.run_async(task_id, execute, on_progress=True)

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

    task_id = task_manager.create_task(TaskType.BACKTEST_RUN, req.model_dump())

    def execute(on_progress=None):
        return _run_backtest(req, on_progress)

    task_manager.run_async(task_id, execute, on_progress=True)

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
    return lab.load_contract_setttings()


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
    lab = _get_alpha_lab()
    lab.add_contract_setting(vt_symbol, long_rate, short_rate, size, pricetick)
    return {"success": True, "message": f"合约 {vt_symbol} 配置已保存"}


# =============================================================================
# K线数据列表
# =============================================================================

@router.get("/bar-data")
async def get_bar_data_list() -> dict:
    """
    获取已有K线数据列表（含详情）。

    返回已下载的日线和分钟线数据文件列表，每个条目包含行数、时间范围、文件大小。
    """
    if not _check_alpha_installed():
        return {"daily": [], "minute": []}

    import polars as pl

    lab = _get_alpha_lab()
    result: dict[str, list] = {"daily": [], "minute": []}

    for key, folder in [("daily", lab.daily_path), ("minute", lab.minute_path)]:
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*.parquet")):
            try:
                df = pl.read_parquet(f, columns=["datetime"])
                info = {
                    "vt_symbol": f.stem,
                    "row_count": len(df),
                    "start": str(df["datetime"].min()),
                    "end": str(df["datetime"].max()),
                    "file_size_kb": round(f.stat().st_size / 1024, 1),
                }
            except Exception:
                info = {
                    "vt_symbol": f.stem,
                    "row_count": 0,
                    "start": "",
                    "end": "",
                    "file_size_kb": round(f.stat().st_size / 1024, 1),
                }
            result[key].append(info)

    return result


# =============================================================================
# CSV 导入 (必须在动态路由 /bar-data/{interval}/{vt_symbol} 之前)
# =============================================================================

@router.post("/bar-data/import/preview")
async def preview_csv_import(
    file: UploadFile = File(...),
    field_mapping: str | None = None,
) -> dict[str, Any]:
    """
    预览CSV文件，自动识别字段映射。

    上传CSV文件后，系统会自动匹配字段（支持中英文别名），
    并返回预览信息供用户确认。
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

    lab = _get_alpha_lab()
    try:
        result: dict[str, Any] = lab.preview_csv(contents, custom_mapping)
        return result
    except Exception as e:
        raise HTTPException(500, f"CSV 解析失败: {str(e)}")


@router.post("/bar-data/import")
async def import_csv_data(
    interval: str = Form(default="d", description="d=日线, m=分钟"),
    import_mode: str = Form(default="merge", description="merge=追加, replace=替换"),
    field_mapping: str | None = Form(default=None, description="自定义字段映射 JSON"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    执行 CSV 数据导入。

    支持追加(merge)和替换(replace)两种模式。
    自动匹配字段（支持中英文别名），缺失字段使用默认值。
    """
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")

    if interval not in {"d", "m"}:
        raise HTTPException(400, "interval 参数必须是 d 或 m")

    if import_mode not in {"merge", "replace"}:
        raise HTTPException(400, "import_mode 参数必须是 merge 或 replace")

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

    lab = _get_alpha_lab()
    try:
        result: dict[str, Any] = lab.import_csv(
            csv_content=contents,
            interval=interval,
            import_mode=import_mode,
            custom_mapping=custom_mapping,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"CSV 导入失败: {str(e)}")


# =============================================================================
# K线数据动态路由 (必须在 CSV 导入路由之后)
# =============================================================================

@router.delete("/bar-data/{interval}/{vt_symbol}")
async def delete_bar_data(interval: str, vt_symbol: str) -> dict:
    """删除单个合约的K线数据文件。"""
    if not _check_alpha_installed():
        raise HTTPException(503, "Alpha 模块未安装")
    lab = _get_alpha_lab()
    folder = lab.daily_path if interval == "daily" else lab.minute_path
    file_path = folder.joinpath(f"{vt_symbol}.parquet")
    if not file_path.exists():
        raise HTTPException(404, f"数据文件 {vt_symbol} 不存在")
    file_path.unlink()
    return {"success": True, "message": f"{interval}/{vt_symbol} 数据已删除"}


@router.get("/bar-data/{interval}/{vt_symbol}")
async def get_bar_data_detail(
    interval: str,
    vt_symbol: str,
    limit: int = 0,
    before: str | None = None,
) -> dict:
    """
    获取单个合约K线数据详情。

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

    if interval not in {"daily", "minute"}:
        raise HTTPException(400, "interval 参数必须是 daily 或 minute")

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
    folder = lab.daily_path if interval == "daily" else lab.minute_path
    file_path = folder.joinpath(f"{vt_symbol}.parquet")

    if not file_path.exists():
        raise HTTPException(404, f"数据文件 {vt_symbol} 不存在")

    df = pl.read_parquet(file_path)

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
        "interval": interval,
        "row_count": len(df),
        "start": str(df["datetime"].min()),
        "end": str(df["datetime"].max()),
        "columns": list(df.columns),
        "preview": preview,
        "loaded_count": len(data_df),
        "has_more": has_more,
        "next_before": next_before,
    }
