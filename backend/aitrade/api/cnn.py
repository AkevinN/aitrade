"""
CNN 卷积神经网络量化预测 API 路由。

提供 CNN 模型的训练、查询、删除功能，
复用现有 TaskManager 实现异步任务管理。
"""

from typing import Any, Callable, Optional

import polars as pl
from fastapi import APIRouter, HTTPException

from ..alpha.lab_utils import normalize_vt_symbol
from ..models import CNNTrainRequest, CNNBacktestRequest, CNNPredictRequest, TaskType
from ..models.governance import (
    CNNCandidateTrainRequest,
    CNNGovernanceConfig,
    CNNGovernanceReplayRequest,
    CNNPromotionRequest,
    CNNRollbackRequest,
    CNNWalkForwardRequest,
)
from ..models.screening import CNNScreeningRequest
from ..task import task_manager

router = APIRouter(prefix="/api/cnn", tags=["CNN量化预测"])


def _normalize_symbol_list(vt_symbols: list[str]) -> list[str]:
    """规范化证券代码列表并去重（保持首次出现顺序）。

    Args:
        vt_symbols: 原始证券代码列表，允许含空项。

    Returns:
        去重后的规范化代码列表。
    """
    return list(dict.fromkeys(normalize_vt_symbol(item) for item in vt_symbols if item))


# =============================================================================
# 模型治理（WF/OOS、候选、晋级、回滚、治理回放）
# =============================================================================

@router.get("/governance/config")
async def get_governance_config() -> dict:
    """读取 CNN 治理配置（晋级门禁参数、调度策略等）。

    GET /api/cnn/governance/config 的处理函数，从治理存储读取当前生效配置。

    Returns:
        治理配置字典，含晋级门禁阈值、调度策略等字段；
        从未保存过配置时返回存储侧的默认配置。
    """
    from ..cnn.governance import store
    return store.get_config()


@router.put("/governance/config")
async def update_governance_config(req: CNNGovernanceConfig) -> dict:
    """更新 CNN 治理配置并追加 config_updated 历史事件。

    PUT /api/cnn/governance/config 的处理函数，整体覆盖式保存治理配置，
    并向治理历史写入一条 config_updated 事件以便审计。

    Args:
        req: 新的治理配置（晋级门禁阈值、调度策略等），整体替换旧配置。

    Returns:
        保存后落盘的治理配置字典。
    """
    from ..cnn.governance import store
    return store.save_config(req)


@router.get("/governance/production")
async def get_governance_production() -> dict:
    """读取当前生产模型信息（model_name/version/promoted_at 等）。

    GET /api/cnn/governance/production 的处理函数，返回当前已晋级到生产的模型指针。

    Returns:
        生产模型信息字典，含 model_name/version/promoted_at 等字段；
        尚无生产模型时返回存储侧定义的空/默认结构。
    """
    from ..cnn.governance import store
    return store.get_production()


@router.get("/governance/candidates")
async def list_governance_candidates() -> list[dict]:
    """列出所有候选模型，按创建时间降序排列。

    GET /api/cnn/governance/candidates 的处理函数。

    Returns:
        候选模型元数据字典列表，最新创建者在前；无候选时返回空列表。
    """
    from ..cnn.governance import store
    return store.list_candidates()


@router.get("/governance/candidates/{candidate_id}")
async def get_governance_candidate(candidate_id: str) -> dict:
    """读取指定候选模型的元数据（含 WF 报告 ID 与训练结果摘要）。

    GET /api/cnn/governance/candidates/{candidate_id} 的处理函数。

    Args:
        candidate_id: 候选模型 ID，取自路径参数。

    Returns:
        候选模型元数据字典，含关联的 WF/OOS 报告 ID 与训练结果摘要。

    Raises:
        HTTPException: 候选不存在时返回 404。
    """
    from ..cnn.governance import store
    candidate = store.get_candidate(candidate_id)
    if candidate is None:
        raise HTTPException(404, f"候选不存在: {candidate_id}")
    return candidate


@router.post("/governance/evaluate")
async def start_governance_evaluate(req: CNNWalkForwardRequest) -> dict:
    """启动 WF/OOS（滚动前向/样本外）评估任务，异步执行并立即返回 task_id。

    POST /api/cnn/governance/evaluate 的处理函数：创建 CNN_WF_EVALUATE 任务并提交后台
    线程执行，不阻塞请求。前端可凭返回的 task_id 轮询进度与结果。

    Args:
        req: WF/OOS 评估请求，含评估名称、标的、滚动窗口与门禁参数等。

    Returns:
        含 task_id（任务 ID，用于轮询）与 name（评估名称）的字典。
    """
    task_id = task_manager.create_task(
        TaskType.CNN_WF_EVALUATE,
        params=req.model_dump(mode="json"),
        title="CNN WF/OOS 评估",
        entity_type="cnn_governance_report",
        entity_name=req.name,
    )

    def _run(on_progress: Optional[Callable[[float, str], None]] = None) -> dict[str, Any]:
        """WF/OOS 评估任务的后台执行体：在任务线程内调用滚动前向评估并返回治理报告。

        Args:
            on_progress: 进度回调 ``(percent, message)``，转交给底层评估逐步上报；可为 None。

        Returns:
            run_walk_forward_evaluate 产出的治理报告字典。
        """
        from ..cnn.governance import run_walk_forward_evaluate
        return run_walk_forward_evaluate(req, on_progress=on_progress)

    task_manager.run_async(task_id, _run, enable_progress=True)
    return {"task_id": task_id, "name": req.name}


@router.post("/governance/candidates/train")
async def start_governance_candidate_train(req: CNNCandidateTrainRequest) -> dict:
    """启动候选模型训练任务（WF/OOS 评估 + 最终模型训练），异步执行并立即返回 task_id。

    POST /api/cnn/governance/candidates/train 的处理函数：创建 CNN_CANDIDATE_TRAIN 任务并
    提交后台线程执行。任务内先做滚动前向评估再训练最终模型，产出一个待晋级候选。

    Args:
        req: 候选训练请求，含候选名称、标的、训练超参与 WF/OOS 评估配置等。

    Returns:
        含 task_id（任务 ID，用于轮询）与 name（候选名称）的字典。
    """
    task_id = task_manager.create_task(
        TaskType.CNN_CANDIDATE_TRAIN,
        params=req.model_dump(mode="json"),
        title="CNN 候选模型训练",
        entity_type="cnn_candidate",
        entity_name=req.name,
    )

    def _run(on_progress: Optional[Callable[[float, str], None]] = None) -> dict[str, Any]:
        """候选训练任务的后台执行体：在任务线程内做 WF/OOS 评估并训练最终模型，返回候选信息。

        Args:
            on_progress: 进度回调 ``(percent, message)``，转交给底层训练逐步上报；可为 None。

        Returns:
            train_candidate 产出的候选模型信息字典。
        """
        from ..cnn.governance import train_candidate
        return train_candidate(req, on_progress=on_progress)

    task_manager.run_async(task_id, _run, enable_progress=True)
    return {"task_id": task_id, "name": req.name}


@router.post("/governance/candidates/{candidate_id}/promote")
async def promote_governance_candidate(candidate_id: str, req: CNNPromotionRequest) -> dict:
    """将指定候选模型晋级为生产模型（同步执行）。

    POST /api/cnn/governance/candidates/{candidate_id}/promote 的处理函数：把候选指针写为
    新的生产模型并记录晋级事件。同步完成，返回时晋级已生效。

    Args:
        candidate_id: 待晋级候选模型 ID，取自路径参数。
        req: 晋级请求，含晋级备注（note）等。

    Returns:
        晋级后的生产模型信息字典。

    Raises:
        HTTPException: 候选不存在（底层抛 FileNotFoundError）时返回 404。
    """
    from ..cnn.governance import promote_candidate
    try:
        return promote_candidate(candidate_id, req)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/governance/candidates/{candidate_id}/reject")
async def reject_governance_candidate(candidate_id: str, req: CNNPromotionRequest) -> dict:
    """拒绝指定候选模型，更新其状态为 rejected（同步执行）。

    POST /api/cnn/governance/candidates/{candidate_id}/reject 的处理函数：将候选标记为
    rejected 并记录拒绝备注，不影响当前生产模型。

    Args:
        candidate_id: 待拒绝候选模型 ID，取自路径参数。
        req: 晋级/拒绝请求，此处仅取其中的 note 作为拒绝备注。

    Returns:
        更新后的候选模型元数据字典（状态已置为 rejected）。

    Raises:
        HTTPException: 候选不存在（底层抛 FileNotFoundError）时返回 404。
    """
    from ..cnn.governance import reject_candidate
    try:
        return reject_candidate(candidate_id, note=req.note)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/governance/rollback")
async def rollback_governance_production(req: CNNRollbackRequest) -> dict:
    """将生产模型回滚到上一版本（或请求指定的模型），同步执行。

    POST /api/cnn/governance/rollback 的处理函数：把生产指针指回历史版本并记录回滚事件。

    Args:
        req: 回滚请求，含目标版本/模型与回滚备注；未指定目标时回滚到上一版本。

    Returns:
        回滚后的生产模型信息字典。

    Raises:
        HTTPException: 目标模型不存在或无可回滚版本（底层抛 FileNotFoundError/ValueError）
            时返回 400。
    """
    from ..cnn.governance import rollback_production
    try:
        return rollback_production(req)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/governance/history")
async def get_governance_history() -> list[dict]:
    """读取全量治理历史事件列表（JSONL，按写入时间升序）。

    GET /api/cnn/governance/history 的处理函数，返回配置更新、晋级、拒绝、回滚等审计事件。

    Returns:
        治理事件字典列表，按写入时间升序（最早在前）；无事件时返回空列表。
    """
    from ..cnn.governance import store
    return store.history()


@router.get("/governance/reports/{report_id}")
async def get_governance_report(report_id: str) -> dict:
    """读取指定 WF/OOS 评估报告（含各折统计与晋级门禁结果）。

    GET /api/cnn/governance/reports/{report_id} 的处理函数。

    Args:
        report_id: WF/OOS 评估报告 ID，取自路径参数。

    Returns:
        评估报告字典，含各折（fold）统计与晋级门禁判定结果。

    Raises:
        HTTPException: 报告不存在时返回 404。
    """
    from ..cnn.governance import store
    report = store.get_report(report_id)
    if report is None:
        raise HTTPException(404, f"报告不存在: {report_id}")
    return report


@router.post("/governance/replay/run")
async def start_governance_replay(req: CNNGovernanceReplayRequest) -> dict:
    """启动治理回放回测任务，在历史区间对比三条基线策略，异步执行并立即返回 task_id。

    POST /api/cnn/governance/replay/run 的处理函数：创建 CNN_GOVERNANCE_REPLAY 任务并提交
    后台线程执行，在选定历史区间回放治理决策并与三条基线策略对比。

    Args:
        req: 治理回放请求，含回放名称、历史区间与基线对比配置等。

    Returns:
        含 task_id（任务 ID，用于轮询）与 name（回放名称）的字典。
    """
    task_id = task_manager.create_task(
        TaskType.CNN_GOVERNANCE_REPLAY,
        params=req.model_dump(mode="json"),
        title="CNN 治理回放回测",
        entity_type="cnn_governance_replay",
        entity_name=req.name,
    )

    def _run(on_progress: Optional[Callable[[float, str], None]] = None) -> dict[str, Any]:
        """治理回放任务的后台执行体：在任务线程内对历史区间做基线对比回放并返回回放报告。

        Args:
            on_progress: 进度回调 ``(percent, message)``，转交给底层回放逐步上报；可为 None。

        Returns:
            run_governance_replay 产出的治理回放报告字典。
        """
        from ..cnn.governance import run_governance_replay
        return run_governance_replay(req, on_progress=on_progress)

    task_manager.run_async(task_id, _run, enable_progress=True)
    return {"task_id": task_id, "name": req.name}


@router.get("/governance/replay")
async def list_governance_replays() -> list[dict]:
    """列出所有治理回放报告，按创建时间降序排列。

    GET /api/cnn/governance/replay 的处理函数。

    Returns:
        治理回放报告字典列表，最新创建者在前；无报告时返回空列表。
    """
    from ..cnn.governance import store
    return store.list_replay_reports()


@router.get("/governance/replay/{replay_id}")
async def get_governance_replay(replay_id: str) -> dict:
    """读取指定治理回放报告（含三条基线对比、晋级事件与结论）。

    GET /api/cnn/governance/replay/{replay_id} 的处理函数。

    Args:
        replay_id: 治理回放报告 ID，取自路径参数。

    Returns:
        治理回放报告字典，含三条基线策略对比、回放期内晋级事件与结论。

    Raises:
        HTTPException: 报告不存在时返回 404。
    """
    from ..cnn.governance import store
    replay = store.get_replay_report(replay_id)
    if replay is None:
        raise HTTPException(404, f"治理回放报告不存在: {replay_id}")
    return replay


# =============================================================================
# 选股（CNN Screening）
# =============================================================================

@router.post("/screening/batch")
async def start_cnn_screening(req: CNNScreeningRequest) -> dict:
    """启动 CNN 选股批量任务，异步执行并立即返回 task_id。

    POST /api/cnn/screening/batch 的处理函数：创建 CNN_SCREENING 任务并提交后台
    线程执行，不阻塞请求。前端可凭返回的 task_id 轮询进度与结果。

    选股两阶段漏斗（由 ScreeningRunner 编排）：
    - Tier-1（只读）：对候选池逐只复用 Profiler 画像 + CNN 代理指标，合成 CNN_Fitness_Score；
    - Tier-2（可选）：对 top_k 入围标的运行 WF/OOS 实证，派生绝对 edge 结论。
    结果经 GET /api/alpha/tasks/{task_id} 轮询获取。

    Args:
        req: CNN 选股请求，含 universe 过滤参数（as_of/exchange/include|exclude_symbols/
            min_bar_count）、漏斗配置（top_k/run_tier2/min_confidence）、Tier-2 超参
            （objective/eval_start）与持久化开关（persist）。
            注意：as_of 为必填，无任何隐式"全量"默认（Requirement 9.1）。

    Returns:
        含 task_id（任务 ID，用于轮询）与 name（选股任务名称）的字典。
    """
    task_id = task_manager.create_task(
        TaskType.CNN_SCREENING,
        params=req.model_dump(mode="json"),
        title="CNN 选股批量评估",
        entity_type="cnn_screening",
        entity_name=req.name,
    )

    def _run(on_progress: Optional[Callable[[float, str], None]] = None) -> dict[str, Any]:
        """选股任务的后台执行体：在任务线程内调用两阶段漏斗并返回 JSON 可序列化的选股结果。

        Args:
            on_progress: 进度回调 ``(percent, message)``，透传给 ScreeningRunner；可为 None。

        Returns:
            run_cnn_screening_batch 产出的 ScreeningResult.model_dump(mode="json") 字典。
        """
        from ..screening.runner import run_cnn_screening_batch  # lazy import（避免循环）
        return run_cnn_screening_batch(req, on_progress=on_progress)

    task_manager.run_async(task_id, _run, enable_progress=True)
    return {"task_id": task_id, "name": req.name}


# =============================================================================
# Torch check helpers
# =============================================================================

def _check_torch() -> bool:
    """检测当前环境是否已安装 PyTorch。

    通过尝试 import torch 判断，供各路由在执行依赖 torch 的训练/推理/回测前做前置校验。

    Returns:
        torch 可正常导入时返回 True，导入失败（未安装）时返回 False。
    """
    try:
        import torch
        return True
    except ImportError:
        return False


def _get_device() -> str:
    """返回 PyTorch 推荐使用的计算设备标识。

    Returns:
        CUDA 可用时返回 "cuda"，仅 CPU 时返回 "cpu"；
        torch 不可用或探测异常时返回 "N/A"。
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "N/A"


# =============================================================================
# 状态检查
# =============================================================================

@router.get("/status")
async def get_cnn_status() -> dict:
    """检查 CNN 功能是否可用（即 PyTorch 是否安装及可用设备）。

    GET /api/cnn/status 的处理函数，供前端在进入 CNN 相关页面前判断功能是否就绪。

    Returns:
        含 torch_installed（bool，是否已安装 PyTorch）与 device（str，可用计算设备；
        torch 未安装时为 "N/A"）的字典。
    """
    torch_ok = _check_torch()
    return {
        "torch_installed": torch_ok,
        "device": _get_device() if torch_ok else "N/A",
    }


# =============================================================================
# 训练管理
# =============================================================================

@router.post("/train")
async def start_cnn_train(req: CNNTrainRequest) -> dict:
    """启动 CNN 训练任务，异步执行并立即返回 task_id。

    POST /api/cnn/train 的处理函数。先做一系列请求级参数校验（日期顺序、train_ratio
    取值、输入数据/周期组合、label 配置、objective↔label 约束），再将观测组与显式标的
    归一化去重后合成最终训练标的集，最后创建 CNN_TRAIN 任务交给 TaskManager 在后台线程
    执行。前端可通过 /api/alpha/tasks/{task_id} 轮询进度。

    Args:
        req: CNN 训练请求。关键字段：start/end 训练区间（start 必须早于 end）；
            train_ratio 训练集占比，须在 (0, 1) 开区间；input_data_kind 输入类型，
            "bar" 或 "tick"；input_interval K 线周期，取 d/1m/5m/10m/15m/30m/60m，
            tick 输入不支持 "d"；vt_symbols 与 observation_groups 提供观测标的，
            target_symbol 为预测目标（缺省则取观测集，但二者不能全空）；label_spec
            标签定义（horizon_bars 模式须给 horizon，oco 模式须给正的 take_profit 与
            stop_loss）；objective 训练目标，path_class 时 label_spec.mode 必须为 oco。

    Returns:
        含 task_id（任务 ID，用于轮询）与 name（模型名）的字典。

    Raises:
        HTTPException: PyTorch 未安装、上述任一参数校验不通过时返回 400。
    """
    if not _check_torch():
        raise HTTPException(400, "PyTorch 未安装，请先执行: pip install torch")

    if req.start >= req.end:
        raise HTTPException(400, "开始日期必须早于结束日期")
    if not 0 < req.train_ratio < 1:
        raise HTTPException(400, "train_ratio 必须在 0 和 1 之间")
    if req.input_data_kind not in {"bar", "tick"}:
        raise HTTPException(400, "input_data_kind 必须是 bar 或 tick")

    symbols_from_groups = [
        normalize_vt_symbol(symbol)
        for group in req.observation_groups
        for symbol in group.symbols
    ]
    target_symbol = normalize_vt_symbol(req.target_symbol) if req.target_symbol else None
    effective_symbols = _normalize_symbol_list([*(req.vt_symbols or []), *symbols_from_groups])
    if target_symbol and target_symbol not in effective_symbols:
        effective_symbols = [target_symbol, *effective_symbols]
    if not effective_symbols:
        raise HTTPException(400, "至少需要提供 target_symbol 或观测证券列表")

    input_interval = str(req.input_interval).lower()
    if input_interval not in {"d", "1m", "5m", "10m", "15m", "30m", "60m"}:
        raise HTTPException(400, "input_interval 仅支持 d/1m/5m/10m/15m/30m/60m")
    if req.input_data_kind == "tick" and input_interval == "d":
        raise HTTPException(400, "Tick 输入目前仅支持分钟周期，请选择 1m/5m/10m/15m/30m/60m")

    label_spec = req.label_spec.model_dump() if hasattr(req.label_spec, "model_dump") else dict(req.label_spec)
    if label_spec.get("mode") == "horizon_bars" and not label_spec.get("horizon"):
        raise HTTPException(400, "horizon_bars 模式必须提供 horizon")
    if label_spec.get("mode") == "oco" and (
        not label_spec.get("take_profit") or not label_spec.get("stop_loss")
    ):
        raise HTTPException(400, "oco 模式必须提供正的 take_profit 与 stop_loss（如 0.03 表示 3%）")

    # Property 6 API 侧：path_class objective 必须配合 oco label，否则路径标签无法构建四类标注
    if req.objective == "path_class" and label_spec.get("mode") != "oco":  # 与上方 "oco" 字符串风格一致（LabelMode.OCO.value == "oco"，等价）
        raise HTTPException(
            400,
            "objective=path_class 需要 label_spec.mode=oco（路径标签依赖三重障碍判定）",
        )

    task_id = task_manager.create_task(
        TaskType.CNN_TRAIN,
        params={"name": req.name, "symbols": effective_symbols},
        title="训练 CNN 模型",
        entity_type="cnn_model",
        entity_name=req.name,
    )

    def _run(on_progress: Optional[Callable[[float, str], None]] = None) -> dict[str, Any]:
        """CNN 训练任务的后台执行体：在任务线程内用请求参数训练 CNN 模型并返回训练结果。

        闭包捕获外层已校验的 effective_symbols、target_symbol、input_interval、label_spec
        等，组装观测分组后调用 train_cnn_model。

        Args:
            on_progress: 进度回调 ``(percent, message)``，转交给训练过程逐步上报；可为 None。

        Returns:
            train_cnn_model 产出的训练结果字典。
        """
        from ..cnn import train_cnn_model
        return train_cnn_model(
            name=req.name,
            vt_symbols=effective_symbols,
            start=req.start,
            end=req.end,
            target_symbol=target_symbol,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            lookback=req.lookback,
            dropout=req.dropout,
            train_ratio=req.train_ratio,
            observation_groups=[
                {
                    **group.model_dump(),
                    "symbols": _normalize_symbol_list(group.symbols),
                }
                for group in req.observation_groups
            ],
            input_data_kind=req.input_data_kind,
            input_interval=input_interval,
            label_spec=label_spec,
            loss_weighting=req.loss_weighting,
            objective=req.objective,
            on_progress=on_progress,
        )

    task_manager.run_async(task_id, _run, enable_progress=True)

    return {"task_id": task_id, "name": req.name}


# =============================================================================
# 模型管理
# =============================================================================

@router.get("/models")
async def list_models() -> list[dict]:
    """列出已保存的 CNN 模型。

    GET /api/cnn/models 的处理函数。

    Returns:
        每个 CNN 模型的摘要信息字典列表；无模型时返回空列表。
    """
    from ..cnn import list_cnn_models
    return list_cnn_models()


@router.get("/models/{name}")
async def get_model_detail(name: str) -> dict:
    """获取指定 CNN 模型的详情（含训练配置与训练历史）。

    GET /api/cnn/models/{name} 的处理函数。

    Args:
        name: 模型名，取自路径参数。

    Returns:
        模型详情字典，含训练配置、指标与逐 epoch 训练历史等。

    Raises:
        HTTPException: 模型不存在（底层抛 FileNotFoundError）时返回 404。
    """
    from ..cnn import get_cnn_model_detail
    try:
        return get_cnn_model_detail(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.get("/models/{name}/architecture")
async def get_model_architecture(name: str) -> dict:
    """探查指定 CNN 模型的真实网络结构（重建实例 + 加载权重 + 逐层形状）。

    GET /api/cnn/models/{name}/architecture 的处理函数。需要 PyTorch 以重建模型实例。

    Args:
        name: 模型名，取自路径参数。

    Returns:
        网络结构描述字典，含各层类型与逐层输入/输出形状等。

    Raises:
        HTTPException: PyTorch 未安装时返回 400；模型不存在（底层抛 FileNotFoundError）
            时返回 404。
    """
    if not _check_torch():
        raise HTTPException(400, "PyTorch 未安装，无法探查模型结构")
    from ..cnn import describe_cnn_architecture
    try:
        return describe_cnn_architecture(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@router.delete("/models/{name}")
async def delete_model(name: str) -> dict:
    """删除指定 CNN 模型。

    DELETE /api/cnn/models/{name} 的处理函数。

    Args:
        name: 待删除模型名，取自路径参数。

    Returns:
        含 deleted（被删除的模型名）的字典。

    Raises:
        HTTPException: 模型不存在（删除返回 False）时返回 404。
    """
    from ..cnn import delete_cnn_model
    ok = delete_cnn_model(name)
    if not ok:
        raise HTTPException(404, f"模型不存在: {name}")
    return {"deleted": name}


# =============================================================================
# 回测
# =============================================================================

def _run_cnn_backtest(
    req: CNNBacktestRequest,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """执行 CNN 模型回测：推理生成信号 → 共享回测引擎 → 返回统计+成交+净值。

    核心步骤：
    1. 校验模型文件存在；
    2. 调用 predict_cnn_signals 生成 [datetime, vt_symbol, signal] 信号表；
    3. 解析出场配置（auto=按 label 自动推导），校验 label↔策略一致性；
    4. 初始化 BacktestingEngine，设置成本参数与 fill_price_mode；
    5. 运行回测并计算统计，附买入持有基准收益。

    隐性失败守护：信号与行情 datetime 无交集时直接抛 ValueError 而非返回空结果。

    Args:
        req: CNN 回测请求，含 model/start/end/capital/exit_mode/buy_threshold 等参数。
        on_progress: 进度回调 ``(percent, message)``，可为 None。

    Returns:
        字典，含 name/model/target_symbol/statistics/trades/equity_curve；
        无信号或无成交时 statistics 含 error 键，trades/equity_curve 为空列表。

    Raises:
        ValueError: CNN 模型不存在、信号与行情无交集、label↔策略硬性不一致时抛出。
    """
    import logging
    from datetime import datetime

    from ..alpha import AlphaLab
    from ..config import ALPHA_LAB_PATH
    from ..cnn.predictor import predict_cnn_signals
    from ..cnn.strategy import CNNSignalStrategy
    from ..cnn.storage import CNN_MODEL_DIR
    from ..backtest.engine import BacktestingEngine
    from ..backtest.artifacts import (
        serialize_trades,
        serialize_equity_curve,
        extract_benchmark_prices,
        attach_benchmark_returns,
        summarize_benchmark,
    )

    logger = logging.getLogger(__name__)

    # 1. Validate model exists
    model_path = CNN_MODEL_DIR / f"{req.model}.pt"
    if not model_path.exists():
        raise ValueError(f"CNN 模型不存在: {req.model}")

    if on_progress:
        on_progress(5, f"开始 CNN 回测: {req.name}")

    # 2. Generate signals via CNN inference
    if on_progress:
        on_progress(10, "正在进行 CNN 推理...")

    def inference_progress(pct: float, msg: str) -> None:
        """推理进度适配器：把推理段的 0~100% 线性映射到整体回测进度的 10~50% 后上报。

        Args:
            pct: 推理自身进度百分比（0~100）。
            msg: 推理阶段的进度文案，会加上 "[推理] " 前缀。
        """
        if on_progress:
            # Map inference progress (0-100) to overall (10-50)
            on_progress(10 + pct * 0.4, f"[推理] {msg}")

    signal_df = predict_cnn_signals(
        model_name=req.model,
        start=req.start,
        end=req.end,
        on_progress=inference_progress,
    )

    if signal_df.is_empty():
        return {
            "name": req.name,
            "statistics": {"error": "CNN 推理未产生任何信号"},
            # 字段恒在：无信号即无成交、无净值曲线
            "trades": [],
            "equity_curve": [],
        }

    # 3. Extract target symbol and setup engine
    import torch
    from ..cnn.consistency import (
        check_label_strategy_consistency,
        derive_strategy_exit_from_label,
    )

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    target_symbol = normalize_vt_symbol(checkpoint["train_config"]["target_symbol"])
    # 回测周期必须与训练/推理使用的 K 线周期一致，否则信号与行情对不齐
    input_interval = checkpoint["train_config"].get("input_interval", "d")
    # 训练 label 定义（用于 label↔策略出场 一致性自检，迭代 2 红线）
    label_spec = checkpoint["train_config"].get("label_spec", {}) or {}
    # 预测目标：分类 signal 为概率(0~1)，回归 signal 为预测收益(无界)；阈值口径随之不同
    objective = checkpoint["train_config"].get("objective", "classification")
    vt_symbols = [target_symbol]
    signal_df = signal_df.with_columns(
        pl.col("vt_symbol").map_elements(normalize_vt_symbol, return_dtype=pl.Utf8)
    )

    # 解析出场配置：auto=按 label 自动推导对齐的出场（固定持有 或 OCO 止盈止损）
    if req.exit_mode == "auto":
        exit_cfg = derive_strategy_exit_from_label(label_spec, input_interval)
        exit_mode = exit_cfg["exit_mode"]
        hold_days = exit_cfg["hold_days"]
        # OCO label 自动对齐时，止盈/止损以 label 为准，确保 label 出场 ≡ 策略出场
        take_profit = exit_cfg.get("take_profit", req.take_profit)
        stop_loss = exit_cfg.get("stop_loss", req.stop_loss)
    else:
        exit_mode = req.exit_mode
        hold_days = req.hold_days
        take_profit = req.take_profit
        stop_loss = req.stop_loss

    # label↔策略出场 一致性自检：硬性不一致直接抛错（回测失败），软性问题收集为告警
    consistency_warnings = check_label_strategy_consistency(
        label_spec, exit_mode, hold_days, input_interval
    )

    if on_progress:
        on_progress(55, f"初始化回测引擎, 目标: {target_symbol}, 周期: {input_interval}")

    lab = AlphaLab(ALPHA_LAB_PATH)
    engine = BacktestingEngine(data_loader=lab)

    start_dt = datetime.combine(req.start, datetime.min.time())
    end_dt = datetime.combine(req.end, datetime.max.time())

    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=input_interval,
        start=start_dt,
        end=end_dt,
        capital=int(req.capital),
    )

    # Set defaults for symbols missing contract config，并按请求统一成本参数
    for vt_symbol in vt_symbols:
        if vt_symbol not in engine.sizes:
            engine.sizes[vt_symbol] = 1
            engine.priceticks[vt_symbol] = 0.01
        # 成本以回测请求为准：佣金、卖出印花税、每笔不利滑点
        engine.long_rates[vt_symbol] = req.commission_rate
        engine.short_rates[vt_symbol] = req.commission_rate
        engine.stamp_duties[vt_symbol] = req.stamp_duty
        engine.slippages[vt_symbol] = req.slippage

    # T+1 卖出限制（当日买入不可当日卖出）
    engine.t_plus1 = req.t_plus1

    # 撮合成交价口径与训练 label 的 price_ref 一一对齐，杜绝回测↔label 背离：
    #   next_open→open、next_close→close、next_vwap→vwap；close(研究口径)仍按开盘近似。
    _PRICE_REF_TO_FILL_MODE = {
        "next_open": "open",
        "next_close": "close",
        "next_vwap": "vwap",
        "close": "open",
    }
    engine.fill_price_mode = _PRICE_REF_TO_FILL_MODE.get(
        str(label_spec.get("price_ref") or "close"), "open"
    )

    # 4. Add strategy（出场配置经一致性自检/自动对齐后注入）
    strategy_setting = {
        "buy_threshold": req.buy_threshold,
        "sell_threshold": req.sell_threshold,
        "price_add": req.price_add,
        "exit_mode": exit_mode,
        "hold_days": hold_days,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        # path_class 专用：否决阈值透传，非 path_class 场景下保持默认 1.0（等效关闭）
        "veto_threshold": req.veto_threshold,
    }
    engine.add_strategy(CNNSignalStrategy, strategy_setting, signal_df)

    # 5. Load data
    if on_progress:
        on_progress(60, "加载历史数据...")
    engine.load_data()

    # 隐性失败守护：信号与行情 datetime 必须有交集，否则全程无成交（疑似周期/对齐错误）
    bar_dts = {dt.replace(tzinfo=None) for dt in engine.dts}
    sig_dts = {
        (d.replace(tzinfo=None) if isinstance(d, datetime) else d)
        for d in signal_df["datetime"].to_list()
    }
    if bar_dts.isdisjoint(sig_dts):
        raise ValueError(
            f"信号与行情 datetime 无交集（信号 {len(sig_dts)} 个时点、行情 {len(bar_dts)} 个时点），"
            f"疑似回测周期({input_interval})与信号周期不一致或数据缺失，回测判定失败。"
        )

    # 6. Run backtest
    if on_progress:
        on_progress(70, "运行回测...")
    engine.run_backtesting()

    # 7. Calculate results
    if on_progress:
        on_progress(85, "计算结果...")
    daily_df = engine.calculate_result()

    if daily_df is None or engine.trade_count == 0:
        logger.warning("CNN 回测期间无成交记录")
        # veto_count 在零成交时尤其有意义：激进 veto 导致全程否决恰是最需要解释的场景
        return {
            "name": req.name,
            "statistics": {
                "error": "回测期间无成交，请调整阈值或检查数据",
                "capital": req.capital,
                "total_trade_count": 0,
                "veto_count": int(getattr(engine.strategy, "_veto_count", 0)),
            },
            # 字段恒在：无成交即空成交列表与空净值曲线
            "trades": [],
            "equity_curve": [],
        }

    statistics = engine.calculate_statistics()
    # path_class 否决计数：「本要买入却被否决的次数」，供前端显示策略激进程度
    # 防御式读取：非 path_class 场景下策略无 _veto_count 属性时回退 0
    statistics["veto_count"] = int(getattr(engine.strategy, "_veto_count", 0))
    # 回传本次回测使用的成本参数，便于前端展示「成本假设」
    statistics["commission_rate"] = req.commission_rate
    statistics["stamp_duty"] = req.stamp_duty
    statistics["slippage"] = req.slippage
    statistics["price_add"] = req.price_add
    statistics["exit_mode"] = exit_mode
    statistics["hold_days"] = hold_days
    statistics["t_plus1"] = req.t_plus1
    statistics["objective"] = objective
    # 回测撮合成交价口径（与训练 label price_ref 对齐）
    statistics["fill_price_mode"] = engine.fill_price_mode
    # label↔策略一致性：回传训练 label 口径与软性告警，便于前端提示「回测可信度」
    statistics["label_spec"] = label_spec
    statistics["consistency_warnings"] = consistency_warnings

    if on_progress:
        on_progress(100, "CNN 回测完成")

    # 基准（买入持有标的）：CNN 回测的基准即目标标的本身
    equity_curve = serialize_equity_curve(engine.daily_df)
    benchmark_prices = extract_benchmark_prices(engine.daily_results, target_symbol)
    attach_benchmark_returns(equity_curve, benchmark_prices, req.capital)
    statistics.update(summarize_benchmark(equity_curve, target_symbol))

    return {
        "name": req.name,
        "model": req.model,
        "target_symbol": target_symbol,
        "statistics": statistics,
        # 成交明细与逐日净值序列：equity_curve 必须在 calculate_statistics() 之后取
        # （此时 engine.daily_df 才补入 balance/drawdown 列）
        "trades": serialize_trades(engine.trades),
        "equity_curve": equity_curve,
    }


@router.post("/backtest/run")
async def start_cnn_backtest(req: CNNBacktestRequest) -> dict:
    """启动 CNN 模型回测任务，异步执行并立即返回 task_id。

    POST /api/cnn/backtest/run 的处理函数。先做请求级校验（日期顺序、buy/sell 阈值取值与
    大小关系），再在模型已存在时读取 checkpoint 的 objective，按其口径（分类=概率尺度、
    回归=收益尺度）校验阈值是否被误用——回归模型误用概率阈值或概率阈值越界会当场拒绝，
    避免任务静默空跑；该校验与回测策略、实盘 service 共用同一 threshold_scale_check。
    校验通过后创建 CNN_BACKTEST 任务，任务内经 CNN 推理生成信号并用共享回测引擎执行回测。

    Args:
        req: CNN 回测请求。关键字段：start/end 回测区间（start 须早于 end）；model 模型名；
            buy_threshold/sell_threshold 买入/卖出阈值，均须落在 (-1, 1) 开区间且
            buy_threshold > sell_threshold；其余成本/出场参数透传给回测任务。

    Returns:
        含 task_id（任务 ID，用于轮询）与 message（启动提示）的字典。

    Raises:
        HTTPException: PyTorch 未安装、日期顺序错误、阈值越界或大小关系不成立、
            阈值尺度与模型 objective 不匹配时返回 400。
    """
    if not _check_torch():
        raise HTTPException(400, "PyTorch 未安装，请先执行: pip install torch")

    if req.start >= req.end:
        raise HTTPException(400, "开始日期必须早于结束日期")
    # 阈值范围放宽到 (-1, 1)：分类用概率尺度(如 0.6/0.4)，回归用收益尺度(如 0.005/-0.005，可为负)
    if not (-1 < req.buy_threshold < 1):
        raise HTTPException(400, "buy_threshold 必须在 (-1, 1) 之间")
    if not (-1 < req.sell_threshold < 1):
        raise HTTPException(400, "sell_threshold 必须在 (-1, 1) 之间")
    if req.buy_threshold <= req.sell_threshold:
        raise HTTPException(400, "buy_threshold 必须大于 sell_threshold")

    # 阈值尺度校验（请求即拒）：读 checkpoint 的 objective 后按其口径校验 buy/sell 阈值，
    # 回归模型误用概率阈值（buy>=0.5）/ 概率模型阈值越界 → 当场 400，不静默空跑。
    # 与回测策略、实盘 service 共用同一 threshold_scale_check（回测实盘一致红线）。
    # 模型不存在时跳过此校验，交由 _run_cnn_backtest 在任务内抛出"模型不存在"（保持既有行为）。
    from ..cnn.storage import CNN_MODEL_DIR
    from ..cnn.thresholds import threshold_scale_check

    model_path = CNN_MODEL_DIR / f"{req.model}.pt"
    if model_path.exists():
        import torch

        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        objective = checkpoint.get("train_config", {}).get("objective", "classification")
        reasons = threshold_scale_check(objective, req.buy_threshold, req.sell_threshold)
        if reasons:
            raise HTTPException(400, "；".join(reasons))

    task_id = task_manager.create_task(
        TaskType.CNN_BACKTEST,
        params={"name": req.name, "model": req.model},
        title=f"CNN 回测: {req.model}",
        entity_type="cnn_backtest",
        entity_name=req.name,
    )

    def execute(on_progress: Optional[Callable[[float, str], None]] = None) -> dict:
        """CNN 回测任务的后台执行体：在任务线程内调用 _run_cnn_backtest 并返回回测结果。

        Args:
            on_progress: 进度回调 ``(percent, message)``，转交给回测过程逐步上报；可为 None。

        Returns:
            _run_cnn_backtest 产出的回测结果字典。
        """
        return _run_cnn_backtest(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "CNN 回测任务已启动"}


# =============================================================================
# 推理（生成信号）
# =============================================================================

def _run_cnn_predict(
    req: CNNPredictRequest,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """执行 CNN 推理，将信号保存到信号库（供 Alpha 回测复用）。

    在 CNN_PREDICT 任务的后台线程内运行：校验模型存在 → 调用 predict_cnn_signals 生成
    [datetime, vt_symbol, signal] 信号表 → 以请求名保存到 AlphaLab 信号库。期间通过
    on_progress 上报进度（5% 起步、推理段映射到 5~90%、保存后 100%）。

    Args:
        req: CNN 推理请求，含 name（保存的信号名）、model（模型名）、start/end（推理区间）。
        on_progress: 进度回调 ``(percent, message)``，可为 None（不上报）。

    Returns:
        结果字典。产生信号时含 name/model/signal_count/signal_mean/start/end；
        未产生任何信号时含 name/signal_count=0/message，且不写入信号库。

    Raises:
        ValueError: CNN 模型文件不存在时抛出。
    """
    from ..alpha import AlphaLab
    from ..config import ALPHA_LAB_PATH
    from ..cnn.predictor import predict_cnn_signals
    from ..cnn.storage import CNN_MODEL_DIR

    # 1. 校验模型存在
    model_path = CNN_MODEL_DIR / f"{req.model}.pt"
    if not model_path.exists():
        raise ValueError(f"CNN 模型不存在: {req.model}")

    if on_progress:
        on_progress(5, f"开始 CNN 推理: {req.name}")

    # 2. CNN 推理生成信号
    def inference_progress(pct: float, msg: str) -> None:
        """推理进度适配器：把推理段的 0~100% 线性映射到整体推理任务进度的 5~90% 后上报。

        Args:
            pct: 推理自身进度百分比（0~100）。
            msg: 推理阶段的进度文案，会加上 "[推理] " 前缀。
        """
        if on_progress:
            # 将推理进度 (0-100) 映射到整体 (5-90)
            on_progress(5 + pct * 0.85, f"[推理] {msg}")

    signal_df = predict_cnn_signals(
        model_name=req.model,
        start=req.start,
        end=req.end,
        on_progress=inference_progress,
    )

    if signal_df.is_empty():
        return {
            "name": req.name,
            "signal_count": 0,
            "message": "CNN 推理未产生任何信号",
        }

    # 3. 保存信号到信号库
    if on_progress:
        on_progress(92, "保存信号...")

    lab = AlphaLab(ALPHA_LAB_PATH)
    lab.save_signal(req.name, signal_df)

    if on_progress:
        on_progress(100, f"信号已保存: {req.name} ({signal_df.height} 条)")

    return {
        "name": req.name,
        "model": req.model,
        "signal_count": signal_df.height,
        "signal_mean": float(signal_df["signal"].mean()),
        "start": str(req.start),
        "end": str(req.end),
    }


@router.post("/predict")
async def start_cnn_predict(req: CNNPredictRequest) -> dict:
    """启动 CNN 推理任务，生成信号并保存到信号库，异步执行并立即返回 task_id。

    POST /api/cnn/predict 的处理函数。校验通过后创建 CNN_PREDICT 任务交后台线程执行；
    任务产出的信号会写入信号库，可在「Alpha 因子回测」中作为普通信号直接复用。

    Args:
        req: CNN 推理请求，含 name（保存的信号名）、model（模型名）、start/end（推理区间，
            start 须早于 end）。

    Returns:
        含 task_id（任务 ID，用于轮询）与 message（启动提示）的字典。

    Raises:
        HTTPException: PyTorch 未安装或日期顺序错误时返回 400。
    """
    if not _check_torch():
        raise HTTPException(400, "PyTorch 未安装，请先执行: pip install torch")

    if req.start >= req.end:
        raise HTTPException(400, "开始日期必须早于结束日期")

    task_id = task_manager.create_task(
        TaskType.CNN_PREDICT,
        params={"name": req.name, "model": req.model},
        title=f"CNN 推理: {req.model}",
        entity_type="cnn_signal",
        entity_name=req.name,
    )

    def execute(on_progress: Optional[Callable[[float, str], None]] = None) -> dict:
        """CNN 推理任务的后台执行体：在任务线程内调用 _run_cnn_predict 生成并保存信号。

        Args:
            on_progress: 进度回调 ``(percent, message)``，转交给推理过程逐步上报；可为 None。

        Returns:
            _run_cnn_predict 产出的结果字典（含信号统计或未产生信号提示）。
        """
        return _run_cnn_predict(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "CNN 推理任务已启动"}
