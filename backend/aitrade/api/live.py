"""
交易操作台（Trading Console）API 路由 `/api/live/*`。

把既有实盘原语（`predict_cnn_signals` → `SignalService` → `RiskManager` →
`Notifier` → `DecisionStore`）经 `LiveSignalOrchestrator` 串联后暴露为 HTTP 接口：

- `POST   /api/live/decision`：触发一次今日决策（异步任务，返回 task_id）。
- `GET    /api/live/decisions`：列出已持久化决策的 signal_id 集合。
- `GET    /api/live/decisions/{signal_id}`：单条决策详情。
- `DELETE /api/live/decisions/{signal_id}`：归档式删除决策 + trace（解除幂等占位）。
- `POST   /api/live/decisions/batch-delete`：批量归档式删除（部分成功语义）。

安全边界（v1）：本路由**不 import 也不调用任何券商网关 / 下单接口**，产出仅限
Decision 落盘 + Notifier 提醒（Property 7）。当前后端无鉴权（TD-015），真实下单能力
依赖鉴权与 kill-switch UI 前置条件，超出本特性范围。
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any
from collections.abc import Callable

from fastapi import APIRouter, Body, HTTPException

from ..alpha.lab import AlphaLab
from ..alpha.lab_utils import normalize_vt_symbol
from ..config import (
    ALPHA_LAB_PATH,
    CNN_MODEL_PATH,
    LIVE_DECISION_PATH,
    LIVE_PORTFOLIO_PATH,
    LIVE_REBALANCE_PATH,
    LIVE_RUNTIME_STATE_PATH,
    SCHEDULER_RUN_LOG_PATH,
    TRADING_PLAN_PATH,
)
from ..live.decision import Decision, DecisionStore
from ..live.jsonl_store import JsonlDayStore
from ..live.position_book import PositionBook
from ..live.rebalance_decision import RebalanceStore
from ..live.scheduler_run_log import SchedulerRunLog
from ..live.decision_instant import (
    INTRADAY_BAR_FREQS,
    DecisionInstant,
    bar_freq_of_interval,
)
from ..live.decision_trace import DecisionTraceStore
from ..live.notifier import LogNotifier
from ..live.notifier_channels import build_notifier
from ..live.orchestrator import run_live_decision
from ..live.plan_scheduler import (
    _LAST_TRIGGERED_KEY,
    PlanScheduler,
    _normalize_last_triggered,
)
from ..live.portfolio_risk import PortfolioRiskConfig, PortfolioRiskManager
from ..live.runtime_state import RuntimeStateStore
from ..live.trading_plan import (
    TradingPlan,
    TradingPlanStore,
    effective_trigger_times,
)
from ..live.rebalance import run_rebalance_decision
from ..models import TaskType
from ..models.live import (
    LiveDecisionRequest,
    PortfolioSnapshotRequest,
    RebalanceRequest,
    RiskConfigRequest,
)
from ..models.trading_plan import (
    SchedulerStatus,
    TradingPlanRequest,
    TradingPlanSummary,
)
from ..task import task_manager

router = APIRouter(
    prefix="/api/live",
    tags=["交易操作台"],
)

# 🆕 决策存储目录接线：模块级单实例，每个 signal_id 一个 JSON 文件（既有 DecisionStore 行为）。
_store = DecisionStore(LIVE_DECISION_PATH)

# 🆕 决策过程档案存储：与 _store 同目录，每个 signal_id 一个 {signal_id}.trace.json（sibling 于决策 JSON）。
_trace_store = DecisionTraceStore(LIVE_DECISION_PATH)

# 🆕 交易计划自动化：计划存储 + 调度运行时状态（Last_Triggered_Map）。
_plan_store = TradingPlanStore(TRADING_PLAN_PATH)
_runtime_state = RuntimeStateStore(LIVE_RUNTIME_STATE_PATH)

# 进程内调度器单例引用（由 main.py lifespan 装配并注册；状态端点据此读取）。
_scheduler: PlanScheduler | None = None

# Phase 3 M2：调仓决策 + 持仓账本单例（RebalanceStore/PositionBook 首次实例化自动 mkdir）。
_rebalance_store = RebalanceStore(LIVE_REBALANCE_PATH)
_position_book = PositionBook(LIVE_PORTFOLIO_PATH)

# task-scheduler-observability Wave 2b: 调度运行日志单例（供后续 API 端点及 build_plan_scheduler 使用）。
# M1 修复：JsonlDayStore 使用本地时间分桶，与调度器写入时刻口径一致（调度器 now_fn 默认 datetime.now 本地时间）。
_scheduler_run_log = SchedulerRunLog(JsonlDayStore(SCHEDULER_RUN_LOG_PATH, now_fn=datetime.now))


def _get_lab() -> AlphaLab:
    """创建 AlphaLab 实例（轻量对象，无状态副作用）。

    每次调用均返回新实例；测试通过 monkeypatch 此函数注入含行情的预置 lab，
    无需修改调用方代码。

    Returns:
        以 ``ALPHA_LAB_PATH`` 为根目录的 AlphaLab 实例。
    """
    return AlphaLab(ALPHA_LAB_PATH)


def _decision_to_dict(decision: Decision) -> dict[str, Any]:
    """将 Decision 数据类实例转为可序列化 dict（用于 API 响应）。

    Args:
        decision: Decision dataclass 实例。

    Returns:
        Decision 的 ``dataclasses.asdict`` 序列化结果（datetime 以 isoformat 字符串表示）。
    """
    return asdict(decision)


def _validate_bar_freq_against_model(model: str, bar_freq: str) -> None:
    """间隔锁定（Req 2）：校验 bar_freq 是否 == 模型训练间隔对应的 bar_freq，不一致则抛错。

    模型在固定间隔的 bar 分布上训练，喂别的周期是分布外输入——与回测端
    「回测周期必须 = 训练周期」是同一条红线在实时端的延伸。在 start_live_decision /
    创建计划 / 更新计划等入口处调用，作为下游编排前的前置校验。

    校验策略按 checkpoint 可读性分流：

    - checkpoint 可读 → 训练间隔与 bar_freq 不一致时返回 400；一致则静默放过；
    - 模型缺失/不可读 → 日内 bar_freq 返回 404（日内必须能锁定间隔）；
      "1d" 放过（保持既有宽松行为：允许先建计划，触发时再报错）。

    模型路径源与本路由其它模型存在性检查一致（模块级 ``CNN_MODEL_PATH``，测试可整体替换）。

    Args:
        model: CNN 模型名（不含 ``.pt`` 扩展名），用于在 ``CNN_MODEL_PATH`` 下定位
            checkpoint 文件 ``{model}.pt`` 并读取其训练输入间隔。
        bar_freq: 本次请求/计划拟用的 K 线周期，如 "1d"（日线）或属于
            ``INTRADAY_BAR_FREQS`` 的日内周期（如 "30m"）；须与模型训练间隔对应的
            bar_freq 一致。

    Returns:
        None。校验通过（一致，或模型不可读但 bar_freq 为 "1d"）时静默返回，
        不一致或日内无法锁定间隔时改为抛出 HTTPException。

    Raises:
        HTTPException: 状态码 400——checkpoint 可读但 bar_freq 与模型训练间隔对应的
            bar_freq 不一致。
        HTTPException: 状态码 404——模型缺失或不可读，且 bar_freq 为日内周期
            （属于 ``INTRADAY_BAR_FREQS``），无法锁定训练间隔。
    """
    from ..cnn.storage import checkpoint_input_interval

    try:
        interval = checkpoint_input_interval(CNN_MODEL_PATH / f"{model}.pt")
    except Exception:  # noqa: BLE001 — 缺失或不可读：均视为无法锁定间隔
        if bar_freq in INTRADAY_BAR_FREQS:
            raise HTTPException(
                404, f"CNN 模型不存在或不可读: {model}（日内计划必须锁定模型训练间隔）"
            )
        return
    expected = bar_freq_of_interval(interval)
    if bar_freq != expected:
        raise HTTPException(
            400,
            f"bar_freq 与模型训练间隔不一致：模型 {model} 训练间隔为 {interval}"
            f"（对应 bar_freq={expected}），请求为 {bar_freq}",
        )


@router.post(
    "/decision",
    description=(
        "触发一次今日决策（异步任务，返回 task_id），仅产出决策与提醒，不下任何真实订单。"
        "当前处于无鉴权环境（TD-015），真实下单能力依赖鉴权与 kill-switch UI 前置条件。"
    ),
)
async def start_live_decision(req: LiveDecisionRequest) -> dict:
    """触发今日决策：校验必填字段/模型存在/间隔锁定后，创建后台异步任务并立即返回 task_id。

    校验通过即把 ``run_live_decision`` 编排封进后台任务（不阻塞请求），仅产出 Decision
    落盘 + Notifier 提醒，绝不下任何真实订单。前端用返回的 task_id 轮询任务结果。

    Args:
        req: 今日决策请求体。其中 model（CNN 模型名）、vt_symbol（标的代码）、scheme
            （方案名）为业务必填非空；as_of 为决策时刻（缺省取当前本地时间）；bar_freq
            须与模型训练间隔一致；其余字段（buy_threshold/portfolio/risk/position_ratio
            等）透传给编排器。

    Returns:
        形如 ``{"task_id": "...", "message": "今日决策任务已启动"}``。

    Raises:
        HTTPException(400): 缺 model/vt_symbol/scheme，或 bar_freq 与模型训练间隔不一致。
        HTTPException(404): 指定 CNN 模型文件不存在；或日内计划下模型不可读无法锁定间隔。
    """
    # 400：缺必填字段（Pydantic 已校验类型，这里兜底业务必填非空）。
    if not req.model or not req.vt_symbol or not req.scheme:
        raise HTTPException(400, "缺少必填字段：model / vt_symbol / scheme")
    vt_symbol = normalize_vt_symbol(req.vt_symbol)

    # 404：模型不存在。
    if not (CNN_MODEL_PATH / f"{req.model}.pt").exists():
        raise HTTPException(404, f"CNN 模型不存在: {req.model}")

    # 间隔锁定（Req 2.1）：bar_freq 必须与模型训练间隔一致。
    _validate_bar_freq_against_model(req.model, req.bar_freq)

    # 决策时刻默认当前；仅 close_time <= as_of 的 bar 可见（无前视结构性保证）。
    instant = DecisionInstant(as_of=req.as_of or datetime.now(), bar_freq=req.bar_freq)

    portfolio = req.portfolio.to_domain()
    risk_config = req.risk.to_domain()

    task_id = task_manager.create_task(
        TaskType.LIVE_DECISION,
        params={"model": req.model, "vt_symbol": vt_symbol, "scheme": req.scheme},
        title=f"今日决策: {req.scheme}",
        entity_type="live_decision",
        entity_name=req.scheme,
    )

    def execute(on_progress: Callable[[float, str], None] | None = None) -> dict:
        """后台任务体：调用 run_live_decision 执行一次今日决策（手动触发）。

        作为闭包捕获本路由内已解析的模型、标的、组合与风控等参数，由 task_manager
        在后台线程调用；仅产出决策与提醒，不下真实订单。

        Args:
            on_progress: 可选进度回调，签名 ``(进度比例, 阶段描述)``，由任务框架注入用于上报进度。

        Returns:
            run_live_decision 的执行结果 dict。
        """
        return run_live_decision(
            model_name=req.model,
            vt_symbol=vt_symbol,
            scheme_name=req.scheme,
            instant=instant,
            portfolio=portfolio,
            buy_threshold=req.buy_threshold,
            risk_config=risk_config,
            store=_store,
            notifier=LogNotifier(),
            position_ratio=req.position_ratio,
            min_volume=req.min_volume,
            model_version=req.model_version,
            should_exit=req.should_exit,
            halted=req.halted,
            on_progress=on_progress,
            trace_store=_trace_store,
            trigger_source="manual",
        )

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "今日决策任务已启动"}


@router.get(
    "/decisions",
    description="列出 DecisionStore 中已持久化决策的 signal_id 集合。",
)
async def list_decisions() -> dict:
    """列出 DecisionStore 中所有已持久化决策的标识符集合（Req 4.1）。

    Returns:
        形如 ``{"signal_ids": ["...", ...]}``；无决策时 signal_ids 为空列表。
    """
    return {"signal_ids": _store.list_ids()}


@router.get(
    "/decisions/{signal_id}",
    description="按 signal_id 返回单条决策详情，不存在则 404。",
)
async def get_decision(signal_id: str) -> dict:
    """返回指定 signal_id 的完整决策详情（Req 4.2 / 4.3）。

    Args:
        signal_id: 决策幂等键。

    Returns:
        Decision 的完整 dict 序列化（datetime 字段为 isoformat 字符串）。

    Raises:
        HTTPException(404): signal_id 对应决策不存在。
    """
    decision = _store.get(signal_id)
    if decision is None:
        raise HTTPException(404, f"决策不存在: {signal_id}")
    return _decision_to_dict(decision)


@router.get(
    "/decisions/{signal_id}/trace",
    description="按 signal_id 返回该决策的完整 Decision_Trace（六段决策过程档案），不存在则 404。",
)
async def get_decision_trace(signal_id: str) -> dict:
    """返回指定 signal_id 的完整 Decision_Trace（六段决策过程档案）（Req 8.4 / 8.5）。

    Args:
        signal_id: 决策幂等键（与对应 Decision 共用同一键）。

    Returns:
        决策过程档案的完整 dict（六段决策过程）。

    Raises:
        HTTPException(404): signal_id 对应过程档案不存在。
    """
    trace = _trace_store.get(signal_id)
    if trace is None:
        raise HTTPException(404, f"决策过程档案不存在: {signal_id}")  # 满足 8.5
    return trace


@router.delete(
    "/decisions/{signal_id}",
    description=(
        "归档式删除单条决策及其过程档案：文件移入 archive/ 子目录（保留审计痕迹），"
        "并解除该 signal_id 的幂等占位，使同一 Decision_Bar 可重新产出决策与提醒。"
        "注意：若当日交易计划仍有未到期的触发时点，删除后会重新决策并再次提醒。"
    ),
)
async def delete_decision(signal_id: str) -> dict:
    """归档式删除单条决策及其过程档案，并解除该 signal_id 的幂等占位。

    决策与 trace 整体处理（避免「新决策配旧档案」错位）：文件移入 archive/ 子目录保留
    审计痕迹，同时解除幂等占位——若当日交易计划仍有未到期触发时点，删除后会重新决策并再次提醒。

    Args:
        signal_id: 待归档决策的幂等键。

    Returns:
        形如 ``{"signal_id": "...", "deleted": True, "trace_archived": bool}``；
        trace_archived 表示是否存在并归档了对应过程档案。

    Raises:
        HTTPException(404): signal_id 对应决策不存在（trace 缺失不报错）。
    """
    archived = _store.archive(signal_id)
    if archived is None:
        raise HTTPException(404, f"决策不存在: {signal_id}")
    trace_archived = _trace_store.archive(signal_id)
    return {
        "signal_id": signal_id,
        "deleted": True,
        "trace_archived": trace_archived is not None,
    }


@router.post(
    "/decisions/batch-delete",
    description=(
        "批量归档式删除决策及其过程档案（语义与单条 DELETE 一致：移入 archive/、"
        "解除幂等占位）。部分成功：存在的逐条归档，不存在的归入 missing 返回，"
        "不因个别缺失而整体失败。signal_ids 为空则 400。"
    ),
)
async def batch_delete_decisions(signal_ids: list[str] = Body(..., embed=True)) -> dict:
    """批量归档式删除决策及其过程档案（部分成功语义，不因个别缺失整体失败）。

    语义与单条 DELETE 一致：存在的逐条移入 archive/ 并解除幂等占位，不存在的归入 missing。
    入参先去重后保序处理，结果列表同样保持入参顺序。

    Args:
        signal_ids: 待归档的决策幂等键列表（embed 在请求体中）；不能为空。

    Returns:
        形如 ``{"deleted": [...], "missing": [...]}``：deleted 为成功归档的 id，
        missing 为不存在的 id，两者均去重保序。

    Raises:
        HTTPException(400): signal_ids 为空。
    """
    if not signal_ids:
        raise HTTPException(400, "signal_ids 不能为空")
    deleted: list[str] = []
    missing: list[str] = []
    for signal_id in dict.fromkeys(signal_ids):  # 去重且保序
        if _store.archive(signal_id) is None:
            missing.append(signal_id)
        else:
            _trace_store.archive(signal_id)
            deleted.append(signal_id)
    return {"deleted": deleted, "missing": missing}


# =============================================================================
# 交易计划自动化（Trading Plan Automation）：计划 CRUD + 按计划触发 + 调度状态
# =============================================================================


def _trigger_plan(
    plan: TradingPlan,
    on_progress: Callable[[float, str], None] | None = None,
    trigger_source: str = "manual",
) -> dict:
    """用计划配置触发一次今日决策（手动与自动调度共用同一编排）。

    按 ``plan.strategy_type`` 分派：
    - "rule"  → ``run_rebalance_decision``（组合调仓，规则信号）
    - "cnn"（默认）→ ``run_live_decision``（单标的，CNN 推理）；原有路径零改动。

    通知通道由 ``build_notifier(plan.notify_channels)`` 在运行时按环境变量装配（凭证不入计划）。
    不调用任何券商网关 / 下单接口（Property 12）。

    Args:
        plan: 已持久化的交易计划；其 strategy_type 决定分派路径，其余字段透传给对应编排器。
        on_progress: 可选进度回调 ``(fraction, message)``；由任务管理器注入，用于上报执行进度。
        trigger_source: 触发来源标记，"manual"（手动）或 "scheduler"（自动调度），随决策落盘。

    Returns:
        对应编排器（rule 走 run_rebalance_decision，cnn 走 run_live_decision）的结果 dict，
        含决策内容、幂等命中标记、风控与（可能的）跳过原因。
    """
    if plan.strategy_type == "rule":
        # rule 计划路径：组合调仓编排。
        # capital 从 plan.portfolio dict 取 "portfolio_value" 键，缺省 1_000_000。
        capital = float((plan.portfolio or {}).get("portfolio_value", 1_000_000))
        # rules 包在 api/strategy.py import 时已完成注册；live 路径在 api/live.py 模块级
        # import api.strategy 前可能尚未注册——此处补充确保注册（import 幂等，无副作用）。
        from .. import rules  # noqa: F401  确保 etf_momentum 等信号源已注册
        return run_rebalance_decision(
            plan_name=plan.name,
            signal_source=plan.signal_source,
            signal_params=dict(plan.signal_params),
            strategy_params={
                "top_k": (plan.portfolio or {}).get("top_k", 5),
                "min_volume": plan.min_volume,
            },
            portfolio_id=plan.portfolio_id or plan.scheme,
            instant=DecisionInstant(as_of=datetime.now(), bar_freq=plan.bar_freq),
            capital=capital,
            rebalance_store=_rebalance_store,
            position_book=_position_book,
            risk_manager=PortfolioRiskManager(_runtime_state, _portfolio_risk_config),
            notifier=build_notifier(plan.notify_channels),
            lab=_get_lab(),
            on_progress=on_progress,
            trigger_source=trigger_source,
        )
    # 默认 "cnn" 路径：单标的 CNN 推理，原有逻辑零改动。
    portfolio = PortfolioSnapshotRequest(**plan.portfolio).to_domain()
    risk_config = RiskConfigRequest(**(plan.risk or {})).to_domain()
    return run_live_decision(
        model_name=plan.model,
        vt_symbol=normalize_vt_symbol(plan.vt_symbol),
        scheme_name=plan.scheme,
        instant=DecisionInstant(as_of=datetime.now(), bar_freq=plan.bar_freq),
        portfolio=portfolio,
        buy_threshold=plan.buy_threshold,
        risk_config=risk_config,
        store=_store,
        notifier=build_notifier(plan.notify_channels),  # 🆕 真实通道装配
        position_ratio=plan.position_ratio,
        min_volume=plan.min_volume,
        model_version=plan.model_version,
        should_exit=plan.should_exit,
        halted=plan.halted,
        on_progress=on_progress,
        trace_store=_trace_store,
        data_source_type=plan.data_source,
        trigger_source=trigger_source,
    )


def _request_to_plan(req: TradingPlanRequest, *, plan_id: str, created_at: str | None = None) -> TradingPlan:
    """将 TradingPlanRequest 请求体映射为可持久化的 TradingPlan 数据对象。

    notify_channels 仅存储通道名（无凭证），凭证在运行时经环境变量由 build_notifier 装配。
    PUT 操作需传 created_at 以保持创建时间不变；POST 操作不传，由 TradingPlan 默认工厂生成。

    Args:
        req:        解析/校验通过的请求体。
        plan_id:    目标计划 ID（POST 时传新生成 ID，PUT 时传既有 ID）。
        created_at: 可选的创建时间字符串（ISO 格式）；None 时由 TradingPlan 默认工厂填入。

    Returns:
        对应的 TradingPlan dataclass 实例，已透传 Phase 3 M2 新字段（strategy_type 等）。
    """
    plan = TradingPlan(
        plan_id=plan_id,
        name=req.name,
        model=req.model,
        vt_symbol=req.vt_symbol,
        scheme=req.scheme,
        buy_threshold=req.buy_threshold,
        position_ratio=req.position_ratio,
        min_volume=req.min_volume,
        model_version=req.model_version,
        data_source=req.data_source,
        should_exit=req.should_exit,
        halted=req.halted,
        portfolio=req.portfolio.model_dump(),
        risk=req.risk.model_dump(),
        enabled=req.enabled,
        bar_freq=req.bar_freq,
        trigger_times=list(req.trigger_times),
        notify_channels=list(req.notify_channels),
        # Phase 3 M2：新字段透传
        strategy_type=req.strategy_type,
        signal_source=req.signal_source,
        signal_params=dict(req.signal_params),
        trigger_schedule=req.trigger_schedule,
        portfolio_id=req.portfolio_id,
    )
    if created_at is not None:
        plan.created_at = created_at
    return plan


def _plan_summary(plan: TradingPlan) -> TradingPlanSummary:
    """将 TradingPlan 转为计划列表摘要 TradingPlanSummary。

    从运行时状态中取 Last_Triggered_Map，归一化为 ``{plan_id: "YYYY-MM-DD"}``
    后填入 last_triggered 字段，兼容新（``{date, slots}``）与旧（``"YYYY-MM-DD"``）两种状态形态。

    Args:
        plan: 要转换的 TradingPlan 实例。

    Returns:
        TradingPlanSummary，含 plan_id / name / enabled / last_triggered 等摘要字段。
    """
    # last_triggered 归一化为日期字符串，兼容新（{date,slots}）/旧（"YYYY-MM-DD"）状态形态。
    mapping = _normalize_last_triggered(_runtime_state.get(_LAST_TRIGGERED_KEY, {}) or {})
    return TradingPlanSummary(
        plan_id=plan.plan_id,
        name=plan.name,
        vt_symbol=plan.vt_symbol,
        scheme=plan.scheme,
        bar_freq=plan.bar_freq,
        trigger_times=effective_trigger_times(plan),
        enabled=plan.enabled,
        last_triggered=mapping.get(plan.plan_id),
        strategy_type=plan.strategy_type,
        portfolio_id=plan.portfolio_id,
        signal_source=plan.signal_source,
    )


@router.post("/plans", description="创建交易计划，返回完整内容（notify_channels 仅通道名，无凭证）。")
async def create_plan(req: TradingPlanRequest) -> dict:
    """创建一个交易计划并持久化（Req 2.1）。

    先做间隔锁定校验（bar_freq 必须与模型训练间隔一致），再生成新 plan_id 落盘。

    Args:
        req: 交易计划请求体；notify_channels 仅存通道名，凭证不入计划。

    Returns:
        新建计划的完整内容 dict（含自动生成的 plan_id / created_at）。

    Raises:
        HTTPException(400/404): 间隔锁定校验失败（详见 ``_validate_bar_freq_against_model``）。
    """
    _validate_bar_freq_against_model(req.model, req.bar_freq)  # 间隔锁定
    plan = _request_to_plan(req, plan_id=TradingPlan.new_id())
    _plan_store.save(plan)
    return asdict(plan)


@router.get("/plans", description="列出所有交易计划摘要（含启用状态与最近触发日）。")
async def list_plans() -> list[dict]:
    """列出所有交易计划的摘要（含启用状态与最近触发日）（Req 2.2）。

    Returns:
        TradingPlanSummary dict 列表；无计划时返回空列表。
    """
    return [_plan_summary(p).model_dump() for p in _plan_store.list_all()]


@router.get("/plans/{plan_id}", description="按 plan_id 返回计划完整内容，不存在则 404。")
async def get_plan(plan_id: str) -> dict:
    """按 plan_id 返回交易计划的完整内容（Req 2.3 / 2.7）。

    Args:
        plan_id: 计划 ID。

    Returns:
        计划完整内容 dict。

    Raises:
        HTTPException(404): plan_id 对应计划不存在。
    """
    plan = _plan_store.get(plan_id)
    if plan is None:
        raise HTTPException(404, f"交易计划不存在: {plan_id}")
    return asdict(plan)


@router.put("/plans/{plan_id}", description="按 plan_id 更新计划（保持 plan_id 与 created_at 不变）。")
async def update_plan(plan_id: str, req: TradingPlanRequest) -> dict:
    """按 plan_id 整体更新交易计划，保持 plan_id 与 created_at 不变（Req 2.4 / 2.7）。

    先确认计划存在并通过间隔锁定校验，再用请求体重建计划落盘（updated_at 自动刷新）。

    Args:
        plan_id: 待更新计划的 ID。
        req:     新的计划内容请求体。

    Returns:
        更新后计划的完整内容 dict。

    Raises:
        HTTPException(404): plan_id 对应计划不存在。
        HTTPException(400/404): 间隔锁定校验失败（详见 ``_validate_bar_freq_against_model``）。
    """
    existing = _plan_store.get(plan_id)
    if existing is None:
        raise HTTPException(404, f"交易计划不存在: {plan_id}")
    _validate_bar_freq_against_model(req.model, req.bar_freq)  # 间隔锁定
    plan = _request_to_plan(req, plan_id=plan_id, created_at=existing.created_at)
    _plan_store.save(plan)  # updated_at 由 dataclass 默认工厂在重建时刷新
    return asdict(plan)


@router.delete("/plans/{plan_id}", description="按 plan_id 删除计划，不存在则 404。")
async def delete_plan(plan_id: str) -> dict:
    """按 plan_id 删除交易计划（Req 2.5 / 2.7）。

    Args:
        plan_id: 待删除计划的 ID。

    Returns:
        形如 ``{"plan_id": "...", "deleted": True}``。

    Raises:
        HTTPException(404): plan_id 对应计划不存在。
    """
    if not _plan_store.delete(plan_id):
        raise HTTPException(404, f"交易计划不存在: {plan_id}")
    return {"plan_id": plan_id, "deleted": True}


@router.patch("/plans/{plan_id}/enabled", description="启用/停用计划。")
async def toggle_plan(plan_id: str, enabled: bool = Body(..., embed=True)) -> dict:
    """启用或停用指定交易计划（Req 2.6 / 2.7）。

    Args:
        plan_id: 目标计划 ID。
        enabled: 目标启用状态（True 启用 / False 停用），embed 在请求体中。

    Returns:
        形如 ``{"plan_id": "...", "enabled": bool}``，回显设置后的状态。

    Raises:
        HTTPException(404): plan_id 对应计划不存在。
    """
    plan = _plan_store.get(plan_id)
    if plan is None:
        raise HTTPException(404, f"交易计划不存在: {plan_id}")
    plan.enabled = enabled
    _plan_store.save(plan)
    return {"plan_id": plan_id, "enabled": enabled}


@router.post(
    "/plans/{plan_id}/run",
    description=(
        "按计划立即触发一次今日决策（异步任务，返回 task_id），仅产出决策与提醒，不下任何真实订单。"
        "当前处于无鉴权环境（TD-015）。"
    ),
)
async def run_plan(plan_id: str) -> dict:
    """按已有计划立即手动触发一次今日决策（异步任务，返回 task_id）（Req 3.1 / 3.4）。

    取出计划后封进后台任务执行 ``_trigger_plan``，仅产出决策与提醒，不下任何真实订单。

    Args:
        plan_id: 待触发计划的 ID。

    Returns:
        形如 ``{"task_id": "...", "message": "按计划触发任务已启动"}``，前端据此轮询结果。

    Raises:
        HTTPException(404): plan_id 对应计划不存在。
    """
    plan = _plan_store.get(plan_id)
    if plan is None:
        raise HTTPException(404, f"交易计划不存在: {plan_id}")

    task_id = task_manager.create_task(
        TaskType.LIVE_DECISION,
        title=f"按计划触发: {plan.name}",
        entity_type="trading_plan",
        entity_name=plan.name,
    )

    def execute(on_progress: Callable[[float, str], None] | None = None) -> dict:
        """后台任务体：按已取出的计划调用 _trigger_plan 触发一次今日决策。

        闭包捕获本路由查出的 plan，由 task_manager 在后台线程调用；仅产出决策与提醒，不下真实订单。

        Args:
            on_progress: 可选进度回调，签名 ``(进度比例, 阶段描述)``，由任务框架注入用于上报进度。

        Returns:
            _trigger_plan 的执行结果 dict。
        """
        return _trigger_plan(plan, on_progress=on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "按计划触发任务已启动"}


@router.get("/scheduler/status", description="返回进程内调度器运行状态与各计划最近触发日。")
async def scheduler_status() -> dict:
    """返回进程内调度器的运行状态与各计划最近触发日（Req 8.5）。

    调度器单例缺失（未装配）时退化为 running=False、按计划存储统计启用数。

    Returns:
        SchedulerStatus dict，含 running（是否在跑）、tick_seconds（轮询间隔秒）、
        enabled_plan_count（启用计划数）、last_triggered（``{plan_id: "YYYY-MM-DD"}``，兼容新旧状态形态）。
    """
    # 归一化为 {plan_id: "YYYY-MM-DD"}，满足 SchedulerStatus.last_triggered: dict[str,str]（兼容新旧值）。
    mapping = _normalize_last_triggered(_runtime_state.get(_LAST_TRIGGERED_KEY, {}) or {})
    running = _scheduler.is_running() if _scheduler is not None else False
    from ..config import SCHEDULER_TICK_SECONDS

    enabled_count = (
        _scheduler.enabled_plan_count()
        if _scheduler is not None
        else sum(1 for p in _plan_store.list_all() if p.enabled)
    )
    return SchedulerStatus(
        running=running,
        tick_seconds=SCHEDULER_TICK_SECONDS,
        enabled_plan_count=enabled_count,
        last_triggered=mapping,
    ).model_dump()


@router.get(
    "/scheduler/runs",
    description=(
        "查询调度运行日志（只读，默认当日，倒序，最新在前）。"
        "plan_id 过滤可选；date 格式 YYYY-MM-DD，非法日期返回 422。"
    ),
)
async def get_scheduler_runs(
    plan_id: str | None = None,
    date: str | None = None,  # noqa: A002 — FastAPI 查询参数名须与 API 文档一致
    limit: int = 200,
) -> list[dict]:
    """查询调度运行日志（R6.1 / TSO-7 API 侧只读）。

    Args:
        plan_id: 可选过滤计划 ID；不传则返回所有计划的日志。
        date:    查询日期，格式 ``YYYY-MM-DD``；不传默认当日（本地时间）。
        limit:   最多返回条数，默认 200；结果倒序（最新在前）。

    Returns:
        事件记录列表，每条形态::

            {"ts": "...", "event": "skip"|"trigger"|"error",
             "plan_id": "...", "reason"|"slot"|"error": "...", ...}

        无匹配记录时返回空列表。

    Raises:
        HTTPException: 状态码 422——date 非空且不能按 ``YYYY-MM-DD`` 解析
            （``date.fromisoformat`` 抛 ValueError）时抛出。

    Example::

        GET /api/live/scheduler/runs?plan_id=my_plan&date=2026-06-12&limit=50
    """
    from datetime import date as _date_cls

    if date is not None:
        try:
            query_day = _date_cls.fromisoformat(date)
        except ValueError as exc:
            raise HTTPException(422, f"日期格式非法，应为 YYYY-MM-DD: {date!r}") from exc
    else:
        # 默认当日（本地时间，与写入口径一致）
        query_day = datetime.now().date()

    return _scheduler_run_log.query(plan_id=plan_id, day=query_day, limit=limit)


def build_plan_scheduler(*, tick_seconds: float) -> PlanScheduler:
    """装配 PlanScheduler（注入计划存储、运行时状态、_trigger_plan 与调度运行日志）。

    调度器触发路径固定传 trigger_source="scheduler"（Wave 2c），以便与手动触发区分落盘。
    通常由 main.py lifespan 在应用启动时调用。

    Args:
        tick_seconds: 调度轮询间隔（秒），决定多久检查一次各计划是否到达触发时点。

    Returns:
        已接线好依赖、可直接 start 的 PlanScheduler 实例。
    """
    return PlanScheduler(
        store=_plan_store,
        state=_runtime_state,
        trigger_fn=lambda plan: _trigger_plan(plan, trigger_source="scheduler"),
        tick_seconds=tick_seconds,
        run_log=_scheduler_run_log,
    )


def register_scheduler(scheduler: PlanScheduler | None) -> None:
    """注册（或清除）进程内调度器单例，供 ``scheduler_status`` 端点读取运行状态。

    由 main.py lifespan 在启动时注册实例、在关闭时传 None 清除。写入模块级全局 ``_scheduler``。

    Args:
        scheduler: 要注册的 PlanScheduler 实例；传 None 表示解除注册（端点将退化为按存储统计）。

    Returns:
        None。
    """
    global _scheduler
    _scheduler = scheduler


# =============================================================================
# Phase 3 M2：调仓决策（Rebalance）+ 持仓账本（Portfolio）端点
# =============================================================================


@router.get(
    "/rebalances",
    description="列出已持久化的调仓决策摘要（signal_id、status、portfolio_id、created_at）。",
)
async def list_rebalances() -> dict:
    """返回所有活跃调仓决策的摘要列表（不含 archive/ 子目录的已归档决策）。

    Returns:
        形如 ``{"items": [{signal_id, status, portfolio_id, created_at}, ...]}``，
        按文件名升序（即 signal_id 字典序）。

    Example::

        GET /api/live/rebalances
        → {"items": [{"signal_id": "rule:...:...", "status": "proposed", ...}]}
    """
    decisions = _rebalance_store.list_all()
    items = [
        {
            "signal_id": d.signal_id,
            "status": d.status,
            "portfolio_id": d.portfolio_id,
            "created_at": d.created_at,
        }
        for d in decisions
    ]
    return {"items": items}


@router.get(
    "/rebalances/{signal_id}",
    description="按 signal_id 返回完整调仓决策，不存在则 404。",
)
async def get_rebalance(signal_id: str) -> dict:
    """返回指定 signal_id 的完整 RebalanceDecision；不存在则 404。

    Args:
        signal_id: 幂等键，格式 ``rule:<plan_name>:<bar_dt>:<bar_freq>``。

    Returns:
        RebalanceDecision 的完整 dict 序列化，含 items / target_portfolio / risk_summary。

    Raises:
        HTTPException(404): signal_id 对应文件不存在。
    """
    decision = _rebalance_store.get(signal_id)
    if decision is None:
        raise HTTPException(404, f"调仓决策不存在: {signal_id}")
    return asdict(decision)


@router.get(
    "/portfolios/{portfolio_id}",
    description="返回指定持仓账本的当前状态；账本文件缺失时返回空账本（不 404）。",
)
async def get_portfolio(portfolio_id: str) -> dict:
    """返回持仓账本状态（缺失时返回空账本，不 404）。

    Args:
        portfolio_id: 组合 ID，对应 PositionBook 文件键（特殊字符自动安全化）。

    Returns:
        PortfolioState dict，含 portfolio_id / positions / last_signal_id / updated_at；
        账本文件不存在时返回 positions={} 的空账本。
    """
    state = _position_book.load(portfolio_id)
    return asdict(state)


@router.post(
    "/rebalances/{signal_id}/confirm",
    description=(
        "确认调仓决策：①决策不存在→404 ②已 confirmed→409 "
        "③apply_rebalance（超卖/重复→400/409）④更新 status=confirmed ⑤返回 {decision, portfolio}。"
    ),
)
async def confirm_rebalance(signal_id: str) -> dict:
    """确认调仓决策：将 items 应用到持仓账本，并将决策状态更新为 confirmed。

    幂等自愈设计：若账本已应用（last_signal_id 命中）而决策仍为 proposed（崩溃窗口遗留），
    仅补写决策状态，不重复应用账本，保持最终一致性。

    Args:
        signal_id: 待确认的调仓决策幂等键。

    Returns:
        ``{"decision": {...}, "portfolio": {...}}``，decision 为更新后的完整 RebalanceDecision，
        portfolio 为更新后的 PortfolioState。

    Raises:
        HTTPException(404): 决策不存在。
        HTTPException(409): 决策已 confirmed，或账本层面检测到重复确认。
        HTTPException(400): 超卖（卖出量超过当前持仓）或其他业务校验失败。
    """
    now_iso = datetime.now().isoformat(timespec="seconds")

    # ① 决策不存在 → 404
    decision = _rebalance_store.get(signal_id)
    if decision is None:
        raise HTTPException(404, f"调仓决策不存在: {signal_id}")

    # ② 已 confirmed → 409
    if decision.status == "confirmed":
        raise HTTPException(409, f"该调仓决策已确认过: {signal_id}")

    # ② 对账自愈：若决策仍为 proposed 但账本已应用（last_signal_id 命中），
    # 说明上次 apply_rebalance 成功而 update_status 在崩溃窗口内未完成，
    # 此时仅补写决策状态，不重复应用账本。
    book_state = _position_book.load(decision.portfolio_id)
    if book_state.last_signal_id == signal_id:
        # 半完成态自愈：账本已应用，仅补写决策状态以恢复一致性
        healed_decision = _rebalance_store.update_status(
            signal_id, "confirmed", confirmed_at=now_iso
        )
        return {
            "decision": asdict(healed_decision) if healed_decision is not None else asdict(decision),
            "portfolio": asdict(book_state),
        }

    # ③ apply_rebalance（ValueError → 409 防重复 / 400 超卖）
    try:
        portfolio_state = _position_book.apply_rebalance(decision.portfolio_id, decision)
    except ValueError as exc:
        msg = str(exc)
        # 防重复确认（账本层面）→ 409
        if "已确认过" in msg:
            raise HTTPException(409, msg) from exc
        # 超卖或其它业务校验失败 → 400
        raise HTTPException(400, msg) from exc

    # ④ 更新 status=confirmed + confirmed_at
    updated_decision = _rebalance_store.update_status(
        signal_id,
        status="confirmed",
        confirmed_at=now_iso,
    )

    # ⑤ 返回 {decision, portfolio}
    return {
        "decision": asdict(updated_decision) if updated_decision is not None else asdict(decision),
        "portfolio": asdict(portfolio_state),
    }


# =============================================================================
# Phase 3 M2 Task 3.4：组合级风控（PortfolioRiskManager）端点
# =============================================================================

# 模块级单例：复用 _runtime_state（键 "portfolio_risk" 与调度器键不冲突）
_portfolio_risk_config = PortfolioRiskConfig()


def _get_portfolio_risk_state(portfolio_id: str) -> dict:
    """从运行时状态中读取指定组合的风控状态，缺失时返回默认未熔断态。

    从 ``_runtime_state`` 的 "portfolio_risk" 键下取 portfolio_id 子 dict，
    组装为标准风控状态 dict 返回。组合从未初始化时 peak_value=None、broken=False。

    Args:
        portfolio_id: 组合 ID。

    Returns:
        含 portfolio_id / peak_value / broken / broken_date / reason 的 dict。
    """
    all_states: dict = _runtime_state.get("portfolio_risk", {}) or {}
    pstate = all_states.get(portfolio_id) or {}
    return {
        "portfolio_id": portfolio_id,
        "peak_value": pstate.get("peak_value"),
        "broken": bool(pstate.get("broken", False)),
        "broken_date": pstate.get("broken_date"),
        "reason": pstate.get("reason"),
    }


@router.get(
    "/portfolio-risk/{portfolio_id}",
    description="返回指定组合的当前风控状态；组合从未初始化时返回默认未熔断态。",
)
async def get_portfolio_risk(portfolio_id: str) -> dict:
    """查询组合风控状态（Task 3.4）。

    Args:
        portfolio_id: 组合 ID。

    Returns:
        形如 ``{portfolio_id, peak_value, broken, broken_date, reason}``；
        若该组合从未经过 evaluate，peak_value 为 None，broken 为 False。
    """
    return _get_portfolio_risk_state(portfolio_id)


@router.post(
    "/portfolio-risk/{portfolio_id}/reset",
    description="人工复位组合熔断：清除 broken/broken_date/reason 并重置 peak（以新起点衡量）。",
)
async def reset_portfolio_risk(portfolio_id: str) -> dict:
    """人工复位组合风控熔断（Task 3.4）。

    清除 broken/broken_date/reason 并**清零 peak**（以新起点衡量），
    避免残留旧高点导致复位后立即重新触发熔断。

    Args:
        portfolio_id: 组合 ID。

    Returns:
        复位后的风控状态 dict（broken=False，peak_value=None）。
    """
    mgr = PortfolioRiskManager(_runtime_state, _portfolio_risk_config)
    mgr.reset(portfolio_id)
    return _get_portfolio_risk_state(portfolio_id)


# =============================================================================
# Phase 3 M2 Task 3.5/3.7：手动触发组合调仓决策
# =============================================================================


@router.post(
    "/rebalance",
    description=(
        "手动触发一次组合调仓决策（异步任务，返回 task_id）。"
        "plan_id 引用已有 rule 计划（展开参数），或内联传参二选一。"
        "仅产出调仓清单与提醒，不下任何真实订单（Property 7）。"
    ),
)
async def start_rebalance(req: RebalanceRequest) -> dict:
    """手动触发组合调仓决策异步任务（Task 3.7）。

    支持两种调用模式：
    1. **plan_id 引用模式**：传 ``plan_id`` 引用已有 rule 类型交易计划，
       自动展开 signal_source / strategy_params / portfolio_id 等参数。
    2. **内联模式**：直接传 ``plan_name + signal_source``（以及可选的
       ``signal_params`` / ``strategy_params`` / ``portfolio_id`` 等字段）。

    两种模式均不调用任何券商网关或真实下单接口（Property 7）。

    Args:
        req: RebalanceRequest，含 plan_id（引用模式）或 plan_name+signal_source（内联模式）。

    Returns:
        ``{"task_id": "...", "message": "..."}``，前端通过 ``GET /api/alpha/tasks/{task_id}``
        轮询任务结果。结果 dict 含 decision / idempotent_hit / risk / skipped_reason。

    Raises:
        HTTPException(404): plan_id 对应计划不存在。
        HTTPException(400): plan_id 对应计划不是 rule 类型；或内联模式缺少必填字段。
    """
    # plan_id 引用模式：从 _plan_store 取 rule 计划展开参数。
    if req.plan_id:
        plan = _plan_store.get(req.plan_id)
        if plan is None:
            raise HTTPException(404, f"交易计划不存在: {req.plan_id}")
        if plan.strategy_type != "rule":
            raise HTTPException(400, f"计划 {req.plan_id} 不是 rule 类型（strategy_type={plan.strategy_type!r}）")
        plan_name = plan.name
        signal_source = plan.signal_source
        signal_params = dict(plan.signal_params)
        strategy_params: dict = {
            "top_k": (plan.portfolio or {}).get("top_k", 5),
            "min_volume": plan.min_volume,
        }
        portfolio_id = plan.portfolio_id or plan.scheme
        capital = float((plan.portfolio or {}).get("portfolio_value", 1_000_000))
        as_of = req.as_of or datetime.now()
        bar_freq = plan.bar_freq
        notify_channels = plan.notify_channels
    else:
        # 内联模式：直接使用请求字段。
        if not req.plan_name or not req.signal_source:
            raise HTTPException(400, "内联模式下 plan_name 与 signal_source 必填非空")
        plan_name = req.plan_name
        signal_source = req.signal_source
        signal_params = dict(req.signal_params)
        strategy_params = dict(req.strategy_params)
        portfolio_id = req.portfolio_id or req.plan_name
        capital = req.capital
        as_of = req.as_of or datetime.now()
        bar_freq = req.bar_freq
        notify_channels = []

    instant = DecisionInstant(as_of=as_of, bar_freq=bar_freq)

    task_id = task_manager.create_task(
        TaskType.LIVE_REBALANCE,
        params={"plan_name": plan_name, "signal_source": signal_source},
        title=f"组合调仓: {plan_name}",
        entity_type="rebalance",
        entity_name=plan_name,
    )

    def execute(on_progress: Callable[[float, str], None] | None = None) -> dict:
        """后台任务体：调用 run_rebalance_decision 执行一次组合调仓决策。

        先幂等 import rules 以确保 etf_momentum 等信号源完成注册，再以本路由解析好的
        计划/内联参数发起调仓；由 task_manager 在后台线程调用。

        Args:
            on_progress: 可选进度回调，签名 ``(进度比例, 阶段描述)``，由任务框架注入用于上报进度。

        Returns:
            run_rebalance_decision 的执行结果 dict。
        """
        # 确保 rules 信号源已注册（import 幂等）。
        from .. import rules  # noqa: F401  确保 etf_momentum 等信号源已注册
        return run_rebalance_decision(
            plan_name=plan_name,
            signal_source=signal_source,
            signal_params=signal_params,
            strategy_params=strategy_params,
            portfolio_id=portfolio_id,
            instant=instant,
            capital=capital,
            rebalance_store=_rebalance_store,
            position_book=_position_book,
            risk_manager=PortfolioRiskManager(_runtime_state, _portfolio_risk_config),
            notifier=build_notifier(notify_channels) if notify_channels else LogNotifier(),
            lab=_get_lab(),
            on_progress=on_progress,
            trigger_source="manual",
        )

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "组合调仓任务已启动"}
