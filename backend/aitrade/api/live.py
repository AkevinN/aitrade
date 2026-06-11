"""
交易操作台（Trading Console）API 路由 `/api/live/*`。

把既有实盘原语（`predict_cnn_signals` → `SignalService` → `RiskManager` →
`Notifier` → `DecisionStore`）经 `LiveSignalOrchestrator` 串联后暴露为 HTTP 接口：

- `POST   /api/live/decision`：触发一次今日决策（异步任务，返回 task_id）。
- `GET    /api/live/decisions`：列出已持久化决策的 signal_id 集合。
- `GET    /api/live/decisions/{signal_id}`：单条决策详情。
- `DELETE /api/live/decisions/{signal_id}`：归档式删除决策 + trace（解除幂等占位）。

安全边界（v1）：本路由**不 import 也不调用任何券商网关 / 下单接口**，产出仅限
Decision 落盘 + Notifier 提醒（Property 7）。当前后端无鉴权（TD-015），真实下单能力
依赖鉴权与 kill-switch UI 前置条件，超出本特性范围。
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Optional

from fastapi import APIRouter, Body, HTTPException

from ..alpha.lab_utils import normalize_vt_symbol
from ..config import (
    CNN_MODEL_PATH,
    LIVE_DECISION_PATH,
    LIVE_RUNTIME_STATE_PATH,
    TRADING_PLAN_PATH,
)
from ..live.decision import Decision, DecisionStore
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
from ..live.runtime_state import RuntimeStateStore
from ..live.trading_plan import (
    TradingPlan,
    TradingPlanStore,
    effective_trigger_times,
)
from ..models import TaskType
from ..models.live import (
    LiveDecisionRequest,
    PortfolioSnapshotRequest,
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
_scheduler: Optional[PlanScheduler] = None


def _decision_to_dict(decision: Decision) -> dict[str, Any]:
    """把 Decision 数据对象转为可序列化 dict（用于 API 响应）。"""
    return asdict(decision)


def _validate_bar_freq_against_model(model: str, bar_freq: str) -> None:
    """间隔锁定（Req 2）：bar_freq 必须 == 模型训练间隔对应的 bar_freq。

    模型在固定间隔的 bar 分布上训练，喂别的周期是分布外输入——与回测端
    「回测周期必须 = 训练周期」是同一条红线在实时端的延伸。

    - checkpoint 可读 → 不一致返回 400；
    - 模型缺失/不可读 → 日内 bar_freq 返回 404（日内必须能锁定间隔）；
      "1d" 放过（保持既有宽松行为：允许先建计划，触发时再报错）。

    路径源与本路由其它模型存在性检查一致（模块级 `CNN_MODEL_PATH`，测试可整体替换）。
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
    """触发今日决策：校验必填字段(400)/模型存在(404) → 创建异步任务 → 返回 task_id。"""
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

    def execute(on_progress: Optional[Callable[[float, str], None]] = None) -> dict:
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
        )

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "今日决策任务已启动"}


@router.get(
    "/decisions",
    description="列出 DecisionStore 中已持久化决策的 signal_id 集合。",
)
async def list_decisions() -> dict:
    """返回所有已持久化决策的标识符集合（Req 4.1）。"""
    return {"signal_ids": _store.list_ids()}


@router.get(
    "/decisions/{signal_id}",
    description="按 signal_id 返回单条决策详情，不存在则 404。",
)
async def get_decision(signal_id: str) -> dict:
    """返回指定 signal_id 的完整决策；不存在则 404（Req 4.2 / 4.3）。"""
    decision = _store.get(signal_id)
    if decision is None:
        raise HTTPException(404, f"决策不存在: {signal_id}")
    return _decision_to_dict(decision)


@router.get(
    "/decisions/{signal_id}/trace",
    description="按 signal_id 返回该决策的完整 Decision_Trace（六段决策过程档案），不存在则 404。",
)
async def get_decision_trace(signal_id: str) -> dict:
    """返回指定 signal_id 的完整决策过程档案；不存在则 404（Req 8.4 / 8.5）。"""
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
    """归档决策与 trace（整体处理，避免「新决策配旧档案」错位）；决策不存在则 404。"""
    archived = _store.archive(signal_id)
    if archived is None:
        raise HTTPException(404, f"决策不存在: {signal_id}")
    trace_archived = _trace_store.archive(signal_id)
    return {
        "signal_id": signal_id,
        "deleted": True,
        "trace_archived": trace_archived is not None,
    }


# =============================================================================
# 交易计划自动化（Trading Plan Automation）：计划 CRUD + 按计划触发 + 调度状态
# =============================================================================


def _trigger_plan(
    plan: TradingPlan,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """用计划配置触发一次今日决策（手动与自动调度共用同一编排）。

    复用既有 `run_live_decision`（CNN 推理 → 取价 → SignalService），通知通道由
    `build_notifier(plan.notify_channels)` 在运行时按环境变量装配（凭证不入计划）。
    不调用任何券商网关 / 下单接口（Property 12）。
    """
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
    )


def _request_to_plan(req: TradingPlanRequest, *, plan_id: str, created_at: str | None = None) -> TradingPlan:
    """把请求体映射为持久化 TradingPlan（notify_channels 仅通道名，无凭证）。"""
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
    )
    if created_at is not None:
        plan.created_at = created_at
    return plan


def _plan_summary(plan: TradingPlan) -> TradingPlanSummary:
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
    )


@router.post("/plans", description="创建交易计划，返回完整内容（notify_channels 仅通道名，无凭证）。")
async def create_plan(req: TradingPlanRequest) -> dict:
    """创建交易计划（Req 2.1）。"""
    _validate_bar_freq_against_model(req.model, req.bar_freq)  # 间隔锁定
    plan = _request_to_plan(req, plan_id=TradingPlan.new_id())
    _plan_store.save(plan)
    return asdict(plan)


@router.get("/plans", description="列出所有交易计划摘要（含启用状态与最近触发日）。")
async def list_plans() -> list[dict]:
    """计划列表摘要（Req 2.2）。"""
    return [_plan_summary(p).model_dump() for p in _plan_store.list_all()]


@router.get("/plans/{plan_id}", description="按 plan_id 返回计划完整内容，不存在则 404。")
async def get_plan(plan_id: str) -> dict:
    """计划详情（Req 2.3 / 2.7）。"""
    plan = _plan_store.get(plan_id)
    if plan is None:
        raise HTTPException(404, f"交易计划不存在: {plan_id}")
    return asdict(plan)


@router.put("/plans/{plan_id}", description="按 plan_id 更新计划（保持 plan_id 与 created_at 不变）。")
async def update_plan(plan_id: str, req: TradingPlanRequest) -> dict:
    """更新计划（Req 2.4 / 2.7）。"""
    existing = _plan_store.get(plan_id)
    if existing is None:
        raise HTTPException(404, f"交易计划不存在: {plan_id}")
    _validate_bar_freq_against_model(req.model, req.bar_freq)  # 间隔锁定
    plan = _request_to_plan(req, plan_id=plan_id, created_at=existing.created_at)
    _plan_store.save(plan)  # updated_at 由 dataclass 默认工厂在重建时刷新
    return asdict(plan)


@router.delete("/plans/{plan_id}", description="按 plan_id 删除计划，不存在则 404。")
async def delete_plan(plan_id: str) -> dict:
    """删除计划（Req 2.5 / 2.7）。"""
    if not _plan_store.delete(plan_id):
        raise HTTPException(404, f"交易计划不存在: {plan_id}")
    return {"plan_id": plan_id, "deleted": True}


@router.patch("/plans/{plan_id}/enabled", description="启用/停用计划。")
async def toggle_plan(plan_id: str, enabled: bool = Body(..., embed=True)) -> dict:
    """切换启用状态（Req 2.6 / 2.7）。"""
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
    """按计划手动触发（Req 3.1 / 3.4）。"""
    plan = _plan_store.get(plan_id)
    if plan is None:
        raise HTTPException(404, f"交易计划不存在: {plan_id}")

    task_id = task_manager.create_task(
        TaskType.LIVE_DECISION,
        title=f"按计划触发: {plan.name}",
        entity_type="trading_plan",
        entity_name=plan.name,
    )

    def execute(on_progress: Optional[Callable[[float, str], None]] = None) -> dict:
        return _trigger_plan(plan, on_progress=on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "按计划触发任务已启动"}


@router.get("/scheduler/status", description="返回进程内调度器运行状态与各计划最近触发日。")
async def scheduler_status() -> dict:
    """调度器状态（Req 8.5）。"""
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


def build_plan_scheduler(*, tick_seconds: float) -> PlanScheduler:
    """装配 PlanScheduler（注入计划存储、运行时状态与 _trigger_plan）。"""
    return PlanScheduler(
        store=_plan_store,
        state=_runtime_state,
        trigger_fn=lambda plan: _trigger_plan(plan),
        tick_seconds=tick_seconds,
    )


def register_scheduler(scheduler: Optional[PlanScheduler]) -> None:
    """由 main.py lifespan 注册调度器单例，供状态端点读取。"""
    global _scheduler
    _scheduler = scheduler
