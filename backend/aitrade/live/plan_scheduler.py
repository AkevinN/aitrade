"""
进程内交易计划调度器（Trading Plan Automation / 盘中监控决策）。

随 FastAPI 应用生命周期启停；周期性遍历启用的交易计划，按计划的 `bar_freq` 走两种形态：

- **日频（`bar_freq == "1d"`）**：对计划的多个生效唤醒时刻（`effective_trigger_times`）
  复用 `scheduler.due_slots` 判定，到期时点（slot）经注入的 `trigger_fn` 触发。多时点为
  「同一交易日的多次触发尝试」；决策层幂等（`signal_id` 不含时点）使同日多 slot 收敛为
  当日一次决策、一次提醒，多时点价值是鲁棒性回退。
- **日内（监控模式，`bar_freq` 为分钟频）**：忽略 `trigger_times`，以 `scheduler.due_bar_slot`
  按 Bar_Grid（交易时段内 bar 收盘时刻网格）判定——每根 bar 收盘后触发一次决策；触发后
  仅当决策的实际 `decision_bar_dt` 已跟上该网格时刻才记 slot，否则（本地数据滞后）告警并
  在下一 tick 重试，**绝不静默用旧 bar 假装监控成功**。

两种形态共享同一去重状态：`RuntimeStateStore` 中记录计划当日已完成的时点 slot
（**按 slot 幂等** + 重启可恢复 + 跨日自动重置）。

复用既有原语：`scheduler.due_slots`/`due_bar_slot`、`TradingCalendar`、
`degradation.decide_trading`、`RuntimeStateStore`、`SingleInstanceLock`。本模块**不向任何
券商网关提交真实订单**——触发动作完全委托注入的 `trigger_fn`（API 层注入复用 `run_live_decision`）。

设计要点：
- 触发动作经 `trigger_fn(plan) -> dict` 注入（返回 `run_live_decision` 结果，监控模式据其
  `decision.decision_bar_dt` 判定推进），解耦调度判定与决策执行，便于确定性单测。
- 单计划触发异常逐计划隔离；Tick 级再兜底，保证调度线程不退出。
- 降级（decide_trading 判定暂停）时跳过且**不**记该 slot，使恢复正常后同日仍可触发。
- Last_Triggered_Map 值为 `{plan_id: {"date","slots"}}`，旧字符串值按「整日已触发」兼容读取；
  监控模式的 slot 即 bar 收盘时刻 "HH:MM"，状态形态零变化。
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Callable, Optional

from .calendar import TradingCalendar
from .decision_instant import INTRADAY_BAR_FREQS
from .degradation import decide_trading
from .runtime_state import RuntimeStateStore
from .scheduler import due_bar_slot, due_slots
from .single_instance import SingleInstanceLock
from .trading_plan import TradingPlan, TradingPlanStore, effective_trigger_times

if TYPE_CHECKING:
    from .scheduler_run_log import SchedulerRunLog

logger = logging.getLogger("aitrade.live.plan_scheduler")

# RuntimeStateStore 中持久化 Last_Triggered_Map 的键。
# 值形态（升级）：{plan_id: {"date": "YYYY-MM-DD", "slots": ["HH:MM", ...]}}，按时点 slot 去重。
# 兼容旧值：纯 "YYYY-MM-DD" 字符串解释为「该日整日已触发」。
_LAST_TRIGGERED_KEY = "plan_last_triggered"
# 旧状态哨兵：旧字符串值迁移读取时表示「整日已触发」，使升级当日不重复触发。
_LEGACY_ALL_DONE = "*"

# 单机研究环境：无独立行情新鲜度源，降级判定主要依赖 health_fn；
# last_data_time 传 now、放宽 staleness，使「健康」时不因数据时点误判暂停。
_STALENESS_SECONDS = 86400.0


def _schedule_matches_today(plan: TradingPlan, today: date, calendar: TradingCalendar) -> bool:
    """判断 today 是否满足计划的 trigger_schedule 调度粒度（周/月闸门，纯函数）。

    - "daily"         → 恒真（交易日判定由 due_slots 内部负责，闸门不重复判）。
    - "weekly_first"  → today 是本 ISO 周内第一个交易日：
                        从本周周一扫到 today 前一天，若已存在任何交易日则返回 False；
                        且 today 自身必须是交易日。
    - "monthly_first" → 同理从当月 1 日扫到 today 前一天，若已存在任何交易日则 False；
                        且 today 自身必须是交易日。

    已知局限：默认 TradingCalendar 以工作日-节假日近似判定交易日，月初多天连续
    节假日可能导致首个交易日判定偏差——需使用精确日历（trading_days 参数）消除。
    """
    if plan.trigger_schedule == "daily":
        return True
    if not calendar.is_trading_day(today):
        return False
    if plan.trigger_schedule == "weekly_first":
        # 从本周一扫到 today 前一天，若有任何交易日则今日非本周第一个交易日
        week_start = today - timedelta(days=today.weekday())  # ISO 周一
        d = week_start
        while d < today:
            if calendar.is_trading_day(d):
                return False
            d += timedelta(days=1)
        return True
    if plan.trigger_schedule == "monthly_first":
        # 从当月 1 日扫到 today 前一天，若有任何交易日则今日非本月第一个交易日
        month_start = today.replace(day=1)
        d = month_start
        while d < today:
            if calendar.is_trading_day(d):
                return False
            d += timedelta(days=1)
        return True
    # 未知值视为 daily（宽松降级）
    return True


def _normalize_last_triggered(raw: dict) -> dict[str, str]:
    """把 Last_Triggered_Map 归一化为 {plan_id: "YYYY-MM-DD"}，兼容新旧两种值形态。

    - 新值 `{"date","slots"}` → 取 `date`。
    - 旧值 `"YYYY-MM-DD"` 字符串 → 原样。
    供调度状态端点与计划摘要展示（`last_triggered` 仍为日期字符串，前端无需感知 slots）。
    """
    out: dict[str, str] = {}
    for plan_id, value in (raw or {}).items():
        if isinstance(value, str):
            out[plan_id] = value
        elif isinstance(value, dict) and value.get("date"):
            out[plan_id] = value["date"]
    return out


class PlanScheduler:
    """进程内后台调度器：周期遍历启用计划，到点经回调触发。"""

    def __init__(
        self,
        store: TradingPlanStore,
        state: RuntimeStateStore,
        trigger_fn: Callable[[TradingPlan], dict[str, Any]],
        *,
        calendar: Optional[TradingCalendar] = None,
        tick_seconds: float = 30.0,
        now_fn: Callable[[], datetime] = datetime.now,
        health_fn: Optional[Callable[[], tuple[bool, str]]] = None,
        run_log: SchedulerRunLog | None = None,
    ) -> None:
        """初始化 PlanScheduler。

        Args:
            store:        计划存储，用于 list_all()。
            state:        运行时状态存储，用于 Last_Triggered_Map 读写。
            trigger_fn:   触发回调 (plan) -> result dict；注入 run_live_decision 或 run_rebalance_decision。
            calendar:     交易日历，None 时使用默认工作日日历。
            tick_seconds: 调度轮询周期（秒），默认 30。
            now_fn:       时刻注入函数，默认 datetime.now（测试可注入固定时间）。
            health_fn:    系统健康检查函数 () -> (ok, reason)，默认始终健康。
            run_log:      调度运行日志，None 时跳过日志写入（best-effort）。
        """
        self._store = store
        self._state = state
        self._trigger_fn = trigger_fn
        self._calendar = calendar or TradingCalendar()
        self._tick = tick_seconds
        self._now = now_fn
        self._health = health_fn or (lambda: (True, ""))
        self._run_log: SchedulerRunLog | None = run_log
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock: Optional[SingleInstanceLock] = None

    # —— SchedulerRunLog 包装方法（None 时 no-op；调用包 try/except，R3.5）——

    def _log_skip(self, plan_id: str, reason: str, detail: str = "") -> None:
        """记录跳过事件，失败仅 WARNING 不传播（best-effort）。"""
        if self._run_log is None:
            return
        try:
            self._run_log.record_skip(plan_id, reason, detail)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[plan %s] _log_skip 失败（best-effort）: %s", plan_id, exc)

    def _log_trigger(self, plan_id: str, slot: str, detail: str = "") -> None:
        """记录触发事件，失败仅 WARNING 不传播（best-effort）。"""
        if self._run_log is None:
            return
        try:
            self._run_log.record_trigger(plan_id, slot, detail)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[plan %s] _log_trigger 失败（best-effort）: %s", plan_id, exc)

    def _log_error(self, plan_id: str, error: str) -> None:
        """记录触发错误事件，失败仅 WARNING 不传播（best-effort）。"""
        if self._run_log is None:
            return
        try:
            self._run_log.record_error(plan_id, error)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[plan %s] _log_error 失败（best-effort）: %s", plan_id, exc)

    # —— Last_Triggered_Map 读写（按时点 slot 去重；兼容旧字符串值）——
    def _triggered_slots(self, plan_id: str, today: date) -> set[str]:
        """返回该计划当日已触发的时点 slot 集合（"HH:MM"）。

        - 新值 `{"date","slots"}`：date == today 时返回 slots，否则空集（跨日重置，Req 2.5）。
        - 旧值 `"YYYY-MM-DD"` 字符串：等于今日时返回 `{"*"}`（整日已触发，Req 4.2），否则空集。
        - 缺失：空集。
        """
        raw = (self._state.get(_LAST_TRIGGERED_KEY, {}) or {}).get(plan_id)
        if raw is None:
            return set()
        if isinstance(raw, str):  # 旧值：整日粒度
            return {_LEGACY_ALL_DONE} if raw == today.isoformat() else set()
        if raw.get("date") != today.isoformat():
            return set()
        return set(raw.get("slots", []))

    def _mark_slot(self, plan_id: str, today: date, slot: str) -> None:
        """把某时点 slot 记入当日已触发集合并持久化（跨日则重置当日 slots）。"""
        mapping = self._state.get(_LAST_TRIGGERED_KEY, {}) or {}
        raw = mapping.get(plan_id)
        if isinstance(raw, dict) and raw.get("date") == today.isoformat():
            rec = {"date": today.isoformat(), "slots": list(raw.get("slots", []))}
        else:
            rec = {"date": today.isoformat(), "slots": []}
        if slot not in rec["slots"]:
            rec["slots"].append(slot)
        mapping[plan_id] = rec
        self._state.set(_LAST_TRIGGERED_KEY, mapping)

    def last_triggered_map(self) -> dict[str, str]:
        """返回 {plan_id: "YYYY-MM-DD"}（取最近触发日；兼容新旧值），供状态端点/摘要展示。"""
        return _normalize_last_triggered(self._state.get(_LAST_TRIGGERED_KEY, {}) or {})

    @staticmethod
    def _parse_time(decision_time: str) -> time:
        hh, mm = (int(x) for x in decision_time.split(":"))
        return time(hh, mm)

    def _trading_allowed(self, plan: TradingPlan, now: datetime) -> bool:
        """降级判定（Req 7.1）：暂停则跳过且不记 slot（恢复后同日仍可触发，Req 3.5）。"""
        healthy, hreason = self._health()
        ok, dreason = decide_trading(
            now=now,
            last_data_time=now,
            max_staleness_seconds=_STALENESS_SECONDS,
            healthy=healthy,
            reconcile_blocked=False,
        )
        if not ok:
            logger.warning("[plan %s] 降级跳过：%s / %s", plan.plan_id, hreason, dreason)
        return ok

    def _tick_daily_plan(self, plan: TradingPlan, now: datetime, today: date, done: set[str]) -> None:
        """日频路径：用户配置的唤醒时刻 + due_slots（行为与历史逐位一致）。

        Phase 3 M2：先经 trigger_schedule 闸门（周/月首日）判定；非触发日直接返回，
        slot 去重状态机无需修改——非触发日无记录，天然兼容（跨日重置语义不受影响）。
        """
        if not _schedule_matches_today(plan, today, self._calendar):
            self._log_skip(plan.plan_id, "schedule_gate")  # R3.1/R3.2
            return
        times = [self._parse_time(t) for t in effective_trigger_times(plan)]
        slots = due_slots(now, times, self._calendar, done)
        if not slots:
            # due_slots 为空有两种语义：
            # (a) 已到达时刻的 slot 都触发过 → passed 非空，应记 already_done
            # (b) 当日还没到任何 slot 时刻（now.time() < 所有 trigger_times）→ passed 为空，
            #     属正常等待，不记任何 skip 事件（I1 修复）
            if self._calendar.is_trading_day(today):
                passed = [t for t in times if now.time() >= t]
                if passed:
                    self._log_skip(plan.plan_id, "already_done")  # R3.1/R3.2
            return
        if not self._trading_allowed(plan, now):
            self._log_skip(plan.plan_id, "degraded")  # R3.1/R3.2
            return
        # 逐到期 slot 触发；决策层幂等使同日多 slot 收敛为当日一次决策（Req 3.1）
        for slot in slots:
            logger.info("[plan %s] 触发今日决策 @ %s", plan.plan_id, slot)
            self._log_trigger(plan.plan_id, slot)  # R3.3
            self._trigger_fn(plan)
            self._mark_slot(plan.plan_id, today, slot)  # Req 2.3

    def _tick_monitor_plan(self, plan: TradingPlan, now: datetime, today: date, done: set[str]) -> None:
        """监控模式（日内计划）：Bar_Grid 上最新已收盘 bar 未完成则触发。

        仅当决策的实际 `decision_bar_dt` 已跟上网格时刻才记 slot；本地数据滞后时不记、
        告警并在下一 tick 重试（绝不静默用旧 bar 假装监控成功）。
        """
        slot = due_bar_slot(now, plan.bar_freq, self._calendar, done)
        if slot is None:
            return
        if not self._trading_allowed(plan, now):
            self._log_skip(plan.plan_id, "degraded")  # R3.1/R3.2
            return
        expected = datetime.combine(today, self._parse_time(slot))
        logger.info("[plan %s] 监控触发：%s bar @ %s", plan.plan_id, plan.bar_freq, slot)
        self._log_trigger(plan.plan_id, slot)  # R3.3
        result = self._trigger_fn(plan)
        actual = datetime.fromisoformat(str(result["decision"]["decision_bar_dt"]))
        if actual >= expected:
            self._mark_slot(plan.plan_id, today, slot)
        else:
            logger.warning(
                "[plan %s] 行情滞后：decision_bar=%s < 网格时刻 %s，本 bar 暂不标记，下一 tick 重试",
                plan.plan_id,
                actual.isoformat(),
                expected.isoformat(),
            )
            self._log_skip(plan.plan_id, "data_lag", f"decision_bar={actual.isoformat()} < 网格={expected.isoformat()}")  # R3.1/R3.2

    def tick_once(self) -> None:
        """单次 Tick：遍历启用计划，按 bar_freq 分派日频/监控路径。异常逐计划隔离（Req 7.3）。"""
        now = self._now()
        today = now.date()
        for plan in self._store.list_all():
            if not plan.enabled:
                self._log_skip(plan.plan_id, "disabled")  # R3.1/R3.2 当日首次才落盘（去重）
                continue  # 停用计划跳过（Req 4.5）
            try:
                done = self._triggered_slots(plan.plan_id, today)
                if _LEGACY_ALL_DONE in done:
                    continue  # 旧状态：当日整日已触发，跳过（Req 4.2）
                # not_trading_day 探测（仅用于记录，不改变后续判定路径）：
                # 日频路径中 due_slots 内部会判定交易日；此处仅探测用于记录，判定仍由原逻辑负责。
                # 监控路径（INTRADAY）由 due_bar_slot 内部判定，探测逻辑不适用，故仅对日频记录。
                if plan.bar_freq not in INTRADAY_BAR_FREQS and not self._calendar.is_trading_day(today):
                    self._log_skip(plan.plan_id, "not_trading_day")
                if plan.bar_freq in INTRADAY_BAR_FREQS:
                    self._tick_monitor_plan(plan, now, today, done)
                else:
                    self._tick_daily_plan(plan, now, today, done)
            except Exception as exc:  # noqa: BLE001  单计划失败隔离（Req 7.3）
                logger.warning(
                    "[plan %s] 调度触发异常（已隔离）：%s", plan.plan_id, exc
                )
                self._log_error(plan.plan_id, str(exc))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick_once()
            except Exception as exc:  # noqa: BLE001  Tick 级兜底，线程不退出
                logger.error("调度 Tick 异常（已兜底）：%s", exc)
            self._stop.wait(self._tick)

    def start(self, lock: Optional[SingleInstanceLock] = None) -> bool:
        """启动调度线程；若单实例锁被占用则不启动并返回 False（Req 7.2）。

        Args:
            lock: 可选的单实例互斥锁；传入时尝试 acquire()，失败则放弃启动，返回 False。

        Returns:
            True 表示线程启动成功；False 表示锁被占用，未启动。
        """
        if lock is not None and not lock.acquire():
            logger.warning("单实例锁被占用，PlanScheduler 不启动")
            return False
        self._lock = lock
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="aitrade-plan-scheduler", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """停止调度线程并释放单实例锁（等待当前 tick 完成，超时 tick+1 秒）。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._tick + 1)
            self._thread = None
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def is_running(self) -> bool:
        """返回调度线程是否正在运行（线程存活 且 未收到停止信号）。"""
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def enabled_plan_count(self) -> int:
        """返回当前启用（enabled=True）的计划数量。"""
        return sum(1 for p in self._store.list_all() if p.enabled)
