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
from datetime import date, datetime, time
from typing import Any, Callable, Optional

from .calendar import TradingCalendar
from .decision_instant import INTRADAY_BAR_FREQS
from .degradation import decide_trading
from .runtime_state import RuntimeStateStore
from .scheduler import due_bar_slot, due_slots
from .single_instance import SingleInstanceLock
from .trading_plan import TradingPlan, TradingPlanStore, effective_trigger_times

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
    ) -> None:
        self._store = store
        self._state = state
        self._trigger_fn = trigger_fn
        self._calendar = calendar or TradingCalendar()
        self._tick = tick_seconds
        self._now = now_fn
        self._health = health_fn or (lambda: (True, ""))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock: Optional[SingleInstanceLock] = None

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
        """日频路径：用户配置的唤醒时刻 + due_slots（行为与历史逐位一致）。"""
        times = [self._parse_time(t) for t in effective_trigger_times(plan)]
        slots = due_slots(now, times, self._calendar, done)
        if not slots:
            return
        if not self._trading_allowed(plan, now):
            return
        # 逐到期 slot 触发；决策层幂等使同日多 slot 收敛为当日一次决策（Req 3.1）
        for slot in slots:
            logger.info("[plan %s] 触发今日决策 @ %s", plan.plan_id, slot)
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
            return
        expected = datetime.combine(today, self._parse_time(slot))
        logger.info("[plan %s] 监控触发：%s bar @ %s", plan.plan_id, plan.bar_freq, slot)
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

    def tick_once(self) -> None:
        """单次 Tick：遍历启用计划，按 bar_freq 分派日频/监控路径。异常逐计划隔离（Req 7.3）。"""
        now = self._now()
        today = now.date()
        for plan in self._store.list_all():
            if not plan.enabled:
                continue  # 停用计划跳过（Req 4.5）
            try:
                done = self._triggered_slots(plan.plan_id, today)
                if _LEGACY_ALL_DONE in done:
                    continue  # 旧状态：当日整日已触发，跳过（Req 4.2）
                if plan.bar_freq in INTRADAY_BAR_FREQS:
                    self._tick_monitor_plan(plan, now, today, done)
                else:
                    self._tick_daily_plan(plan, now, today, done)
            except Exception as exc:  # noqa: BLE001  单计划失败隔离（Req 7.3）
                logger.warning(
                    "[plan %s] 调度触发异常（已隔离）：%s", plan.plan_id, exc
                )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick_once()
            except Exception as exc:  # noqa: BLE001  Tick 级兜底，线程不退出
                logger.error("调度 Tick 异常（已兜底）：%s", exc)
            self._stop.wait(self._tick)

    def start(self, lock: Optional[SingleInstanceLock] = None) -> bool:
        """启动调度线程；若单实例锁被占用则不启动并返回 False（Req 7.2）。"""
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
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._tick + 1)
            self._thread = None
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def enabled_plan_count(self) -> int:
        return sum(1 for p in self._store.list_all() if p.enabled)
