"""
组合调仓编排器（Rebalance Orchestrator）。

与单标的 orchestrator.py 平行——不改其单标的逻辑，仅在组合层新增。

``run_rebalance_decision`` 是核心函数，串联**既有原语**完成一次「规则策略调仓决策」：

    build_signal_source -> provider.predict -> 选 Decision_Bar
    -> PortfolioRiskManager.evaluate（组合风控）
    -> TopK 目标组合 -> diff 持仓 -> RebalanceDecision 落盘
    -> format_rebalance_message -> Notifier.send

本模块**不重复实现**任何决策 / 风控逻辑，也**不 import 也不调用任何券商网关 /
下单接口**（Property 7：无券商下单路径）。产出仅限 RebalanceDecision 落盘 + Notifier 提醒。
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from ..backtest.registry import build_signal_source
from .decision_instant import (
    DecisionInstant,
    make_signal_id,
    select_decision_bar,
    session_close,
)
from .notifier import Notifier
from .portfolio_risk import PortfolioRiskManager
from .position_book import PositionBook
from .rebalance_decision import RebalanceDecision, RebalanceItem, RebalanceStore

# 可观测性 logger（与 orchestrator.py 约定一致）。
logger = logging.getLogger("aitrade.live.rebalance")

# 信号回看窗口（日历日）：覆盖 as_of 之前至少一根已收盘 1d bar（含长假裕量）。
_DECISION_WINDOW_DAYS = 16


# =============================================================================
# 纯函数：格式化调仓通知（单独可测）
# =============================================================================


def format_rebalance_message(decision: RebalanceDecision) -> tuple[str, str]:
    """格式化调仓决策为人读清单消息（纯函数，无副作用）。

    把一条 RebalanceDecision 渲染成可直接推送给用户的标题与正文，供调用方
    传入 notifier.send。

    Args:
        decision: 待格式化的调仓决策。本函数读取其以下字段：
            - scheme：方案名，渲染标题时去除 "rule:" 前缀。
            - decision_bar_dt：决策 bar 时刻字符串（如 "2026-06-09T15:00:00" 或
              "2026-06-09"），仅取前 10 位日期部分进标题。
            - items：调仓明细列表，按 action 分为卖出/买入两段，逐条渲染
              "{动作} {vt_symbol} {volume}股 @≈{price}"；price 为 None 时省略价格。
            - risk_summary：风控记录列表，据其中 check/passed/detail 渲染熔断、
              趋势弱、取价失败等风控提示；当处于熔断（circuit/drawdown 未通过）
              且 items 为空时，输出"持仓维持现状"的人工处置指引。

    Returns:
        (title, message) 二元组：title 为单行标题，message 为多行正文
        （以 "\\n" 连接）。两者均为非空字符串，message 末行固定提示人工执行。

    消息格式：
    - 标题：含 scheme（去 "rule:" 前缀）与 decision_bar_dt 日期部分。
    - 卖出段：每行 "卖出 {sym} {volume}股 @≈{price}"（price=None 时省略价格）。
    - 买入段：同上。
    - 风控提示段：熔断或趋势弱时显著标注。
    - 尾行：提示人工执行。
    """
    scheme_display = decision.scheme.removeprefix("rule:")
    # decision_bar_dt 取日期部分（可能为 "2026-06-09T15:00:00" 或 "2026-06-09"）
    bar_dt_display = decision.decision_bar_dt[:10]
    title = f"【调仓决策】{scheme_display} @ {bar_dt_display}"

    lines: list[str] = []

    # 判断是否处于熔断状态（risk_summary 含 circuit passed=False）。
    is_circuit_broken = any(
        not r.get("passed", True) and r.get("check", "") in ("circuit", "drawdown")
        for r in decision.risk_summary
    )

    # 熔断时：items 为空（持仓不动），发出显著人工处置指引。
    if is_circuit_broken and not decision.items:
        lines.append(
            "⚠️ 组合熔断中：已暂停全部买入建议，持仓维持现状，"
            "请人工评估是否减仓，处置后可在操作台复位熔断"
        )
    else:
        # 卖出段
        sells = [item for item in decision.items if item.action == "sell"]
        if sells:
            lines.append("── 卖出 ──")
            for item in sells:
                price_str = f" @≈{item.price:.2f}" if item.price is not None else ""
                lines.append(f"卖出 {item.vt_symbol} {item.volume}股{price_str}")

        # 买入段
        buys = [item for item in decision.items if item.action == "buy"]
        if buys:
            lines.append("── 买入 ──")
            for item in buys:
                price_str = f" @≈{item.price:.2f}" if item.price is not None else ""
                lines.append(f"买入 {item.vt_symbol} {item.volume}股{price_str}")

        # 无调仓时的特殊提示
        if not sells and not buys:
            lines.append("持仓已与目标一致，无需调仓")

    # 风控提示段（熔断/趋势弱/取价失败时显著标注）
    risk_alerts: list[str] = []
    for record in decision.risk_summary:
        check = record.get("check", "")
        passed = record.get("passed", True)
        detail = record.get("detail", "")
        if not passed and check in ("circuit", "drawdown"):
            # 熔断且无 items 时已在上方单独输出，此处仍追加到风控提示段供完整展示。
            risk_alerts.append(f"⚠️ 熔断警告：{detail}")
        elif not passed and check == "trend":
            risk_alerts.append(f"⚠️ 趋势弱：{detail}")
        elif not passed and check == "pricing":
            risk_alerts.append(f"⚠️ 取价失败：{detail}")

    if risk_alerts:
        lines.append("── 风控提示 ──")
        lines.extend(risk_alerts)

    lines.append("请人工执行后在操作台确认")

    message = "\n".join(lines)
    return title, message


# =============================================================================
# 核心编排函数
# =============================================================================


def run_rebalance_decision(
    *,
    plan_name: str,
    signal_source: str,
    signal_params: dict,
    strategy_params: dict,
    portfolio_id: str,
    instant: DecisionInstant,
    capital: float,
    rebalance_store: RebalanceStore,
    position_book: PositionBook,
    risk_manager: PortfolioRiskManager,
    notifier: Notifier,
    lab: Any | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    trigger_source: str = "manual",
) -> dict[str, Any]:
    """编排一次组合调仓决策（规则策略，多标的）。

    参数：
        plan_name       — 计划名称，用于 scheme 命名空间（"rule:{plan_name}"）。
        signal_source   — 注册表信号源名（如 "etf_momentum"）。
        signal_params   — 透传给信号源的参数 dict。
        strategy_params — 组合选股参数：top_k（默认 5）、min_volume（默认 100，A 股最小手数）。
        portfolio_id    — 持仓账本 ID（PositionBook 的键）。
        instant         — 决策时刻（DecisionInstant）。
        capital         — 组合目标市值（用于计算目标仓位）。
                          **v1 近似语义**：此处直接用传入的 capital 作为组合净值，
                          不读账本现金/持仓市值，因 PositionBook v1 不追踪现金与市值。
                          风控的 drawdown 在 v1 实际等于恒 0（capital 恒为峰值近似）——
                          **这是 Phase 5 接真实市值前的占位**，但熔断状态（人工触发/复位）
                          与趋势闸门（基准指数信号）立即生效，不受此近似影响。
        rebalance_store — 调仓决策持久化（RebalanceStore）。
        position_book   — 持仓账本（PositionBook）。
        risk_manager    — 组合级风控（PortfolioRiskManager）。
        notifier        — 通知通道。
        lab             — AlphaLab 实例（可选，用于趋势闸门基准数据；None 时 fail-open）。
        on_progress     — 进度回调 (float 0~100, str 消息)。
        trigger_source  — 触发来源标记，默认 "manual"（人工在操作台手动触发）；
                          定时任务/调度器触发时传 "scheduled"。仅作为审计标签
                          原样写入 RebalanceDecision.trigger_source，不影响决策逻辑。

    返回：
        {
          "decision": asdict(RebalanceDecision) | None（幂等命中时为旧决策 dict）,
          "idempotent_hit": bool,
          "risk": list[{check, passed, detail}],
          "skipped_reason": str | None,
        }

    不调用任何券商网关 / 下单接口（Property 7）。

    与单标的 orchestrator.py 的差异：
    - 不产出 DecisionTrace（v1 调仓决策以 risk_summary+items 自解释，无六段 trace 结构需求）。
    - 信号来源为注册表信号源（build_signal_source），非 predict_cnn_signals。
    - 幂等键不含 vt_symbol（组合级，scheme 命名空间 "rule:" 前缀隔离单标的决策）。
    """
    scheme = f"rule:{plan_name}"
    as_of_date = instant.as_of.date()

    # Wave 2c: 入口计时（elapsed_ms）。
    _t0 = time.monotonic()

    # ------------------------------------------------------------------
    # 步骤 1：幂等前置（两段式）
    # 第一段：用 as_of 推导初步 bar 时刻估算 signal_id，在任何重活之前判定。
    # ------------------------------------------------------------------
    _prelim_bar_dt = session_close(as_of_date, instant.bar_freq)
    signal_id = make_signal_id(_prelim_bar_dt, instant.bar_freq, scheme)

    if on_progress:
        on_progress(5, "幂等前置检查...")

    existed = rebalance_store.get(signal_id) is not None
    if existed:
        # 命中：直接返回旧决策，不做任何重活。
        old_decision = rebalance_store.get(signal_id)
        logger.info("幂等命中（初步 signal_id=%s），直接返回旧决策", signal_id)
        return {
            "decision": asdict(old_decision) if old_decision else None,
            "idempotent_hit": True,
            "risk": old_decision.risk_summary if old_decision else [],
            "skipped_reason": None,
        }

    # ------------------------------------------------------------------
    # 步骤 2：信号生产
    # 注入 "_lab" 参数供支持 lab 的信号源读取行情（如 etf_momentum）。
    # ------------------------------------------------------------------
    if on_progress:
        on_progress(10, f"调用信号源 {signal_source}...")

    # rules 包需在调用前已注册——调用方（api/live.py 或测试）负责装配，
    # rebalance.py 本身不 import rules，最小侵入。
    provider = build_signal_source(signal_source, {**signal_params, "_lab": lab})

    start_dt = as_of_date - timedelta(days=_DECISION_WINDOW_DAYS)
    signal_df = provider.predict(start=start_dt, end=as_of_date)

    if on_progress:
        on_progress(30, "选取 Decision_Bar...")

    # 选 Decision_Bar（close_time <= as_of 的最后一根，无前视红线）。
    bar_row = select_decision_bar(signal_df, instant)
    if bar_row is None or bar_row.is_empty():
        # 无有效信号行：不落盘，通知后返回。
        logger.warning("信号源 %s 在 %s 之前无已收盘 bar，返回 skipped", signal_source, instant.as_of)
        notifier.send(
            f"【调仓跳过】{plan_name}",
            f"当日（{as_of_date.isoformat()}）无有效信号（防御性空仓或数据缺失）",
        )
        return {
            "decision": None,
            "idempotent_hit": False,
            "risk": [],
            "skipped_reason": "当日无有效信号（防御性空仓或数据缺失）",
        }

    # 取 Decision_Bar 实际时刻，重算 signal_id（覆盖初步推导值，处理回退至上一交易日的情形）。
    # 这是幂等两段式的第二段校正（仿 orchestrator.py:249 模式）。
    decision_bar_dt: datetime = bar_row["datetime"][0]
    signal_id = make_signal_id(decision_bar_dt, instant.bar_freq, scheme)

    # 二次幂等检查（decision_bar 确定后，校正后的 signal_id 是否已落盘）。
    existed_after_correction = rebalance_store.get(signal_id) is not None
    if existed_after_correction:
        old_decision = rebalance_store.get(signal_id)
        logger.info("幂等命中（校正后 signal_id=%s），直接返回旧决策", signal_id)
        return {
            "decision": asdict(old_decision) if old_decision else None,
            "idempotent_hit": True,
            "risk": old_decision.risk_summary if old_decision else [],
            "skipped_reason": None,
            # Wave 2c：幂等命中返回旧决策原样，notify_ok 为首次运行的实测值
        }

    # ------------------------------------------------------------------
    # 步骤 3：当日信号排行（signal 降序）
    # ------------------------------------------------------------------
    # signal_df 含 [datetime, vt_symbol, signal]；取已选 bar 当日所有标的信号。
    bar_signals = signal_df.filter(
        signal_df["datetime"] == decision_bar_dt
    ).sort("signal", descending=True)

    # ------------------------------------------------------------------
    # 步骤 4：组合风控评估
    # v1 近似：用 capital 作为组合净值（docstring 已说明 drawdown≈0 的占位语义）。
    # ------------------------------------------------------------------
    if on_progress:
        on_progress(50, "组合风控评估...")

    verdict = risk_manager.evaluate(
        portfolio_id,
        portfolio_value=capital,
        as_of=as_of_date,
    )

    # ------------------------------------------------------------------
    # 步骤 5：构建目标组合
    # ------------------------------------------------------------------
    if on_progress:
        on_progress(65, "构建目标组合...")

    top_k: int = int(strategy_params.get("top_k", 5))
    min_volume: int = int(strategy_params.get("min_volume", 100))

    target_portfolio: dict[str, int] = {}

    # 收集取价失败的标的（供 risk_summary 可见化）。
    skipped_symbols: list[str] = []

    if verdict.allow_buy:
        # 风控通过：按信号降序取 top_k 标的，计算目标股数。
        buy_factor: float = verdict.buy_factor
        per_symbol_value = capital * buy_factor / max(top_k, 1)

        rows = bar_signals.head(top_k)
        for row in rows.iter_rows(named=True):
            sym: str = row["vt_symbol"]
            # 参考价：从 signal_df 取同一标的同一 bar 的 close（若存在）；否则从 lab 读。
            price: float | None = _get_price(sym, signal_df, decision_bar_dt, lab, instant)
            if price is None or price <= 0:
                logger.warning("标的 %s 无有效价格，跳过，不纳入目标组合", sym)
                skipped_symbols.append(sym)
                continue
            # A 股一手 100 股：floor(目标市值 / 价格 / 100) * 100
            lots = math.floor(per_symbol_value / price / 100)
            volume = lots * 100
            if volume < min_volume:
                logger.warning(
                    "标的 %s 目标股数 %d < min_volume %d，跳过", sym, volume, min_volume
                )
                continue
            target_portfolio[sym] = volume
    else:
        # 熔断：目标组合 = 当前持仓（保持不动，不自动产生清仓建议）。
        # diff 将自然为空，走"空 items 决策占幂等位"路径。
        # 通知文案由 format_rebalance_message 负责显著标注熔断状态与人工处置指引。
        current_for_circuit = position_book.load(portfolio_id)
        target_portfolio = dict(current_for_circuit.positions)
        logger.info(
            "组合风控 allow_buy=False（熔断），目标组合=当前持仓（保持不动，不自动产生清仓建议）"
        )

    # ------------------------------------------------------------------
    # 步骤 6：diff 当前持仓与目标组合
    # ------------------------------------------------------------------
    if on_progress:
        on_progress(75, "计算持仓 diff...")

    current_state = position_book.load(portfolio_id)
    current_positions: dict[str, int] = dict(current_state.positions)

    items: list[RebalanceItem] = []
    _all_symbols = set(current_positions.keys()) | set(target_portfolio.keys())

    for sym in sorted(_all_symbols):
        current_vol = current_positions.get(sym, 0)
        target_vol = target_portfolio.get(sym, 0)
        if target_vol < current_vol:
            # 卖出（包含完全清仓 target_vol=0 的情形）。
            sell_vol = current_vol - target_vol
            # I-3 整手门槛：sell 差额 < 100 股且 target>0（非全部清出）时跳过，
            # 避免细碎 sell item（A 股散股操作无意义）。
            # target==0（全仓清出）允许任意股数，A 股零股可一次性卖出。
            if sell_vol < 100 and target_vol > 0:
                logger.warning(
                    "标的 %s sell 差额 %d 股 < 100 且 target>0，跳过细碎卖出", sym, sell_vol
                )
                continue
            # 参考价
            price = _get_price(sym, signal_df, decision_bar_dt, lab, instant)
            items.append(RebalanceItem(
                vt_symbol=sym,
                action="sell",
                volume=sell_vol,
                price=price,
            ))
        elif target_vol > current_vol:
            # 买入
            buy_vol = target_vol - current_vol
            price = _get_price(sym, signal_df, decision_bar_dt, lab, instant)
            items.append(RebalanceItem(
                vt_symbol=sym,
                action="buy",
                volume=buy_vol,
                price=price,
            ))
        # target_vol == current_vol：无需操作，不产生 item。

    # ------------------------------------------------------------------
    # 步骤 7：落盘（幂等 save_if_absent）
    # ------------------------------------------------------------------
    if on_progress:
        on_progress(85, "落盘调仓决策...")

    # I-1c：取价失败标的写入 risk_summary pricing record，用户/通知可见。
    enriched_risk_records = list(verdict.records)
    if skipped_symbols:
        enriched_risk_records.append({
            "check": "pricing",
            "passed": False,
            "detail": f"以下标的无法取价被跳过: {', '.join(skipped_symbols)}",
        })

    # Wave 2c: elapsed_ms 在落盘前计算（save_if_absent 之前）。
    elapsed_ms = int((time.monotonic() - _t0) * 1000)

    decision = RebalanceDecision(
        signal_id=signal_id,
        decision_bar_dt=decision_bar_dt.isoformat(timespec="seconds"),
        as_of=instant.as_of.isoformat(timespec="seconds"),
        bar_freq=instant.bar_freq,
        scheme=scheme,
        portfolio_id=portfolio_id,
        items=items,
        target_portfolio=target_portfolio,
        risk_summary=enriched_risk_records,
        status="proposed",
        trigger_source=trigger_source,
        elapsed_ms=elapsed_ms,
    )

    _saved, decision = rebalance_store.save_if_absent(decision)

    # ------------------------------------------------------------------
    # 步骤 8：通知
    # ------------------------------------------------------------------
    if on_progress:
        on_progress(95, "发送通知...")

    title, message = format_rebalance_message(decision)
    notify_ok: bool = notifier.send(title, message)
    if not notify_ok:
        logger.warning(
            "调仓通知发送失败（send 返回 False），计划名=%s，通道数量未知", plan_name
        )

    # Wave 2c: 回写 notify_ok 到 decision（save_if_absent 返回的是已存在或新建的 decision）。
    # 直接更新内存对象；若 _saved=True 表示本次新写入，需要追加 notify_ok 字段到磁盘。
    decision.notify_ok = notify_ok
    if _saved:
        rebalance_store.save(decision)

    if on_progress:
        on_progress(100, "调仓决策完成")

    return {
        "decision": asdict(decision),
        "idempotent_hit": False,
        "risk": enriched_risk_records,
        "skipped_reason": None,
    }


# =============================================================================
# 内部辅助：取标的参考价
# =============================================================================


def _get_price(
    vt_symbol: str,
    signal_df: Any,
    decision_bar_dt: datetime,
    lab: Any | None,
    instant: DecisionInstant,
) -> float | None:
    """从信号 DataFrame 或 AlphaLab 读取标的参考价（close）。

    用于调仓时估算下单参考价；取价失败不抛错，由调用方据返回的 None 决定跳过该标的。

    Args:
        vt_symbol: 目标标的合约代码，如 "510300.SSE"；用于在 signal_df / 行情表中筛选行。
        signal_df: 信号源产出的 polars DataFrame，至少含 [datetime, vt_symbol] 列；
            若额外含 "close" 列则作为首选价源（路径 1）。
        decision_bar_dt: 已选定的决策 bar 时刻，用于在 signal_df 中定位该标的当日行。
        lab: AlphaLab 实例；为 None 时跳过行情回退（路径 2 不可用），仅依赖 signal_df。
        instant: 决策时刻（DecisionInstant），提供 as_of 截断点与 bar_freq，
            供路径 2 按 as-of 选取无前视的 bar（截断窗口为 as_of 前 _DECISION_WINDOW_DAYS 日）。

    Returns:
        参考价 close（float）；当 signal_df 无 "close" 列或无匹配行、且 lab 为 None
        或行情取价失败时返回 None，表示无法取价（调用方应跳过该标的）。

    取价优先级：
    1. signal_df 若含 "close" 列，取 decision_bar_dt 当日该标的 close。
    2. lab 不为 None 时，从 AlphaLab 加载 bar 行情取 close（as-of 截断，无前视）。
    3. 均无则返回 None（调用方跳过该标的）。
    """
    import polars as pl

    # 路径 1：信号 DataFrame 含 "close" 列（部分信号源会附带价格）。
    if "close" in signal_df.columns:
        rows = signal_df.filter(
            (pl.col("vt_symbol") == vt_symbol) & (pl.col("datetime") == decision_bar_dt)
        )
        if not rows.is_empty():
            val = rows["close"][0]
            if val is not None:
                return float(val)

    # 路径 2：通过 AlphaLab 读取。
    if lab is not None:
        try:
            from ..alpha import AlphaLab  # noqa: F401 — 类型检查；lab 已是实例
            from ..live.decision_instant import interval_of_bar_freq

            d = instant.as_of.date()
            start_dt = datetime.combine(d - timedelta(days=_DECISION_WINDOW_DAYS), datetime.min.time())
            end_dt = datetime.combine(d, datetime.max.time())
            interval = interval_of_bar_freq(instant.bar_freq)
            frame = lab.load_bar_frame(vt_symbol, interval, start_dt, end_dt)
            if frame is not None and not frame.is_empty() and "close" in frame.columns:
                bar = select_decision_bar(frame, instant)
                if bar is not None and not bar.is_empty():
                    val = bar["close"][0]
                    if val is not None:
                        return float(val)
        except Exception as exc:  # noqa: BLE001 — 取价失败不中断主流程
            logger.warning("AlphaLab 取价 %s 失败：%s", vt_symbol, exc)

    return None
