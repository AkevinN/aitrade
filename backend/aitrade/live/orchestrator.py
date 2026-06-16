"""
交易操作台编排器（Trading Console glue — 新增粘合）。

`LiveSignalOrchestrator` 是一个薄编排层，串联**既有原语**完成一次「今日决策」：

    predict_cnn_signals（CNN 推理） -> 取决策日 bar 的 signal + close price
    -> RiskInspector 包住 RiskManager 注入 SignalService
    -> SignalService.run_for_date（信号→风控→提醒→落盘，幂等）
    -> 返回 {decision, risk_detail, idempotent_hit}

本模块**不重复实现**任何决策 / 风控逻辑，也**不 import 也不调用任何券商网关 /
下单接口**（Property 7：无券商下单路径）。产出仅限 Decision 落盘 + Notifier 提醒。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import polars as pl

from ..cnn.predictor import predict_cnn_signals
from .decision import Decision, DecisionStore
from .decision_instant import (
    DecisionInstant,
    interval_of_bar_freq,
    make_signal_id,
    select_decision_bar,
    session_close,
)
from .decision_trace import DecisionTraceStore, TraceBuilder
from .notifier import Notifier
from .risk import RiskConfig, RiskManager
from .risk_inspector import RiskInspector
from .signal_service import PortfolioSnapshot, SignalService

# 决策过程可观测性约定 logger（与 decision_trace 模块一致）：每行带 run_id 前缀。
logger = logging.getLogger("aitrade.live.orchestrator")

# 决策回看窗口（日历日）：as_of 当日若尚未收盘，则回退到 as_of 之前最近一根已收盘 bar。
# 取 ~16 自然日（含长假裕量），覆盖「as_of 之前至少一根已收盘 1d bar」。窗口加宽不改变
# 单根 bar 的信号值（推理按训练统计逐 bar 归一化、与区间无关），故日频决策内容与逐位等价不变。
_DECISION_WINDOW_DAYS = 16


def _decision_to_dict(decision: Decision) -> dict[str, Any]:
    """把 Decision 数据对象转为可序列化 dict（用于任务结果 / API 响应）。

    Args:
        decision: 已落盘 / 待返回的决策对象。

    Returns:
        decision 全字段的浅拷贝 dict（dataclasses.asdict 结果），可直接 JSON 序列化。
    """
    return asdict(decision)


def _load_close_price(vt_symbol: str, instant: DecisionInstant) -> tuple[float, str]:
    """从 AlphaLab 行情读 Decision_Bar 收盘价作为建议价位（as-of 截断，无前视）。

    `predict_cnn_signals` 返回的 DataFrame 仅含 `[datetime, vt_symbol, signal]`，不含价格，
    因此价位需单独从本地 bar 行情读取：取 `close_time <= as_of` 的最后一根已收盘 bar 的 `close`。
    首选周期与决策同频（`interval_of_bar_freq(bar_freq)`），无则回退到该标的其它可用周期。

    Args:
        vt_symbol: 目标标的合约代码，如 "000001.SZSE"；内部经 normalize_vt_symbol 规整。
        instant: 决策时刻，提供 as_of（价位截断上界，只取 close_time <= as_of 的 bar）
            与 bar_freq（决定首选取价周期 interval_of_bar_freq(bar_freq)）。

    Returns:
        `(close, interval_used)`：close 为命中 bar 的收盘价（元/股），interval_used 为
        实际命中的取价周期（首选同频，否则为回退命中的周期）供 trace 记录。

    Raises:
        ValueError: as_of 之前无可用的已收盘 Decision_Bar 行情（如 as_of 早于当日收盘、
            或该标的所有候选周期均无数据）时抛出，由 API 译为错误响应。
    """
    from ..alpha import AlphaLab
    from ..alpha.lab import normalize_vt_symbol
    from ..config import ALPHA_LAB_PATH

    canonical = normalize_vt_symbol(vt_symbol)
    lab = AlphaLab(ALPHA_LAB_PATH)

    d = instant.as_of.date()
    # 回看窗口：覆盖 as_of 之前最近一根已收盘 bar（如当日未收盘则回退到上一交易日）。
    start_dt = datetime.combine(d - timedelta(days=_DECISION_WINDOW_DAYS), datetime.min.time())
    end_dt = datetime.combine(d, datetime.max.time())

    # 候选周期：首选与决策同频的周期，再回退到该标的其它可用周期。
    candidates: list[str] = [interval_of_bar_freq(instant.bar_freq)]
    for available in lab.available_bar_intervals(canonical, include_derived=True):
        if available not in candidates:
            candidates.append(available)

    for candidate in candidates:
        frame = lab.load_bar_frame(
            canonical, candidate, start_dt, end_dt, include_derived=True
        )
        if frame is None or frame.is_empty() or "close" not in frame.columns:
            continue
        bar = select_decision_bar(frame, instant)  # close_time <= as_of 的最后一根（无前视）
        if bar is None:
            continue
        close = bar["close"][0]
        if close is not None:
            return float(close), candidate

    raise ValueError(
        f"决策时刻 {instant.as_of.isoformat()} 之前的 {vt_symbol} {instant.bar_freq} 行情缺失"
    )


def _select_signal_bar(
    signal_df: pl.DataFrame, vt_symbol: str, instant: DecisionInstant
) -> dict[str, Any]:
    """取目标标的的 Decision_Bar（`close_time <= as_of` 的最后一根）的 signal 与 bar 时刻。

    - signal/decision_bar_dt：过滤目标标的后经 `select_decision_bar` 取最后一根已收盘 bar。
    - price/pricing_interval：委托 `_load_close_price` 读同一 as-of 口径下的收盘价
      （首选与决策同频的周期，记录实际命中周期）。

    Args:
        signal_df: CNN 推理输出帧，至少含 `[datetime, vt_symbol, signal]` 三列，
            可含多只标的多根 bar；本函数先按 vt_symbol 过滤。
        vt_symbol: 目标标的合约代码，如 "000001.SZSE"。
        instant: 决策时刻，提供 as_of（选 bar 的截断上界）与 bar_freq（取价周期）。

    Returns:
        dict，含四个键：
        - `decision_bar_dt`（datetime）：命中 Decision_Bar 的收盘时刻；
        - `signal`（float）：该 bar 的模型信号值（概率/得分）；
        - `price`（float）：同一 as-of 口径下的收盘价（元/股）；
        - `pricing_interval`（str）：实际命中的取价周期。

    Raises:
        ValueError: as_of 之前无该标的的已收盘 bar（信号缺失）；或取价阶段
            `_load_close_price` 因行情缺失而抛出时向上传播。
    """
    rows = signal_df.filter(pl.col("vt_symbol") == vt_symbol)
    bar = select_decision_bar(rows, instant)
    if bar is None:
        raise ValueError(
            f"决策时刻 {instant.as_of.isoformat()} 之前的 {vt_symbol} 信号缺失"
        )
    price, pricing_interval = _load_close_price(vt_symbol, instant)
    return {
        "decision_bar_dt": bar["datetime"][0],
        "signal": float(bar["signal"][0]),
        "price": price,
        "pricing_interval": pricing_interval,
    }


def run_live_decision(
    *,
    model_name: str,
    vt_symbol: str,
    scheme_name: str,
    instant: DecisionInstant,
    portfolio: PortfolioSnapshot,
    buy_threshold: float,
    risk_config: RiskConfig,
    store: DecisionStore,
    notifier: Notifier,
    position_ratio: float = 0.95,
    min_volume: int = 100,
    model_version: str = "",
    should_exit: bool = False,
    halted: bool = False,
    on_progress: Optional[Callable[[float, str], None]] = None,
    trace_store: DecisionTraceStore | None = None,
    data_source_type: str = "pull",
    signal_fn: Callable[..., pl.DataFrame] | None = None,
    trigger_source: str = "manual",
) -> dict[str, Any]:
    """编排一次今日决策。

    流程：决策日 CNN 推理 -> 取当日 bar 的 signal + price -> RiskInspector 注入
    SignalService -> run_for_date（幂等：信号→风控→提醒→落盘）-> 返回结果。

    进度透传：推理段映射到 10~70%，风控/编排 80%，完成 100%。

    不调用任何券商网关 / 下单接口（Property 7）。

    Args:
        model_name: CNN 模型名，传给 predict_cnn_signals（或注入的 signal_fn）。
        vt_symbol: 目标标的合约代码，如 "000001.SZSE"。
        scheme_name: 方案名，参与 signal_id 生成与提醒标题。
        instant: 决策时刻（含 as_of 与 bar_freq），决定回看窗口与 Decision_Bar 选取。
        portfolio: 组合快照（总市值 / 持仓 / 单票市值），供风控与仓位规模计算。
        buy_threshold: 买入信号阈值（概率 / 得分），超过即触发买入风控检查。
        risk_config: 风控配置（仓位上限 / 黑名单 / 停牌放行），用于构造 RiskManager。
        store: DecisionStore，用于幂等查询与决策落盘。
        notifier: 通知器，买入 / 卖出决策时发送提醒。
        position_ratio: 目标仓位占组合市值的比例（0~1），默认 0.95。
        min_volume: 最小交易手数（股数），不足一手不买入，默认 100。
        model_version: 模型版本标签，参与 signal_id 生成（空串则不含版本）。
        should_exit: 是否到出场条件（由调用方依持有期 / 出场规则给出），默认 False。
        halted: 标的是否停牌 / 封死，默认 False。
        on_progress: 进度回调 ``(percent, message)``；为 None 时不上报进度。
        trace_store: Decision_Trace 持久化存储；为 None 时仅在内存累积、不落盘（trace 落盘
            为 best-effort，失败不影响决策返回）。
        data_source_type: 数据源类型标签（"upload" | "pull"），仅记入 trace，绝不含 token。
        signal_fn: 可选注入的推理函数，取代模块全局 predict_cnn_signals（见下方契约）。
        trigger_source: 触发来源标签（如 "manual" / "schedule"），写入 Decision 落盘。

    Returns:
        `{decision, risk_detail, idempotent_hit}`：
        - `decision`：决策 dict（action ∈ buy/sell/hold，含 volume/price/signal/reason）。
        - `risk_detail`：风控逐项明细 `list[{check, passed, detail}]`；幂等命中或未走买入
          风控的路径（如概率未达阈值的 hold、出场 sell）为空 list。
        - `idempotent_hit`：是否幂等命中（运行前同 signal_id 决策已落盘，未重新走风控/提醒）。

    Raises:
        ValueError: as_of 之前无已收盘 bar（信号或 Decision_Bar 行情缺失）时抛出；
            推理 / 取价阶段的其它异常亦原样向上传播（中止前先记 abort_reason 与 best-effort
            trace，再重新抛出，以保持既有错误响应行为）。

    signal_fn 契约（可选注入，用于测试或 Phase 3 组合调仓路径）：
    - 若传入非 None，调用该函数取代模块全局 `predict_cnn_signals`。
    - 若为 None（默认），在调用时从模块全局解析 `predict_cnn_signals`，使
      `monkeypatch.setattr(orchestrator, "predict_cnn_signals", ...)` 桩继续生效。
    - 注入的 signal_fn 必须接受与 `predict_cnn_signals` 相同的 kwargs 调用形态：
      ``signal_fn(model_name=..., start=..., end=..., on_progress=..., on_meta=...)``
    - 返回值硬约束：包含 ``[datetime, vt_symbol, signal]`` 三列的 polars DataFrame，
      下游 ``_select_signal_bar`` 依赖该 schema。

    可观测性（Requirement 8）：开头生成 `run_id` 并逐段累积 Decision_Trace（六段：
    run_header / inference / pricing / decision_logic / risk / result）。trace 持久化为
    **best-effort**——仅当传入 `trace_store` 时在 Decision 落盘后写盘，失败只记 warning 并在
    结果段标注 `trace_persisted=false`，绝不影响 Decision 落盘与返回（8.12）。产出 Decision 前
    于推理/取价阶段中止时，结果段记 `abort_reason` 且 `completed_sections` 仅含失败点之前的
    前缀（8.11），随后重新抛出异常以保持既有错误响应行为。脱敏：不含任何凭证（含 Tushare
    token），数据源仅记录类型 + bar 数量（8.7/8.8）。
    """
    # 0. 运行头：生成短码 run_id，预派生 signal_id（与最终 Decision 的 signal_id 必然一致），
    #    据此构造逐段累积的 TraceBuilder。Wave 2c：入口计时（elapsed_ms）。
    _t0 = time.monotonic()
    run_id = uuid.uuid4().hex[:8]
    # signal_id 由 Decision_Bar 决定；1d 下 = as_of 当日（收盘后触发的常态），可据 as_of 提前推导，
    # 与 run_for_instant 内据实际选中 bar 计算的 signal_id 一致（用于 trace 键与持久化）。
    signal_id = make_signal_id(
        session_close(instant.as_of.date(), instant.bar_freq),
        instant.bar_freq,
        scheme_name,
        model_version,
    )
    builder = TraceBuilder(run_id, signal_id=signal_id, logger=logger)

    # run_header：脱敏摘要——无凭证；风控配置仅摘要（比率 + 黑名单长度）；数据源仅类型。
    builder.set_section("run_header", {
        "run_id": run_id,
        "model_name": model_name,
        "model_version": model_version,
        "vt_symbol": vt_symbol,
        "scheme": scheme_name,
        "as_of": instant.as_of.isoformat(),
        "bar_freq": instant.bar_freq,
        "data_source_type": data_source_type,  # "upload" | "pull"，不含 token
        "buy_threshold": buy_threshold,
        "portfolio": {
            "portfolio_value": portfolio.portfolio_value,
            "total_position_value": portfolio.total_position_value,
            "current_position": portfolio.current_position,
            "current_symbol_value": portfolio.current_symbol_value,
        },
        "risk_config_summary": {
            "max_total_position_ratio": risk_config.max_total_position_ratio,
            "max_single_position_ratio": risk_config.max_single_position_ratio,
            "allow_when_halted": risk_config.allow_when_halted,
            "blacklist_size": len(risk_config.blacklist),  # 仅长度，不展开内容
        },
    })

    # 风控明细薄包装：包住既有 RiskManager，逐项记录而不改判定。
    inspector = RiskInspector(RiskManager(risk_config))

    try:
        # 1. 决策日推理（start == end == trade_date，只取决策日那一根 bar）。
        #    predict_cnn_signals 内部已按 lookback 预热（extended_start），只传决策日仍能产出当日信号。
        if on_progress:
            on_progress(10, f"对 as_of={instant.as_of.isoformat()} 进行 CNN 推理...")

        def _infer_progress(p: float, m: str) -> None:
            """把 CNN 推理内部进度转发到整体进度回调，并映射到 10~70 进度段。

            Args:
                p: 推理子任务进度，取值 0~100。
                m: 推理子任务的进度描述文案，转发时会加上 "[推理] " 前缀。
            """
            if on_progress:
                on_progress(10 + p * 0.6, f"[推理] {m}")  # 推理段 0~100 -> 10~70

        meta_box: dict[str, Any] = {}  # on_meta 收集器（仅符号/计数/时间，无凭证）
        # 调用时解析：默认分支从模块全局取 predict_cnn_signals，使 monkeypatch 桩继续生效；
        # 注入分支直接使用传入的 signal_fn（测试 / Phase 3 组合调仓路径）。
        fn = signal_fn if signal_fn is not None else predict_cnn_signals
        signal_df: pl.DataFrame = fn(
            model_name=model_name,
            start=instant.as_of.date() - timedelta(days=_DECISION_WINDOW_DAYS),
            end=instant.as_of.date(),
            on_progress=_infer_progress,
            on_meta=meta_box.update,
        )

        # 2. 取 Decision_Bar（close_time <= as_of 的最后一根）的 signal + price（缺失抛 ValueError）。
        bar = _select_signal_bar(signal_df, vt_symbol, instant)
        decision_bar_dt = bar["decision_bar_dt"]
        signal_value = bar["signal"]
        price = bar["price"]

        # 信号帧自描述的 objective（predict_cnn_signals 写入的常量列；缺列=legacy→None）。
        # 透传给 SignalService 做阈值尺度自检（回测实盘共用 threshold_scale_check）。
        objective: str | None = None
        if "objective" in signal_df.columns and signal_df.height > 0:
            objective = str(signal_df["objective"][0])

        # 实际 Decision_Bar 确定后据其重算 signal_id（与 run_for_instant 一致），并校正 trace 键——
        # 覆盖开头据 as_of 提前推导的临时值（处理收盘前回退到上一已收盘 bar 的情形，使 trace 与决策同键）。
        signal_id = make_signal_id(decision_bar_dt, instant.bar_freq, scheme_name, model_version)
        builder.signal_id = signal_id

        # inference：on_meta 元信息 + signal_df 序列统计；逐点信号入 DEBUG 明细。
        seq_stats = {
            "count": signal_df.height,
            "mean": float(signal_df["signal"].mean()),
            "min": float(signal_df["signal"].min()),
            "max": float(signal_df["signal"].max()),
        }
        builder.set_section(
            "inference",
            {
                **meta_box,
                "signal_seq_stats": seq_stats,
                "decision_signal": signal_value,
                "decision_bar_dt": decision_bar_dt.isoformat(),
            },
            debug_detail={"signals": signal_df["signal"].to_list()},
        )

        # pricing：实际命中的取价周期 + Decision_Bar 收盘价。
        builder.set_section("pricing", {
            "interval_used": bar["pricing_interval"],
            "close_price": price,
        })

        # 3. 用既有 SignalService 编排（风控/提醒/落盘/幂等都在其内部）。
        if on_progress:
            on_progress(80, "风控与决策编排...")

        service = SignalService(
            scheme_name=scheme_name,
            buy_threshold=buy_threshold,
            risk=inspector,  # 鸭子类型：RiskInspector 暴露 check_buy/can_trade 同签名
            store=store,
            notifier=notifier,
            position_ratio=position_ratio,
            min_volume=min_volume,
            model_version=model_version,
        )
        # 幂等判定依据：本次运行前该 signal_id 是否已有落盘决策。不能用风控明细为空反推——
        # 非买入路径（概率未达阈值/持有观望/出场卖出等）同样不调用 check_buy、不产生明细。
        existed_before = store.get(signal_id) is not None
        decision: Decision = service.run_for_instant(
            instant,
            decision_bar_dt=decision_bar_dt,
            signal=signal_value,
            price=price,
            portfolio=portfolio,
            vt_symbol=vt_symbol,
            should_exit=should_exit,
            halted=halted,
            trigger_source=trigger_source,
            objective=objective,
        )
        # Wave 2c: 捕获实测通知结果（SignalService.run_for_instant 存入 last_notify_ok）。
        # None = 未发送（幂等命中/hold 路径）；True/False = send 实测返回值。
        _notify_ok: bool | None = getattr(service, "last_notify_ok", None)

    except Exception as exc:  # noqa: BLE001 — 产出 Decision 前中止：记 abort_reason 后重新抛出
        abort_reason = str(exc)
        logger.warning("[%s] 产出 Decision 前中止: %s", run_id, abort_reason)
        # 结果段记 abort_reason 且不含成功决策字段；直接写入 sections（不计入 completed_sections，
        # 使 completed_sections 仅为失败点之前的已完成段前缀，满足 8.11 / Property 14）。
        builder._sections["result"] = {
            "action": None,
            "volume": None,
            "price": None,
            "reason": None,
            "idempotent_hit": False,
            "notified": False,
            "signal_id": signal_id,
            "trace_persisted": False,
            "trace_persist_error": None,
            "abort_reason": abort_reason,
            "elapsed_ms": int((time.monotonic() - _t0) * 1000),
            "trigger_source": trigger_source,
        }
        if trace_store is not None:
            try:
                trace_store.save_if_absent(signal_id, builder.to_trace())
            except Exception as persist_exc:  # noqa: BLE001 — best-effort
                logger.warning("[%s] 中止 trace 持久化失败: %s", run_id, persist_exc)
        raise  # 保持既有错误响应行为（ValueError 等向上传播）

    # 5. 幂等命中：运行前已存在同 signal_id 的落盘决策，SignalService 直接返回既有
    #    Decision，未重新走风控与提醒。
    idempotent_hit = existed_before
    authoritative_ok = "风控拦截" not in (decision.reason or "")
    # Wave 2c：notified 改为实测值（R5.1 语义变更）。
    # _notify_ok：True/False = send 实测返回值；None = 未发送（幂等命中/hold）。
    # trace 消费者沿用 "notified" 键，值语义更准确，无契约破坏。
    notified: bool = bool(_notify_ok) if _notify_ok is not None else False
    if _notify_ok is False:
        logger.warning(
            "[%s] 通知发送失败（send 返回 False），计划/scheme=%s", run_id, scheme_name
        )


    # decision_logic：信号 vs 阈值 + 仓位规模信息（volume/intended_value 取自决策结果）。
    intended_value = (decision.volume or 0) * (decision.price or 0.0)
    builder.set_section("decision_logic", {
        "signal": signal_value,
        "buy_threshold": buy_threshold,
        "signal_passed": signal_value > buy_threshold,
        "target_value": portfolio.portfolio_value * position_ratio,
        "volume": decision.volume,
        "intended_value": intended_value,
        "should_exit": should_exit,
        "halted": halted,
    })

    # risk：复用 RiskInspector.records + 权威放行标记。
    builder.set_section("risk", {
        "records": inspector.records,
        "authoritative_ok": authoritative_ok,
    }, debug_detail={"records": inspector.records})

    # 6. best-effort 持久化（Decision 已落盘后）：失败不影响返回（8.12）。
    # Wave 2c: elapsed_ms = 入口到此处的毫秒数（落盘完成后计算）。
    elapsed_ms = int((time.monotonic() - _t0) * 1000)
    # result：先按成功乐观标注，持久化失败再回填 trace_persisted/trace_persist_error。
    builder.set_section("result", {
        "action": decision.action,
        "volume": decision.volume,
        "price": decision.price,
        "reason": decision.reason,
        "idempotent_hit": idempotent_hit,
        "notified": notified,
        "signal_id": decision.signal_id,
        "trace_persisted": True,
        "trace_persist_error": None,
        "abort_reason": None,
        "elapsed_ms": elapsed_ms,
        "trigger_source": trigger_source,
    })

    if trace_store is not None:
        try:
            trace_store.save_if_absent(decision.signal_id, builder.to_trace())
        except Exception as exc:  # noqa: BLE001 — best-effort：绝不影响 Decision 返回
            logger.warning("[%s] trace 持久化失败: %s", run_id, exc)
            # 回填结果段持久化状态（in-memory，供后续观测使用）。
            builder._sections["result"]["trace_persisted"] = False
            builder._sections["result"]["trace_persist_error"] = str(exc)

    # 7. 汇总：决策 + 风控逐项明细（保留既有返回结构 {decision, risk_detail,
    #    idempotent_hit}，不外泄额外键——trace 可观测信息已落在持久化 trace 与日志中）。
    if on_progress:
        on_progress(100, "决策完成")

    return {
        "decision": _decision_to_dict(decision),
        "risk_detail": inspector.records,  # list[{check, passed, detail}]
        "idempotent_hit": idempotent_hit,
    }
