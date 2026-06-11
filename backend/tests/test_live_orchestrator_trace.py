"""
orchestrator Decision_Trace 采集单元/示例测试（交易操作台特性，任务 16.2）。

覆盖需求 8.1 / 8.7 / 8.8 / 8.11 / 8.12，验证 `run_live_decision` 在接入 `TraceBuilder`/
`DecisionTraceStore` 后的可观测性行为：

- 8.1：六段（run_header/inference/pricing/decision_logic/risk/result）完整、字段齐全；
  `inference` 段携带 `on_meta` 收集的元信息，且 `signal_seq_stats`（count/mean/min/max）
  与 `decision_day_signal` 由注入的 `signal_df` 正确算出。
- 过程日志：所有日志行带 `[run_id]` 前缀，六段里程碑为 INFO、重负载明细为 DEBUG。
- 8.11：推理/取价阶段中止时重新抛出异常，持久化的 trace `completed_sections` 为已完成段
  的真前缀（不含 "result"），且结果段含 `abort_reason`。
- 8.12：trace 持久化失败时仍返回 decision 且 `DecisionStore.get` 可读回，
  结果段标注 `trace_persisted=false`。

外部 I/O 全部桩化：`predict_cnn_signals`（含其 `on_meta` 回调）与 `_load_close_price`
注入确定 signal/price，`DecisionStore`/`DecisionTraceStore` 用 `tmp_path`。不依赖外部网络。
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import polars as pl
import pytest

from aitrade.live import orchestrator
from aitrade.live.decision import DecisionStore
from aitrade.live.decision_instant import DecisionInstant, make_signal_id
from aitrade.live.decision_trace import DecisionTraceStore
from aitrade.live.notifier import LogNotifier
from aitrade.live.orchestrator import run_live_decision
from aitrade.live.risk import RiskConfig
from aitrade.live.signal_service import PortfolioSnapshot


TRADE_DATE = date(2026, 6, 9)
AS_OF = datetime(2026, 6, 9, 15, 5)  # 收盘后；Decision_Bar = 当日
INSTANT = DecisionInstant(AS_OF, "1d")
VT_SYMBOL = "000001.SZSE"
SCHEME = "eod_buy_v1"
MODEL = "测试"
MODEL_VERSION = "v3"
LOGGER_NAME = "aitrade.live.orchestrator"

# 与 run_live_decision 内据 Decision_Bar 计算一致的 signal_id（1d → 当日日期键）。
SIGNAL_ID = make_signal_id(datetime.combine(TRADE_DATE, datetime.min.time()), "1d", SCHEME, MODEL_VERSION)

# 注入的多根信号序列（同一目标标的、决策日、递增时间），使序列统计有意义：
# decision_day_signal 取当日最后一根 = 0.72；count=3、min=0.2、max=0.72。
SIGNALS = [0.2, 0.5, 0.72]
EXPECTED_DECISION_SIGNAL = 0.72

# on_meta 注入的确定元信息（仅符号/计数/时间，无任何凭证）。
META = {
    "target_symbol": VT_SYMBOL,
    "lookback": 240,
    "input_interval": "30m",
    "objective": "nb_cls",
    "observation_symbols": [VT_SYMBOL, "INDEX.TEST"],
    "observation_group_count": 2,
    "warmup_start": "2026-05-01",
    "total_steps": 480,
    "valid_points": 3,
    "per_symbol_bars": {VT_SYMBOL: 240, "INDEX.TEST": 240},
}

SIX_SECTIONS = [
    "run_header",
    "inference",
    "pricing",
    "decision_logic",
    "risk",
    "result",
]


def _signal_frame() -> pl.DataFrame:
    """构造 predict_cnn_signals 同 schema 的桩输出：[datetime, vt_symbol, signal]。

    三根 bar 同属目标标的、决策日，时间递增，便于校验序列统计与决策日取最后一根。
    """
    base = datetime.combine(TRADE_DATE, datetime.min.time())
    return pl.DataFrame(
        {
            "datetime": [base.replace(hour=h) for h in (10, 11, 14)],
            "vt_symbol": [VT_SYMBOL] * len(SIGNALS),
            "signal": [float(s) for s in SIGNALS],
        }
    )


def _stub_predict_with_meta(monkeypatch) -> None:
    """桩化 predict_cnn_signals：返回确定 signal_df，并以确定 META 调用 on_meta。"""

    def _stub(*, on_meta=None, on_progress=None, **kwargs):
        if on_meta is not None:
            on_meta(dict(META))  # 模拟真实预测器一次性吐出结构化元信息
        return _signal_frame()

    monkeypatch.setattr(orchestrator, "predict_cnn_signals", _stub)


def _stub_pricing(monkeypatch, price: float = 10.0) -> None:
    monkeypatch.setattr(
        orchestrator,
        "_load_close_price",
        lambda vt_symbol, instant: (float(price), "d"),
    )


def _run(
    *,
    store: DecisionStore,
    notifier: LogNotifier,
    portfolio: PortfolioSnapshot,
    trace_store: DecisionTraceStore | None = None,
    buy_threshold: float = 0.6,
    data_source_type: str = "pull",
) -> dict:
    return run_live_decision(
        model_name=MODEL,
        vt_symbol=VT_SYMBOL,
        scheme_name=SCHEME,
        instant=INSTANT,
        portfolio=portfolio,
        buy_threshold=buy_threshold,
        risk_config=RiskConfig(
            max_total_position_ratio=0.95, max_single_position_ratio=0.95
        ),
        store=store,
        notifier=notifier,
        model_version=MODEL_VERSION,
        trace_store=trace_store,
        data_source_type=data_source_type,
    )


# ---------------------------------------------------------------------------
# 1. happy path：六段完整、字段齐全、序列统计与 on_meta 正确（需求 8.1）
# ---------------------------------------------------------------------------
def test_trace_collects_six_complete_sections(tmp_path, monkeypatch) -> None:
    _stub_predict_with_meta(monkeypatch)
    _stub_pricing(monkeypatch, price=10.0)
    store = DecisionStore(tmp_path)
    trace_store = DecisionTraceStore(tmp_path)
    pf = PortfolioSnapshot(portfolio_value=100000, total_position_value=0, current_position=0)

    result = _run(store=store, notifier=LogNotifier(), portfolio=pf, trace_store=trace_store)
    assert result["decision"]["action"] == "buy"

    # 持久化的 trace 可读回，且六段齐全、completed_sections 同序。
    trace = trace_store.get(SIGNAL_ID)
    assert trace is not None
    assert trace["signal_id"] == SIGNAL_ID
    assert trace["completed_sections"] == SIX_SECTIONS
    assert set(trace["sections"].keys()) == set(SIX_SECTIONS)
    sections = trace["sections"]

    # run_header：脱敏摘要——数据源仅类型、风控配置仅摘要（含 blacklist_size），无凭证。
    rh = sections["run_header"]
    assert rh["data_source_type"] == "pull"
    assert rh["model_name"] == MODEL
    assert rh["vt_symbol"] == VT_SYMBOL
    assert rh["risk_config_summary"]["blacklist_size"] == 0
    assert "token" not in rh["risk_config_summary"]

    # inference：on_meta 元信息原样保留 + 序列统计由 signal_df 正确算出。
    inf = sections["inference"]
    assert inf["target_symbol"] == VT_SYMBOL
    assert inf["lookback"] == META["lookback"]
    assert inf["per_symbol_bars"] == META["per_symbol_bars"]
    assert inf["decision_signal"] == EXPECTED_DECISION_SIGNAL
    stats = inf["signal_seq_stats"]
    assert stats["count"] == len(SIGNALS)
    assert stats["min"] == pytest.approx(min(SIGNALS))
    assert stats["max"] == pytest.approx(max(SIGNALS))
    assert stats["mean"] == pytest.approx(sum(SIGNALS) / len(SIGNALS))

    # pricing：取价周期 + 决策日收盘价。
    assert sections["pricing"]["close_price"] == 10.0
    assert sections["pricing"]["interval_used"] == "d"

    # decision_logic：信号 vs 阈值。
    dl = sections["decision_logic"]
    assert dl["signal"] == EXPECTED_DECISION_SIGNAL
    assert dl["buy_threshold"] == 0.6
    assert dl["signal_passed"] is True

    # risk：5 项明细齐全。
    risk_checks = [r["check"] for r in sections["risk"]["records"]]
    assert risk_checks == [
        "kill_switch_or_circuit",
        "blacklist",
        "halted",
        "max_total_position",
        "max_single_position",
    ]

    # result：成功路径标注。
    res = sections["result"]
    assert res["action"] == "buy"
    assert res["signal_id"] == SIGNAL_ID
    assert res["trace_persisted"] is True
    assert res["abort_reason"] is None
    assert res["idempotent_hit"] is False


# ---------------------------------------------------------------------------
# 2. 过程日志：每行带 [run_id] 前缀，六段里程碑 INFO、明细 DEBUG
# ---------------------------------------------------------------------------
def test_trace_logging_run_id_prefix_and_levels(tmp_path, monkeypatch, caplog) -> None:
    _stub_predict_with_meta(monkeypatch)
    _stub_pricing(monkeypatch, price=10.0)
    store = DecisionStore(tmp_path)
    trace_store = DecisionTraceStore(tmp_path)
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        _run(store=store, notifier=LogNotifier(), portfolio=pf, trace_store=trace_store)

    run_id = trace_store.get(SIGNAL_ID)["run_id"]
    prefix = f"[{run_id}]"

    records = [r for r in caplog.records if r.name == LOGGER_NAME]
    assert records, "应至少有一条 orchestrator 日志"
    # 每行均带 [run_id] 前缀。
    for rec in records:
        assert rec.getMessage().startswith(prefix), rec.getMessage()

    # 六段里程碑均为 INFO。
    milestone_records = [
        r for r in records if r.levelno == logging.INFO and "段完成" in r.getMessage()
    ]
    completed_names = {r.getMessage().split("段完成: ")[-1] for r in milestone_records}
    assert completed_names == set(SIX_SECTIONS)
    assert len(milestone_records) == len(SIX_SECTIONS)

    # 重负载明细为 DEBUG（inference 的逐点信号、risk 的逐项记录）。
    debug_records = [
        r for r in records if r.levelno == logging.DEBUG and "明细" in r.getMessage()
    ]
    assert debug_records, "应至少有一条 DEBUG 明细日志"


# ---------------------------------------------------------------------------
# 3. 中止用例：取价抛 ValueError → 重新抛出；completed_sections 为前缀且结果段含 abort_reason
# ---------------------------------------------------------------------------
def test_trace_abort_records_prefix_and_abort_reason(tmp_path, monkeypatch) -> None:
    _stub_predict_with_meta(monkeypatch)

    def _raise(vt_symbol, instant):
        raise ValueError(f"决策时刻 {instant.as_of.isoformat()} 之前的 {vt_symbol} 行情缺失")

    monkeypatch.setattr(orchestrator, "_load_close_price", _raise)
    store = DecisionStore(tmp_path)
    trace_store = DecisionTraceStore(tmp_path)
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)

    with pytest.raises(ValueError, match="行情缺失"):
        _run(store=store, notifier=LogNotifier(), portfolio=pf, trace_store=trace_store)

    # 中止仍持久化了 trace（best-effort），可读回检查。
    trace = trace_store.get(SIGNAL_ID)
    assert trace is not None

    # completed_sections 是六段的真前缀，且绝不含 "result"（需求 8.11 / Property 14）。
    completed = trace["completed_sections"]
    assert "result" not in completed
    assert completed == SIX_SECTIONS[: len(completed)]
    # 取价在 inference 段写入前抛出 → 仅 run_header 完成。
    assert completed == ["run_header"]

    # 结果段存在且含 abort_reason，不含成功决策字段。
    res = trace["sections"]["result"]
    assert res["abort_reason"] is not None
    assert "行情缺失" in res["abort_reason"]
    assert res["action"] is None
    assert res["trace_persisted"] is False

    # 中止前未产出 Decision → 决策未落盘。
    assert store.get(SIGNAL_ID) is None


# ---------------------------------------------------------------------------
# 4. 持久化失败用例：save_if_absent 抛错 → 仍返回 decision 且可读回；结果段 trace_persisted=false
# ---------------------------------------------------------------------------
def test_trace_persist_failure_does_not_affect_decision(tmp_path, monkeypatch) -> None:
    _stub_predict_with_meta(monkeypatch)
    _stub_pricing(monkeypatch, price=10.0)
    store = DecisionStore(tmp_path)
    trace_store = DecisionTraceStore(tmp_path)
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)

    # 让 trace 持久化抛错（模拟磁盘故障）。
    def _boom(signal_id, trace):
        raise RuntimeError("disk full")

    monkeypatch.setattr(trace_store, "save_if_absent", _boom)

    # 捕获 orchestrator 内部构造的 TraceBuilder 以检查 in-memory 结果段状态。
    captured: dict = {}
    real_builder_cls = orchestrator.TraceBuilder

    def _capturing_builder(*args, **kwargs):
        builder = real_builder_cls(*args, **kwargs)
        captured["builder"] = builder
        return builder

    monkeypatch.setattr(orchestrator, "TraceBuilder", _capturing_builder)

    result = _run(store=store, notifier=LogNotifier(), portfolio=pf, trace_store=trace_store)

    # 持久化失败不影响 Decision 返回与落盘（需求 8.12）。
    assert result["decision"]["action"] == "buy"
    persisted = store.get(SIGNAL_ID)
    assert persisted is not None
    assert persisted.signal_id == SIGNAL_ID
    assert persisted.action == "buy"

    # trace 未落盘（save 抛错）。
    assert trace_store.get(SIGNAL_ID) is None

    # in-memory 结果段标注 trace_persisted=false 且记录持久化错误。
    res_section = captured["builder"]._sections["result"]
    assert res_section["trace_persisted"] is False
    assert "disk full" in res_section["trace_persist_error"]
