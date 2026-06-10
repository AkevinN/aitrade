"""
交易操作台「决策过程可观测性」后端属性测试（Hypothesis，Requirement 8 / Property 9–15）。

每条正确性属性用单个属性测试实现，`@settings(max_examples=100)`，外部 I/O
（`predict_cnn_signals`（含其 `on_meta` 回调）、`_load_close_price`）全部桩化以注入确定的
signal/price/META，`DecisionStore` / `DecisionTraceStore` 用临时目录隔离（每个 Hypothesis
样例用独立 `tempfile.TemporaryDirectory()`，避免同一 signal_id 在样例间相互幂等命中，
并规避 Hypothesis 的 function-scoped-fixture 健康检查）。

Property 1–8 见 `test_live_properties.py`，本文件仅覆盖 Property 9–15。
属性文本见 design.md「Correctness Properties」。
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, fields
from datetime import date, datetime

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aitrade.live import orchestrator
from aitrade.live.decision import Decision, DecisionStore
from aitrade.live.decision_instant import DecisionInstant, make_signal_id
from aitrade.live.decision_trace import DecisionTraceStore
from aitrade.live.notifier import LogNotifier
from aitrade.live.orchestrator import run_live_decision
from aitrade.live.risk import RiskConfig, RiskManager
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

# 六段顺序常量（与 TraceBuilder.SECTIONS 一致）。
SIX_SECTIONS = ["run_header", "inference", "pricing", "decision_logic", "risk", "result"]

# 风控五项检查（按 RiskInspector 记录顺序）。
_EXPECTED_RISK_CHECKS = [
    "kill_switch_or_circuit",
    "blacklist",
    "halted",
    "max_total_position",
    "max_single_position",
]

# on_meta 注入的确定元信息（仅符号 / 计数 / 时间，无任何凭证）。
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


def _signal_frame(signal: float) -> pl.DataFrame:
    """构造 predict_cnn_signals 同 schema 的桩输出：[datetime, vt_symbol, signal]。"""
    return pl.DataFrame(
        {
            "datetime": [datetime.combine(TRADE_DATE, datetime.min.time())],
            "vt_symbol": [VT_SYMBOL],
            "signal": [float(signal)],
        }
    )


def _stub_io(monkeypatch, *, signal: float, price: float) -> None:
    """桩化外部 I/O：CNN 推理返回确定 signal（并以确定 META 回调 on_meta），取价返回确定 price。"""

    def _predict(*, on_meta=None, on_progress=None, **kwargs):
        if on_meta is not None:
            on_meta(dict(META))  # 模拟真实预测器一次性吐出结构化元信息
        return _signal_frame(signal)

    monkeypatch.setattr(orchestrator, "predict_cnn_signals", _predict)
    monkeypatch.setattr(
        orchestrator,
        "_load_close_price",
        lambda vt_symbol, instant: (float(price), "d"),
    )


def _patch_risk_manager(monkeypatch, *, kill: bool, circuit: bool) -> None:
    """让编排器构造的 RiskManager 携带指定运行时 kill-switch / 熔断状态（非 RiskConfig 字段）。"""
    real_cls = RiskManager

    def factory(config):
        rm = real_cls(config)
        if kill:
            rm.kill_switch = True
        if circuit:
            rm.circuit_broken = True
        return rm

    monkeypatch.setattr(orchestrator, "RiskManager", factory)


def _run(
    *,
    store: DecisionStore,
    notifier,
    portfolio: PortfolioSnapshot,
    risk_config: RiskConfig,
    buy_threshold: float,
    trace_store: DecisionTraceStore | None = None,
    should_exit: bool = False,
    halted: bool = False,
    data_source_type: str = "pull",
) -> dict:
    return run_live_decision(
        model_name=MODEL,
        vt_symbol=VT_SYMBOL,
        scheme_name=SCHEME,
        instant=INSTANT,
        portfolio=portfolio,
        buy_threshold=buy_threshold,
        risk_config=risk_config,
        store=store,
        notifier=notifier,
        model_version=MODEL_VERSION,
        should_exit=should_exit,
        halted=halted,
        trace_store=trace_store,
        data_source_type=data_source_type,
    )


# ---------------------------------------------------------------------------
# 生成器（见 design.md「生成器要点」，与 test_live_properties.py 同口径）
# ---------------------------------------------------------------------------
_finite = dict(allow_nan=False, allow_infinity=False)

signals = st.floats(min_value=0.0, max_value=1.0, **_finite)
prices = st.floats(min_value=0.01, max_value=10000.0, **_finite)
thresholds = st.floats(min_value=0.0, max_value=1.0, **_finite)
ratios = st.floats(min_value=0.01, max_value=1.0, **_finite)
data_sources = st.sampled_from(["upload", "pull"])


@st.composite
def portfolios(draw) -> PortfolioSnapshot:
    """组合快照：portfolio_value > 0，current_position ∈ {0, >0}，各市值非负。"""
    return PortfolioSnapshot(
        portfolio_value=draw(st.floats(min_value=1.0, max_value=1e9, **_finite)),
        total_position_value=draw(st.floats(min_value=0.0, max_value=1e9, **_finite)),
        current_position=draw(st.integers(min_value=0, max_value=100000)),
        current_symbol_value=draw(st.floats(min_value=0.0, max_value=1e9, **_finite)),
    )


@st.composite
def risk_configs(draw) -> RiskConfig:
    """风控配置：随机黑名单含/不含目标标的、随机比率上限、随机 allow_when_halted。"""
    blacklist: set[str] = set()
    if draw(st.booleans()):
        blacklist.add(VT_SYMBOL)
    if draw(st.booleans()):
        blacklist.add("999999.SZSE")
    return RiskConfig(
        blacklist=blacklist,
        max_total_position_ratio=draw(ratios),
        max_single_position_ratio=draw(ratios),
        allow_when_halted=draw(st.booleans()),
    )


@st.composite
def any_decision_scenarios(draw) -> dict:
    """覆盖 buy/sell/hold 三路径的宽范围决策场景（任意成功完成的决策运行）。"""
    return {
        "signal": draw(signals),
        "price": draw(prices),
        "buy_threshold": draw(thresholds),
        "portfolio": draw(portfolios()),
        "risk_config": draw(risk_configs()),
        "should_exit": draw(st.booleans()),
        "halted": draw(st.booleans()),
        "kill": draw(st.booleans()),
        "circuit": draw(st.booleans()),
        "data_source_type": draw(data_sources),
    }


@st.composite
def buy_candidates(draw) -> dict:
    """生成必然走到 check_buy 的买入候选（空仓 + 信号达标 + 资金充足）。

    据此保证风控段 `records` 完整覆盖五项（RiskInspector 在 check_buy 内总是重放五项）。
    风控配置 / 停牌 / kill-switch / 熔断随机（放行与拦截皆可能）。
    """
    buy_threshold = draw(st.floats(min_value=0.0, max_value=0.8, **_finite))
    signal = draw(st.floats(min_value=0.85, max_value=1.0, **_finite))
    price = draw(st.floats(min_value=1.0, max_value=100.0, **_finite))
    portfolio_value = draw(st.floats(min_value=1e6, max_value=1e8, **_finite))
    portfolio = PortfolioSnapshot(
        portfolio_value=portfolio_value,
        total_position_value=draw(st.floats(min_value=0.0, max_value=1e8, **_finite)),
        current_position=0,  # 空仓 -> 走入场分支
        current_symbol_value=draw(st.floats(min_value=0.0, max_value=1e8, **_finite)),
    )
    return {
        "signal": signal,
        "price": price,
        "buy_threshold": buy_threshold,
        "portfolio": portfolio,
        "risk_config": draw(risk_configs()),
        "halted": draw(st.booleans()),
        "kill": draw(st.booleans()),
        "circuit": draw(st.booleans()),
        "should_exit": draw(st.booleans()),
        "data_source_type": draw(data_sources),
    }


# ---------------------------------------------------------------------------
# Property 9: Decision_Trace 六段完整性
# ---------------------------------------------------------------------------
# Feature: trading-console, Property 9: Decision_Trace 六段完整性
# 对任意成功完成的决策运行，其生成的 Decision_Trace 的 sections 恰好包含六个
# Trace_Section（run_header/inference/pricing/decision_logic/risk/result），且每段含其
# 规定的关键字段（推理段含 total_steps/valid_points/signal_seq_stats，风控段含五项
# {check, passed, detail}，结果段含 action/idempotent_hit/trace_persisted）。
# Validates: Requirements 8.1
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(candidate=buy_candidates())
def test_property_9_decision_trace_six_sections_complete(monkeypatch, candidate):
    _stub_io(monkeypatch, signal=candidate["signal"], price=candidate["price"])
    _patch_risk_manager(monkeypatch, kill=candidate["kill"], circuit=candidate["circuit"])

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        trace_store = DecisionTraceStore(tmpdir)

        _run(
            store=store,
            notifier=LogNotifier(),
            portfolio=candidate["portfolio"],
            risk_config=candidate["risk_config"],
            buy_threshold=candidate["buy_threshold"],
            trace_store=trace_store,
            should_exit=candidate["should_exit"],
            halted=candidate["halted"],
            data_source_type=candidate["data_source_type"],
        )

        trace = trace_store.get(SIGNAL_ID)
        assert trace is not None, "成功运行后应持久化 trace"

        # (8.1) completed_sections 恰为六段且同序；sections 键集合覆盖全部六段。
        assert trace["completed_sections"] == SIX_SECTIONS
        assert set(trace["sections"].keys()) == set(SIX_SECTIONS)
        assert trace["signal_id"] == SIGNAL_ID
        sections = trace["sections"]

        # run_header：脱敏摘要——数据源仅类型、风控配置仅摘要（含 blacklist_size）。
        rh = sections["run_header"]
        for key in ("run_id", "model_name", "vt_symbol", "data_source_type",
                    "buy_threshold", "portfolio", "risk_config_summary"):
            assert key in rh, f"run_header 缺字段 {key}: {rh}"
        assert "blacklist_size" in rh["risk_config_summary"]

        # inference：on_meta 元信息关键字段 + 序列统计。
        inf = sections["inference"]
        for key in ("target_symbol", "total_steps", "valid_points",
                    "per_symbol_bars", "signal_seq_stats", "decision_signal"):
            assert key in inf, f"inference 缺字段 {key}: {inf}"
        for key in ("count", "mean", "min", "max"):
            assert key in inf["signal_seq_stats"], inf["signal_seq_stats"]

        # pricing：使用周期 + 决策日收盘价。
        pr = sections["pricing"]
        assert "interval_used" in pr and "close_price" in pr

        # decision_logic：信号 vs 阈值 + 仓位测算。
        dl = sections["decision_logic"]
        for key in ("signal", "buy_threshold", "signal_passed", "volume", "intended_value"):
            assert key in dl, f"decision_logic 缺字段 {key}: {dl}"

        # risk：五项明细齐全，每项含 check/passed/detail。
        records = sections["risk"]["records"]
        assert [r["check"] for r in records] == _EXPECTED_RISK_CHECKS
        for r in records:
            assert isinstance(r["passed"], bool)
            assert isinstance(r["detail"], str) and r["detail"]
        assert "authoritative_ok" in sections["risk"]

        # result：成功路径关键字段齐全。
        res = sections["result"]
        for key in ("action", "idempotent_hit", "trace_persisted", "signal_id", "abort_reason"):
            assert key in res, f"result 缺字段 {key}: {res}"
        assert res["action"] in ("buy", "sell", "hold")
        assert res["trace_persisted"] is True
        assert res["abort_reason"] is None


# ---------------------------------------------------------------------------
# Property 10: Decision_Trace 持久化往返一致（含模拟重启后从磁盘重读）
# ---------------------------------------------------------------------------
# Feature: trading-console, Property 10: Decision_Trace 持久化往返一致
# 对任意被 Decision_Trace_Store 持久化的 Decision_Trace，以同一 signal_id 读取
# （含模拟重启后重新从磁盘读取）后所有字段值与写入时相等。
# Validates: Requirements 8.2, 8.4
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scenario=any_decision_scenarios())
def test_property_10_decision_trace_persist_roundtrip(monkeypatch, scenario):
    _stub_io(monkeypatch, signal=scenario["signal"], price=scenario["price"])
    _patch_risk_manager(monkeypatch, kill=scenario["kill"], circuit=scenario["circuit"])

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        trace_store = DecisionTraceStore(tmpdir)

        # 捕获实际写盘的 trace 作为「写入时」基准快照（深拷贝，避免后续 in-memory 改写干扰）。
        written: dict = {}
        real_save = trace_store.save_if_absent

        def _capturing_save(signal_id, trace):
            written["snapshot"] = json.loads(json.dumps(trace))
            return real_save(signal_id, trace)

        monkeypatch.setattr(trace_store, "save_if_absent", _capturing_save)

        _run(
            store=store,
            notifier=LogNotifier(),
            portfolio=scenario["portfolio"],
            risk_config=scenario["risk_config"],
            buy_threshold=scenario["buy_threshold"],
            trace_store=trace_store,
            should_exit=scenario["should_exit"],
            halted=scenario["halted"],
            data_source_type=scenario["data_source_type"],
        )
        assert "snapshot" in written, "成功运行应写入一次 trace"

        # (8.2/8.4) 模拟重启：用全新构造的 Store 指向同一目录从磁盘重读。
        restarted_store = DecisionTraceStore(tmpdir)
        reloaded = restarted_store.get(SIGNAL_ID)

        # 读回的 trace 非空，且与写入时所有字段值相等（JSON 往返一致）。
        assert reloaded is not None, "模拟重启后应能从磁盘重读 trace"
        assert reloaded == written["snapshot"], (
            f"trace 往返不一致:\n  written={written['snapshot']}\n  reloaded={reloaded}"
        )


# ---------------------------------------------------------------------------
# Property 11: trace 与 Decision 关联一致且不改 Decision schema
# ---------------------------------------------------------------------------
# Feature: trading-console, Property 11: trace 与 Decision 关联一致且不改 Decision schema
# 对任意一次决策运行，其 Decision_Trace 的 signal_id 与同次产出的 Decision 的 signal_id
# 相等；且 trace 持久化为独立 sibling 文件（{signal_id}.trace.json），对应
# {signal_id}.json 的 Decision schema（字段集合）保持不变。
# Validates: Requirements 8.2, 8.3
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scenario=any_decision_scenarios())
def test_property_11_trace_linked_and_decision_schema_unchanged(monkeypatch, scenario):
    _stub_io(monkeypatch, signal=scenario["signal"], price=scenario["price"])
    _patch_risk_manager(monkeypatch, kill=scenario["kill"], circuit=scenario["circuit"])

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        trace_store = DecisionTraceStore(tmpdir)

        result = _run(
            store=store,
            notifier=LogNotifier(),
            portfolio=scenario["portfolio"],
            risk_config=scenario["risk_config"],
            buy_threshold=scenario["buy_threshold"],
            trace_store=trace_store,
            should_exit=scenario["should_exit"],
            halted=scenario["halted"],
            data_source_type=scenario["data_source_type"],
        )
        decision = result["decision"]
        signal_id = decision["signal_id"]

        # (8.2) trace 的 signal_id 与同次产出的 Decision 的 signal_id 相等。
        trace = trace_store.get(signal_id)
        assert trace is not None
        assert trace["signal_id"] == signal_id == SIGNAL_ID

        # (8.3) Decision schema 不变：从 DecisionStore 读回的 Decision 字段集合恰为
        # Decision dataclass 的字段集合（trace 未污染 Decision）。
        reloaded = store.get(signal_id)
        assert reloaded is not None
        decision_field_names = {f.name for f in fields(Decision)}
        assert set(asdict(reloaded).keys()) == decision_field_names
        # 读回的 Decision 与返回的 decision 完全一致。
        assert asdict(reloaded) == decision

        # (8.3) trace 为独立 sibling 文件：{signal_id}.json 与 {signal_id}.trace.json 各自存在。
        safe = signal_id.replace("/", "_").replace(":", "_")
        decision_file = store.base_path / f"{safe}.json"
        trace_file = trace_store.base_path / f"{safe}.trace.json"
        assert decision_file.exists()
        assert trace_file.exists()
        assert decision_file != trace_file
        # Decision 文件内容确为纯 Decision（字段集合不含 trace 专有键）。
        decision_json = json.loads(decision_file.read_text(encoding="utf-8"))
        assert set(decision_json.keys()) == decision_field_names
        assert "sections" not in decision_json and "completed_sections" not in decision_json


# ---------------------------------------------------------------------------
# Property 12: 幂等命中不新增 trace 且返回首次 trace
# ---------------------------------------------------------------------------
# Feature: trading-console, Property 12: 幂等命中不新增 trace 且返回首次 trace
# 对任意决策请求，对同一 signal_id 第二次触发命中幂等时，Decision_Trace_Store 不写入新的
# trace（首次 trace 文件不被重写），且返回的 trace 与首次相等；该运行的结果段标注
# idempotent_hit=true（首次为 false）。
# Validates: Requirements 8.9, 8.10
#
# 用 buy_candidates（空仓 + 信号达标 + 资金充足）使首次运行走到 check_buy，
# 覆盖明细最全的路径（幂等判定现以「运行前 signal_id 是否已落盘」为准，与是否走
# check_buy 无关；非买入路径的首次/二次判定见 test_live_orchestrator.py）。
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scenario=buy_candidates())
def test_property_12_idempotent_hit_keeps_first_trace(monkeypatch, scenario):
    _stub_io(monkeypatch, signal=scenario["signal"], price=scenario["price"])
    _patch_risk_manager(monkeypatch, kill=scenario["kill"], circuit=scenario["circuit"])

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        trace_store = DecisionTraceStore(tmpdir)

        first = _run(
            store=store,
            notifier=LogNotifier(),
            portfolio=scenario["portfolio"],
            risk_config=scenario["risk_config"],
            buy_threshold=scenario["buy_threshold"],
            trace_store=trace_store,
            should_exit=scenario["should_exit"],
            halted=scenario["halted"],
            data_source_type=scenario["data_source_type"],
        )
        # 首次运行结果段：idempotent_hit=false（从持久化 trace 读取）。
        first_trace = trace_store.get(SIGNAL_ID)
        assert first_trace is not None
        assert first_trace["sections"]["result"]["idempotent_hit"] is False
        assert first["idempotent_hit"] is False

        # 记录首次 trace 文件的原始字节与修改时间，验证不被重写。
        safe = SIGNAL_ID.replace("/", "_").replace(":", "_")
        trace_file = trace_store.base_path / f"{safe}.trace.json"
        first_bytes = trace_file.read_bytes()
        first_mtime = trace_file.stat().st_mtime_ns

        # 第二次触发同一 signal_id：命中幂等。
        second = _run(
            store=store,
            notifier=LogNotifier(),
            portfolio=scenario["portfolio"],
            risk_config=scenario["risk_config"],
            buy_threshold=scenario["buy_threshold"],
            trace_store=trace_store,
            should_exit=scenario["should_exit"],
            halted=scenario["halted"],
            data_source_type=scenario["data_source_type"],
        )

        # (8.10) 第二次运行命中幂等。
        assert second["idempotent_hit"] is True

        # (8.9) Decision_Trace_Store 不写入新 trace：首次 trace 文件内容与 mtime 均未变。
        assert trace_file.read_bytes() == first_bytes
        assert trace_file.stat().st_mtime_ns == first_mtime

        # (8.9) 再次读取返回的 trace 与首次相等。
        assert trace_store.get(SIGNAL_ID) == first_trace


# ---------------------------------------------------------------------------
# Property 13: 日志与 trace 不含敏感字段
# ---------------------------------------------------------------------------
# 哨兵 token：注入到环境变量与风控配置黑名单，断言日志与 trace 均不泄露。
_sentinel_tokens = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=8, max_size=24
).map(lambda s: f"SENTINELTOKEN_{s}")


# Feature: trading-console, Property 13: 日志与 trace 不含敏感字段
# 对任意决策运行（即使把哨兵凭证字符串注入环境或风控配置），其 Process_Log 文本与持久化
# Decision_Trace 的序列化文本均不包含任何凭证或密钥（含 Tushare token）；数据源信息仅以
# 类型（upload/pull）与 bar 数量记录。
# Validates: Requirements 8.7, 8.8
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(candidate=buy_candidates(), sentinel=_sentinel_tokens)
def test_property_13_no_sensitive_fields_in_log_or_trace(monkeypatch, caplog, candidate, sentinel):
    _stub_io(monkeypatch, signal=candidate["signal"], price=candidate["price"])
    _patch_risk_manager(monkeypatch, kill=candidate["kill"], circuit=candidate["circuit"])

    # 注入哨兵凭证：环境变量（含 Tushare token）+ 风控配置黑名单（一个伪标的）。
    monkeypatch.setenv("TUSHARE_TOKEN", sentinel)
    monkeypatch.setenv("AITRADE_SECRET", sentinel)
    risk_config = candidate["risk_config"]
    risk_config.blacklist = set(risk_config.blacklist) | {sentinel}

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        trace_store = DecisionTraceStore(tmpdir)

        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            _run(
                store=store,
                notifier=LogNotifier(),
                portfolio=candidate["portfolio"],
                risk_config=risk_config,
                buy_threshold=candidate["buy_threshold"],
                trace_store=trace_store,
                should_exit=candidate["should_exit"],
                halted=candidate["halted"],
                data_source_type=candidate["data_source_type"],
            )

        trace = trace_store.get(SIGNAL_ID)
        assert trace is not None

        # (8.7) Process_Log（含 DEBUG 明细）不含哨兵 token。
        log_text = "\n".join(
            r.getMessage() for r in caplog.records if r.name == LOGGER_NAME
        )
        assert sentinel not in log_text, "过程日志泄露了哨兵凭证"

        # (8.8) 序列化的 Decision_Trace 文本不含哨兵 token。
        trace_text = json.dumps(trace, ensure_ascii=False)
        assert sentinel not in trace_text, "Decision_Trace 泄露了哨兵凭证"

        # (8.8) 数据源信息仅以类型记录（run_header 仅 data_source_type，不展开黑名单内容）。
        rh = trace["sections"]["run_header"]
        assert rh["data_source_type"] in ("upload", "pull")
        assert "blacklist" not in rh["risk_config_summary"]
        assert "blacklist_size" in rh["risk_config_summary"]


# ---------------------------------------------------------------------------
# Property 14: 中止运行只记录已完成段前缀与中止原因
# ---------------------------------------------------------------------------
@st.composite
def abort_scenarios(draw) -> dict:
    """在产出 Decision 之前于推理 / 取价阶段抛 ValueError 中止的运行场景。"""
    where = draw(st.sampled_from(["inference", "pricing"]))
    marker = draw(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=4, max_size=10)
    )
    return {
        "where": where,
        "reason": f"行情缺失-{marker}",
        "signal": draw(signals),
        "price": draw(prices),
        "buy_threshold": draw(thresholds),
        "portfolio": draw(portfolios()),
        "risk_config": draw(risk_configs()),
        "data_source_type": draw(data_sources),
    }


# Feature: trading-console, Property 14: 中止运行只记录已完成段前缀与中止原因
# 对任意在产出 Decision 之前于某阶段（推理 / 取价）失败中止的运行，其 Decision_Trace 的
# completed_sections 恰为失败点之前已完成段的前缀（绝不含 "result"），且结果段记录
# abort_reason，不含成功决策字段。
# Validates: Requirements 8.11
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scenario=abort_scenarios())
def test_property_14_abort_records_prefix_and_reason(monkeypatch, scenario):
    reason = scenario["reason"]

    if scenario["where"] == "inference":
        # 推理阶段抛错：run_header 之后立即中止。
        def _predict(*, on_meta=None, on_progress=None, **kwargs):
            raise ValueError(reason)

        monkeypatch.setattr(orchestrator, "predict_cnn_signals", _predict)
        monkeypatch.setattr(
            orchestrator,
            "_load_close_price",
            lambda vt_symbol, instant: (float(scenario["price"]), "d"),
        )
    else:
        # 取价阶段抛错：推理成功，_select_signal_bar 内取价时中止。
        def _predict(*, on_meta=None, on_progress=None, **kwargs):
            if on_meta is not None:
                on_meta(dict(META))
            return _signal_frame(scenario["signal"])

        def _raise_price(vt_symbol, instant):
            raise ValueError(reason)

        monkeypatch.setattr(orchestrator, "predict_cnn_signals", _predict)
        monkeypatch.setattr(orchestrator, "_load_close_price", _raise_price)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        trace_store = DecisionTraceStore(tmpdir)

        # 中止应重新抛出 ValueError（保持既有错误响应行为）。
        with pytest.raises(ValueError, match="行情缺失"):
            _run(
                store=store,
                notifier=LogNotifier(),
                portfolio=scenario["portfolio"],
                risk_config=scenario["risk_config"],
                buy_threshold=scenario["buy_threshold"],
                trace_store=trace_store,
                data_source_type=scenario["data_source_type"],
            )

        trace = trace_store.get(SIGNAL_ID)
        assert trace is not None, "中止时仍应 best-effort 持久化 trace"

        completed = trace["completed_sections"]
        # (8.11) completed_sections 绝不含 "result"。
        assert "result" not in completed
        # (8.11) completed_sections 恰为六段的真前缀（失败点之前已完成段）。
        assert completed == SIX_SECTIONS[: len(completed)]
        assert len(completed) < len(SIX_SECTIONS)
        # 推理 / 取价均在 inference 段写入前抛出 -> 仅 run_header 完成。
        assert completed == ["run_header"]

        # (8.11) 结果段记录 abort_reason，不含成功决策字段。
        res = trace["sections"]["result"]
        assert res["abort_reason"] is not None
        assert reason in res["abort_reason"]
        assert res["action"] is None
        assert res["trace_persisted"] is False

        # 中止前未产出 Decision -> 决策未落盘。
        assert store.get(SIGNAL_ID) is None


# ---------------------------------------------------------------------------
# Property 15: trace 持久化失败不影响 Decision 落盘与返回
# ---------------------------------------------------------------------------
# Feature: trading-console, Property 15: trace 持久化失败不影响 Decision 落盘与返回
# 对任意决策运行，当 Decision_Trace_Store 写入失败时，run_live_decision 仍正常返回 Decision
# 且该 Decision 已落盘到 DecisionStore（可被 get 读回），结果段标注 trace_persisted=false
# 并记录失败原因。
# Validates: Requirements 8.12
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scenario=any_decision_scenarios(), err=st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=4, max_size=20))
def test_property_15_trace_persist_failure_does_not_affect_decision(monkeypatch, scenario, err):
    _stub_io(monkeypatch, signal=scenario["signal"], price=scenario["price"])
    _patch_risk_manager(monkeypatch, kill=scenario["kill"], circuit=scenario["circuit"])

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        trace_store = DecisionTraceStore(tmpdir)

        # monkeypatch 让 trace 持久化抛错（模拟磁盘故障）。
        # 捕获被尝试持久化的 trace 对象：其 sections["result"] 与编排器内部 builder 的
        # 结果段为同一引用，编排器在 except 中回填 trace_persisted/trace_persist_error，
        # 故 _run 返回后可借此对象观测结果段持久化状态（8.12）。
        captured: dict = {}

        def _boom(signal_id, trace):
            captured["trace"] = trace
            raise RuntimeError(f"disk error: {err}")

        monkeypatch.setattr(trace_store, "save_if_absent", _boom)

        result = _run(
            store=store,
            notifier=LogNotifier(),
            portfolio=scenario["portfolio"],
            risk_config=scenario["risk_config"],
            buy_threshold=scenario["buy_threshold"],
            trace_store=trace_store,
            should_exit=scenario["should_exit"],
            halted=scenario["halted"],
            data_source_type=scenario["data_source_type"],
        )

        # (8.12) 持久化失败不影响 Decision 返回。
        decision = result["decision"]
        assert decision["action"] in ("buy", "sell", "hold")
        assert decision["signal_id"] == SIGNAL_ID

        # (8.12) Decision 已落盘，可被 DecisionStore.get 读回，字段一致。
        reloaded = store.get(SIGNAL_ID)
        assert reloaded is not None
        assert asdict(reloaded) == decision

        # trace 未落盘（save 抛错）。
        assert trace_store.get(SIGNAL_ID) is None

        # (8.12) 结果段标注 trace_persisted=false 并记录失败原因。
        assert "trace" in captured, "持久化失败前应已尝试写盘（携带结果段的 trace）"
        result_section = captured["trace"]["sections"]["result"]
        assert result_section["trace_persisted"] is False
        assert result_section["trace_persist_error"] is not None
        assert err in result_section["trace_persist_error"]
        # 成功决策字段仍在结果段（持久化失败不抹除决策信息）。
        assert result_section["action"] == decision["action"]
        assert result_section["signal_id"] == SIGNAL_ID
        assert result_section["abort_reason"] is None
