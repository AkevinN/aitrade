"""
Wave 2c 可观测性字段测试（task-scheduler-observability 任务 5）。

覆盖：
- TSO-5 属性：trigger_source 贯穿且旧数据兼容
- TSO-6 属性：通知字段为实测值（notified / notify_ok）
- TSO-8 属性：记录面脱敏（不含凭证）
- elapsed_ms >= 0 且在合理范围

注释规范（Hypothesis 属性测试）：
# Feature: task-scheduler-observability, Property TSO-n: ...
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, time

import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aitrade.live import orchestrator as orchestrator_mod
from aitrade.live.decision import Decision, DecisionStore
from aitrade.live.decision_instant import DecisionInstant, make_signal_id
from aitrade.live.decision_trace import DecisionTraceStore
from aitrade.live.legacy_migration import migrate_decision
from aitrade.live.notifier import LogNotifier
from aitrade.live.orchestrator import run_live_decision
from aitrade.live.rebalance import run_rebalance_decision
from aitrade.live.rebalance_decision import RebalanceStore
from aitrade.live.risk import RiskConfig
from aitrade.live.signal_service import PortfolioSnapshot


# ---------------------------------------------------------------------------
# 共用常量
# ---------------------------------------------------------------------------

TRADE_DATE = date(2026, 6, 9)
INSTANT = DecisionInstant(datetime.combine(TRADE_DATE, time(15, 5)), "1d")
VT_SYMBOL = "000001.SZSE"
SCHEME = "eod_buy_v1"
MODEL = "测试"
MODEL_VERSION = "v3"
PLAN_NAME = "etf_test_plan"
SIGNAL_SOURCE = "stub_src_tso"
PORTFOLIO_ID = "p_test_tso"
CAPITAL = 1_000_000.0

SIGNAL_ID = make_signal_id(
    datetime.combine(TRADE_DATE, datetime.min.time()), "1d", SCHEME, MODEL_VERSION
)


# ---------------------------------------------------------------------------
# Stub / 桩辅助
# ---------------------------------------------------------------------------


class _FalseNotifier:
    """永远返回 False 的通知桩（模拟发送失败）。"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, title: str, message: str) -> bool:
        self.sent.append((title, message))
        return False


class _TrueNotifier:
    """永远返回 True 的通知桩（正常）。"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, title: str, message: str) -> bool:
        self.sent.append((title, message))
        return True


def _signal_frame(signal: float, *, vt_symbol: str = VT_SYMBOL) -> pl.DataFrame:
    """构造 predict_cnn_signals 同 schema 的桩输出。"""
    return pl.DataFrame({
        "datetime": [datetime.combine(TRADE_DATE, datetime.min.time())],
        "vt_symbol": [vt_symbol],
        "signal": [float(signal)],
    })


def _stub_io_orch(monkeypatch, *, signal: float = 0.8, price: float = 10.0) -> None:
    """桩化 orchestrator 的外部 I/O。"""
    monkeypatch.setattr(orchestrator_mod, "predict_cnn_signals",
                        lambda **kw: _signal_frame(signal))
    monkeypatch.setattr(orchestrator_mod, "_load_close_price",
                        lambda vt, inst: (price, "d"))


def _run_decision(
    *,
    store: DecisionStore,
    notifier,
    trace_store: DecisionTraceStore | None = None,
    trigger_source: str = "manual",
    signal: float = 0.8,
    price: float = 10.0,
    monkeypatch,
) -> dict:
    _stub_io_orch(monkeypatch, signal=signal, price=price)
    portfolio = PortfolioSnapshot(portfolio_value=100_000, current_position=0)
    risk_config = RiskConfig(max_total_position_ratio=0.95, max_single_position_ratio=0.95)
    return run_live_decision(
        model_name=MODEL,
        vt_symbol=VT_SYMBOL,
        scheme_name=SCHEME,
        instant=INSTANT,
        portfolio=portfolio,
        buy_threshold=0.6,
        risk_config=risk_config,
        store=store,
        notifier=notifier,
        model_version=MODEL_VERSION,
        trace_store=trace_store,
        trigger_source=trigger_source,
    )


# Rebalance 路径辅助

class _StubProvider:
    """确定性信号源桩。"""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def predict(self, start: date, end: date, on_progress=None) -> pl.DataFrame:
        if not self._rows:
            return pl.DataFrame(
                {"datetime": [], "vt_symbol": [], "signal": [], "close": []},
                schema={
                    "datetime": pl.Datetime,
                    "vt_symbol": pl.Utf8,
                    "signal": pl.Float64,
                    "close": pl.Float64,
                },
            )
        return pl.DataFrame(self._rows).with_columns(pl.col("datetime").cast(pl.Datetime))


def _make_bar_dt(d: date = TRADE_DATE) -> datetime:
    return datetime.combine(d, time(15, 0))


def _register_stub(source_name: str, provider: _StubProvider) -> None:
    from aitrade.backtest.registry import register_signal_source
    register_signal_source(source_name, lambda params: provider)


def _stub_risk_manager_pass():
    from aitrade.live.portfolio_risk import PortfolioRiskVerdict

    class _StubMgr:
        def evaluate(self, portfolio_id: str, *, portfolio_value: float, as_of: date):
            return PortfolioRiskVerdict(
                allow_buy=True, buy_factor=1.0, broken=False,
                records=[
                    {"check": "circuit", "passed": True, "detail": "未熔断"},
                    {"check": "drawdown", "passed": True, "detail": "回撤0%"},
                    {"check": "trend", "passed": True, "detail": "正常"},
                ],
            )
    return _StubMgr()


def _signal_rows_rb(symbols: list[tuple[str, float, float]], bar_dt: datetime | None = None) -> list[dict]:
    dt = bar_dt or _make_bar_dt()
    return [
        {"datetime": dt, "vt_symbol": sym, "signal": sig, "close": price}
        for sym, sig, price in symbols
    ]


def _run_rebalance(
    *,
    store: RebalanceStore,
    book,
    notifier,
    trigger_source: str = "manual",
    source_name: str = SIGNAL_SOURCE,
    rows: list[dict] | None = None,
) -> dict:
    provider = _StubProvider(rows or _signal_rows_rb([("000001.SZSE", 0.9, 10.0)]))
    _register_stub(source_name, provider)
    return run_rebalance_decision(
        plan_name=PLAN_NAME,
        signal_source=source_name,
        signal_params={},
        strategy_params={"top_k": 1, "min_volume": 100},
        portfolio_id=PORTFOLIO_ID,
        instant=INSTANT,
        capital=CAPITAL,
        rebalance_store=store,
        position_book=book,
        risk_manager=_stub_risk_manager_pass(),
        notifier=notifier,
        trigger_source=trigger_source,
    )


# ===========================================================================
# TSO-5：trigger_source 贯穿且旧数据兼容
# ===========================================================================

# Feature: task-scheduler-observability, Property TSO-5: trigger_source 贯穿且旧数据兼容
# 对任意经调度器触发的决策/调仓，落盘 trigger_source=="scheduler"；
# 经手动 API 触发的为 "manual"；读取缺该字段的旧 JSON 不抛错且值为 ""。
# Validates: Requirements 4.1, 4.2, 4.4

def test_tso5_trigger_source_scheduler_in_decision(tmp_path, monkeypatch) -> None:
    """调度器路径（trigger_source="scheduler"）落盘后 Decision.trigger_source == "scheduler"。"""
    store = DecisionStore(tmp_path)
    result = _run_decision(
        store=store, notifier=LogNotifier(), trigger_source="scheduler", monkeypatch=monkeypatch
    )
    decision_dict = result["decision"]
    assert decision_dict["trigger_source"] == "scheduler"
    # 从磁盘读回验证持久化
    reloaded = store.get(decision_dict["signal_id"])
    assert reloaded is not None
    assert reloaded.trigger_source == "scheduler"


def test_tso5_trigger_source_manual_in_decision(tmp_path, monkeypatch) -> None:
    """手动路径（trigger_source="manual"）落盘后 Decision.trigger_source == "manual"。"""
    store = DecisionStore(tmp_path)
    result = _run_decision(
        store=store, notifier=LogNotifier(), trigger_source="manual", monkeypatch=monkeypatch
    )
    reloaded = store.get(result["decision"]["signal_id"])
    assert reloaded is not None
    assert reloaded.trigger_source == "manual"


def test_tso5_trigger_source_scheduler_in_rebalance(tmp_path) -> None:
    """调度器路径调仓：RebalanceDecision.trigger_source == "scheduler"。"""
    from aitrade.live.position_book import PositionBook
    store = RebalanceStore(tmp_path / "rb")
    book = PositionBook(tmp_path / "pb")
    result = _run_rebalance(store=store, book=book, notifier=LogNotifier(), trigger_source="scheduler")
    decision_dict = result["decision"]
    assert decision_dict["trigger_source"] == "scheduler"
    # 从磁盘读回
    reloaded = store.get(decision_dict["signal_id"])
    assert reloaded is not None
    assert reloaded.trigger_source == "scheduler"


def test_tso5_trigger_source_manual_in_rebalance(tmp_path) -> None:
    """手动路径调仓：RebalanceDecision.trigger_source == "manual"。"""
    from aitrade.live.position_book import PositionBook
    store = RebalanceStore(tmp_path / "rb")
    book = PositionBook(tmp_path / "pb")
    result = _run_rebalance(store=store, book=book, notifier=LogNotifier(), trigger_source="manual")
    reloaded = store.get(result["decision"]["signal_id"])
    assert reloaded is not None
    assert reloaded.trigger_source == "manual"


def test_tso5_old_decision_json_missing_trigger_source_migrates_to_empty(tmp_path) -> None:
    """旧 Decision JSON 缺 trigger_source → migrate_decision 补 "" 不抛错。"""
    old_json = {
        "signal_id": "2026-06-09:eod_buy_v1:model@v3",
        "decision_bar_dt": "2026-06-09T15:00:00",
        "as_of": "2026-06-09T15:05:00",
        "bar_freq": "1d",
        "scheme": "eod_buy_v1",
        "action": "hold",
        "vt_symbol": "000001.SZSE",
        "volume": 0,
        "price": None,
        "signal": 0.5,
        "reason": "观望",
        "created_at": "2026-06-09T15:05:01",
    }
    migrated = migrate_decision(old_json)
    assert "trigger_source" in migrated
    assert migrated["trigger_source"] == ""
    # 可正常构造 Decision（不抛 TypeError）
    d = Decision(**migrated)
    assert d.trigger_source == ""


def test_tso5_old_rebalance_json_missing_new_fields_reads_ok(tmp_path) -> None:
    """旧 RebalanceDecision JSON 缺 trigger_source/elapsed_ms/notify_ok → 读取不抛错。"""
    from aitrade.live.rebalance_decision import _decision_from_dict

    old_json = {
        "signal_id": "rule_test_id",
        "decision_bar_dt": "2026-06-09T15:00:00",
        "as_of": "2026-06-09T15:05:00",
        "bar_freq": "1d",
        "scheme": "rule:test",
        "portfolio_id": "p1",
        "items": [],
        "target_portfolio": {},
        "risk_summary": [],
        "status": "proposed",
        "created_at": "2026-06-09T15:05:01",
        "confirmed_at": "",
    }
    rd = _decision_from_dict(old_json)
    assert rd.trigger_source == ""
    assert rd.elapsed_ms is None
    assert rd.notify_ok is None


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(source=st.sampled_from(["scheduler", "manual"]))
def test_tso5_property_trigger_source_persisted_consistently(source, monkeypatch):
    """属性测试：任意合法触发来源经编排器落盘后从磁盘读回一致（不改变值）。"""
    # Feature: task-scheduler-observability, Property TSO-5
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        result = _run_decision(
            store=store, notifier=LogNotifier(),
            trigger_source=source, monkeypatch=monkeypatch,
        )
        sid = result["decision"]["signal_id"]
        reloaded = store.get(sid)
        assert reloaded is not None
        assert reloaded.trigger_source == source


# ===========================================================================
# TSO-6：通知字段为实测值
# ===========================================================================

# Feature: task-scheduler-observability, Property TSO-6: 通知字段为实测值
# 对任意 Notifier.send 返回值 b ∈ {True, False}，落盘的 notified（trace）/
# notify_ok（调仓）等于 b；未尝试发送（hold/幂等命中）时调仓侧为 None。
# Validates: Requirements 5.1

def test_tso6_notified_true_in_trace_when_send_returns_true(tmp_path, monkeypatch) -> None:
    """send 返回 True → trace.result.notified == True。"""
    store = DecisionStore(tmp_path)
    trace_store = DecisionTraceStore(tmp_path)
    notifier = _TrueNotifier()

    result = _run_decision(
        store=store, notifier=notifier, trace_store=trace_store,
        trigger_source="manual", signal=0.8, monkeypatch=monkeypatch,
    )
    decision_dict = result["decision"]
    assert decision_dict["action"] == "buy"  # 确认走了 buy 路径

    trace = trace_store.get(decision_dict["signal_id"])
    assert trace is not None
    assert trace["sections"]["result"]["notified"] is True


def test_tso6_notified_false_in_trace_when_send_returns_false(tmp_path, monkeypatch) -> None:
    """send 返回 False → trace.result.notified == False（实测值，非逻辑预期）。"""
    store = DecisionStore(tmp_path)
    trace_store = DecisionTraceStore(tmp_path)
    notifier = _FalseNotifier()

    result = _run_decision(
        store=store, notifier=notifier, trace_store=trace_store,
        trigger_source="manual", signal=0.8, monkeypatch=monkeypatch,
    )
    decision_dict = result["decision"]
    assert decision_dict["action"] == "buy"

    trace = trace_store.get(decision_dict["signal_id"])
    assert trace is not None
    assert trace["sections"]["result"]["notified"] is False


def test_tso6_notified_false_in_trace_for_hold_path(tmp_path, monkeypatch) -> None:
    """hold 路径未发送通知 → trace.result.notified == False（未尝试发送）。"""
    store = DecisionStore(tmp_path)
    trace_store = DecisionTraceStore(tmp_path)
    notifier = _TrueNotifier()

    result = _run_decision(
        store=store, notifier=notifier, trace_store=trace_store,
        signal=0.3,  # 低于 buy_threshold=0.6 → hold
        monkeypatch=monkeypatch,
    )
    decision_dict = result["decision"]
    assert decision_dict["action"] == "hold"
    assert len(notifier.sent) == 0  # 确认未发送

    trace = trace_store.get(decision_dict["signal_id"])
    assert trace is not None
    assert trace["sections"]["result"]["notified"] is False


def test_tso6_notify_ok_true_in_rebalance_when_send_returns_true(tmp_path) -> None:
    """调仓路径 send 返回 True → RebalanceDecision.notify_ok == True。"""
    from aitrade.live.position_book import PositionBook
    store = RebalanceStore(tmp_path / "rb")
    book = PositionBook(tmp_path / "pb")
    notifier = _TrueNotifier()

    result = _run_rebalance(store=store, book=book, notifier=notifier)
    decision_dict = result["decision"]
    assert decision_dict["notify_ok"] is True

    reloaded = store.get(decision_dict["signal_id"])
    assert reloaded is not None
    assert reloaded.notify_ok is True


def test_tso6_notify_ok_false_in_rebalance_when_send_returns_false(tmp_path) -> None:
    """调仓路径 send 返回 False → RebalanceDecision.notify_ok == False。"""
    from aitrade.live.position_book import PositionBook
    store = RebalanceStore(tmp_path / "rb")
    book = PositionBook(tmp_path / "pb")
    notifier = _FalseNotifier()

    result = _run_rebalance(store=store, book=book, notifier=notifier)
    decision_dict = result["decision"]
    assert decision_dict["notify_ok"] is False

    reloaded = store.get(decision_dict["signal_id"])
    assert reloaded is not None
    assert reloaded.notify_ok is False


def test_tso6_idempotent_hit_returns_first_run_notify_ok(tmp_path) -> None:
    """幂等命中返回的 decision 的 notify_ok 等于首次运行落盘的值（True）。"""
    from aitrade.live.position_book import PositionBook
    store = RebalanceStore(tmp_path / "rb")
    book = PositionBook(tmp_path / "pb")

    # 第一次：用永远返回 True 的通知桩，确保首次 notify_ok 落盘为 True
    first_notifier = _TrueNotifier()
    first = _run_rebalance(store=store, book=book, notifier=first_notifier,
                           source_name="src_idem_test")
    assert first["idempotent_hit"] is False
    assert first["decision"]["notify_ok"] is True  # 首次落盘值

    # 第二次：幂等命中，send 不再被调用
    second_notifier = _TrueNotifier()
    second = _run_rebalance(store=store, book=book, notifier=second_notifier,
                            source_name="src_idem_test")
    assert second["idempotent_hit"] is True
    # 幂等命中返回旧决策原样：notify_ok 为首次运行的实测值，而非 None
    assert second["decision"]["notify_ok"] is True
    # 幂等命中路径未尝试再次发送通知
    assert len(second_notifier.sent) == 0


def test_tso6_single_symbol_idempotent_hit_notified_regression(tmp_path, monkeypatch) -> None:
    """单标的路径幂等命中 notified 回归测试。

    首次 run_live_decision（stub send 返回 True，notified=True 落盘到 trace）→
    同 signal_id 二次调用幂等命中 → trace 不被覆盖（save_if_absent 语义）且
    返回的 result 里 idempotent_hit=True、顶层 notified 缺失（本次未发送）。
    """
    store = DecisionStore(tmp_path)
    trace_store = DecisionTraceStore(tmp_path)

    # 第一次：用 _TrueNotifier，send 返回 True，trace 落盘 notified=True
    first_notifier = _TrueNotifier()
    first = _run_decision(
        store=store, notifier=first_notifier, trace_store=trace_store,
        monkeypatch=monkeypatch,
    )
    assert first["idempotent_hit"] is False
    signal_id = first["decision"]["signal_id"]

    # trace 首次落盘：notified=True（send 实测值）
    trace_after_first = trace_store.get(signal_id)
    assert trace_after_first is not None
    assert trace_after_first["sections"]["result"]["notified"] is True

    # 第二次：幂等命中，send 不再被调用
    second_notifier = _TrueNotifier()
    second = _run_decision(
        store=store, notifier=second_notifier, trace_store=trace_store,
        monkeypatch=monkeypatch,
    )
    assert second["idempotent_hit"] is True
    # 本次未调用 send
    assert len(second_notifier.sent) == 0
    # trace 不被覆盖（save_if_absent 语义）：仍为首次落盘的 notified=True
    trace_after_second = trace_store.get(signal_id)
    assert trace_after_second is not None
    assert trace_after_second["sections"]["result"]["notified"] is True
    # 返回 dict 中无顶层 notified 键（orchestrator 不在 return dict 里暴露），
    # idempotent_hit 正确为 True
    assert "notified" not in second
    assert second["idempotent_hit"] is True


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(send_result=st.booleans())
def test_tso6_property_notified_equals_actual_send_return(send_result, monkeypatch):
    """属性测试：trace.notified 等于实测 send 返回值（buy 路径）。"""
    # Feature: task-scheduler-observability, Property TSO-6

    class _ControlledNotifier:
        def __init__(self, ret: bool) -> None:
            self._ret = ret
        def send(self, title: str, message: str) -> bool:
            return self._ret

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        trace_store = DecisionTraceStore(tmpdir)
        notifier = _ControlledNotifier(send_result)

        result = _run_decision(
            store=store, notifier=notifier, trace_store=trace_store,
            signal=0.8,  # 确保 buy
            monkeypatch=monkeypatch,
        )
        if result["decision"]["action"] == "buy":
            trace = trace_store.get(result["decision"]["signal_id"])
            assert trace is not None
            assert trace["sections"]["result"]["notified"] is send_result


# ===========================================================================
# elapsed_ms 合理范围测试
# ===========================================================================

def test_elapsed_ms_non_negative_in_trace(tmp_path, monkeypatch) -> None:
    """trace.result.elapsed_ms >= 0 且在合理上限（10 秒）内。"""
    store = DecisionStore(tmp_path)
    trace_store = DecisionTraceStore(tmp_path)

    result = _run_decision(
        store=store, notifier=LogNotifier(), trace_store=trace_store,
        monkeypatch=monkeypatch,
    )
    trace = trace_store.get(result["decision"]["signal_id"])
    assert trace is not None
    elapsed = trace["sections"]["result"]["elapsed_ms"]
    assert isinstance(elapsed, int)
    assert elapsed >= 0
    assert elapsed < 10_000  # 应在 10 秒内完成


def test_elapsed_ms_non_negative_in_rebalance(tmp_path) -> None:
    """RebalanceDecision.elapsed_ms >= 0 且在合理上限内。"""
    from aitrade.live.position_book import PositionBook
    store = RebalanceStore(tmp_path / "rb")
    book = PositionBook(tmp_path / "pb")

    result = _run_rebalance(store=store, book=book, notifier=LogNotifier())
    decision_dict = result["decision"]
    elapsed = decision_dict["elapsed_ms"]
    assert isinstance(elapsed, int)
    assert elapsed >= 0
    assert elapsed < 10_000


# ===========================================================================
# TSO-8：记录面脱敏（不含凭证值）
# ===========================================================================

# Feature: task-scheduler-observability, Property TSO-8: 记录面脱敏
# 对任意含凭证形态字符串（http webhook URL、token=、secret 键）的输入场景，
# 决策/调仓/trace 落盘文本不含这些凭证值。
# Validates: Requirements 8.1

_CREDENTIAL_TEXTS = [
    "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=abcdef123456",
    "https://oapi.dingtalk.com/robot/send?access_token=MYSECRET9999",
    "token=SECRETTOKEN_XYZ",
]


@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(cred=st.sampled_from(_CREDENTIAL_TEXTS))
def test_tso8_property_decision_trace_no_credential(cred, monkeypatch):
    """属性测试：含凭证形态参数的决策场景落盘文本不含凭证值。"""
    # Feature: task-scheduler-observability, Property TSO-8
    # 注入凭证到 reason（模拟误传凭证）但落盘文本中不应出现

    with tempfile.TemporaryDirectory() as tmpdir:
        # 注入凭证到环境变量（确保不被 run_header 泄漏）
        monkeypatch.setenv("TUSHARE_TOKEN", cred)
        monkeypatch.setenv("AITRADE_WEBHOOK_URL", cred)

        store = DecisionStore(tmpdir)
        trace_store = DecisionTraceStore(tmpdir)

        result = _run_decision(
            store=store, notifier=LogNotifier(), trace_store=trace_store,
            monkeypatch=monkeypatch,
        )
        signal_id = result["decision"]["signal_id"]

        # Decision 落盘文本不含凭证
        decision_file = store._path(signal_id)
        assert decision_file.exists()
        decision_text = decision_file.read_text(encoding="utf-8")
        assert cred not in decision_text, f"Decision 落盘含凭证: {cred!r}"

        # trace 落盘文本不含凭证
        trace = trace_store.get(signal_id)
        if trace is not None:
            trace_text = json.dumps(trace, ensure_ascii=False)
            assert cred not in trace_text, f"trace 落盘含凭证: {cred!r}"


@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(cred=st.sampled_from(_CREDENTIAL_TEXTS))
def test_tso8_property_rebalance_no_credential(cred, monkeypatch):
    """属性测试：含凭证形态参数的调仓场景落盘文本不含凭证值。"""
    # Feature: task-scheduler-observability, Property TSO-8
    from aitrade.live.position_book import PositionBook

    monkeypatch.setenv("TUSHARE_TOKEN", cred)
    monkeypatch.setenv("AITRADE_WEBHOOK_URL", cred)

    with tempfile.TemporaryDirectory() as tmpdir:
        import pathlib
        store = RebalanceStore(pathlib.Path(tmpdir) / "rb")
        book = PositionBook(pathlib.Path(tmpdir) / "pb")

        result = _run_rebalance(
            store=store, book=book, notifier=LogNotifier(),
            source_name=f"src_tso8_{abs(hash(cred)) % 1000}",
        )
        if result["decision"] is not None:
            signal_id = result["decision"]["signal_id"]
            rb_file = store._path(signal_id)
            if rb_file.exists():
                rb_text = rb_file.read_text(encoding="utf-8")
                assert cred not in rb_text, f"RebalanceDecision 落盘含凭证: {cred!r}"


# ===========================================================================
# trigger_source 在 trace result 段中可见
# ===========================================================================

def test_trigger_source_present_in_trace_result(tmp_path, monkeypatch) -> None:
    """trace.result 段包含 trigger_source 字段，且值与入参一致。"""
    store = DecisionStore(tmp_path)
    trace_store = DecisionTraceStore(tmp_path)

    result = _run_decision(
        store=store, notifier=LogNotifier(), trace_store=trace_store,
        trigger_source="scheduler", monkeypatch=monkeypatch,
    )
    trace = trace_store.get(result["decision"]["signal_id"])
    assert trace is not None
    result_section = trace["sections"]["result"]
    assert "trigger_source" in result_section
    assert result_section["trigger_source"] == "scheduler"


def test_elapsed_ms_present_in_trace_result(tmp_path, monkeypatch) -> None:
    """trace.result 段包含 elapsed_ms 字段。"""
    store = DecisionStore(tmp_path)
    trace_store = DecisionTraceStore(tmp_path)

    result = _run_decision(
        store=store, notifier=LogNotifier(), trace_store=trace_store,
        monkeypatch=monkeypatch,
    )
    trace = trace_store.get(result["decision"]["signal_id"])
    assert trace is not None
    assert "elapsed_ms" in trace["sections"]["result"]
