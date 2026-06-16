"""
LiveSignalOrchestrator 编排器单元/示例测试（交易操作台特性，任务 5.2）。

覆盖需求 1.3 / 1.4 / 3.2 / 5.4 / 7.1 / 7.2：

- 1.3/1.4：编排经 CNN 推理 + SignalService 产出合法 Decision（buy/sell/hold），
  含 volume/price/signal/reason；返回 {decision, risk_detail, idempotent_hit}。
- 3.2：风控拦截则产出 hold 且 reason 含拦截原因。
- 5.4：upload / pull 两数据源分支在行情就绪后取价路径汇合，行为完全一致。
- 5.3（关联）：决策日行情缺失抛 ValueError。
- 7.1/7.2：无券商下单路径——编排仅落盘 + 提醒，不调用任何下单接口。

外部 I/O 全部桩化：`predict_cnn_signals` 与 `_load_close_price` 注入确定 signal/price，
`DecisionStore` 用 `tmp_path`，`Notifier` 用 LogNotifier 记录提醒次数。不依赖外部网络。
"""

from __future__ import annotations

from datetime import date, datetime, time

import polars as pl
import pytest

from aitrade.live import orchestrator
from aitrade.live.decision import DecisionStore
from aitrade.live.decision_instant import DecisionInstant
from aitrade.live.notifier import LogNotifier
from aitrade.live.orchestrator import run_live_decision
from aitrade.live.risk import RiskConfig
from aitrade.live.signal_service import PortfolioSnapshot


TRADE_DATE = date(2026, 6, 9)
# as_of 取决策日收盘后（15:05），bar_freq=1d → Decision_Bar = 当日（与历史日频等价）。
INSTANT = DecisionInstant(datetime.combine(TRADE_DATE, time(15, 5)), "1d")
VT_SYMBOL = "000001.SZSE"
SCHEME = "eod_buy_v1"
MODEL = "测试"

# RiskInspector 同序的 5 项检查标识（与 RiskManager 一致）。
EXPECTED_CHECKS = [
    "kill_switch_or_circuit",
    "blacklist",
    "halted",
    "max_total_position",
    "max_single_position",
]


def _signal_frame(signal: float, *, vt_symbol: str = VT_SYMBOL,
                  trade_date: date = TRADE_DATE) -> pl.DataFrame:
    """构造 predict_cnn_signals 同 schema 的桩输出：[datetime, vt_symbol, signal]。"""
    return pl.DataFrame(
        {
            "datetime": [datetime.combine(trade_date, datetime.min.time())],
            "vt_symbol": [vt_symbol],
            "signal": [float(signal)],
        }
    )


def _stub_io(monkeypatch, *, signal: float, price: float) -> None:
    """桩化外部 I/O：CNN 推理返回确定 signal、取价返回确定 price。"""
    monkeypatch.setattr(
        orchestrator,
        "predict_cnn_signals",
        lambda **kwargs: _signal_frame(signal),
    )
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
    risk_config: RiskConfig | None = None,
    buy_threshold: float = 0.6,
    should_exit: bool = False,
    halted: bool = False,
    on_progress=None,
) -> dict:
    return run_live_decision(
        model_name=MODEL,
        vt_symbol=VT_SYMBOL,
        scheme_name=SCHEME,
        instant=INSTANT,
        portfolio=portfolio,
        buy_threshold=buy_threshold,
        risk_config=risk_config or RiskConfig(
            max_total_position_ratio=0.95, max_single_position_ratio=0.95
        ),
        store=store,
        notifier=notifier,
        model_version="v3",
        should_exit=should_exit,
        halted=halted,
        on_progress=on_progress,
    )


# ---------------------------------------------------------------------------
# 需求 1.3 / 1.4：buy 结果路径
# ---------------------------------------------------------------------------
def test_run_live_decision_buy(tmp_path, monkeypatch) -> None:
    _stub_io(monkeypatch, signal=0.72, price=10.0)
    store = DecisionStore(tmp_path)
    notifier = LogNotifier()
    pf = PortfolioSnapshot(portfolio_value=100000, total_position_value=0, current_position=0)

    result = _run(store=store, notifier=notifier, portfolio=pf)
    decision = result["decision"]

    assert decision["action"] == "buy"
    # 95% * 100000 / 10 = 9500（min_volume=100 取整）
    assert decision["volume"] == 9500
    assert decision["price"] == 10.0
    assert decision["signal"] == 0.72
    assert decision["reason"] != ""
    # 必填字段齐全（需求 1.4）
    for field in ("action", "volume", "price", "signal", "reason"):
        assert field in decision
    # 首次触发 → 走风控 → 5 项明细齐全、非幂等命中
    assert [r["check"] for r in result["risk_detail"]] == EXPECTED_CHECKS
    assert result["idempotent_hit"] is False
    # buy 触发提醒一次
    assert len(notifier.sent) == 1


# ---------------------------------------------------------------------------
# 需求 1.4：hold 结果路径（信号未达阈值）
# ---------------------------------------------------------------------------
def test_run_live_decision_hold(tmp_path, monkeypatch) -> None:
    _stub_io(monkeypatch, signal=0.50, price=10.0)
    store = DecisionStore(tmp_path)
    notifier = LogNotifier()
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)

    result = _run(store=store, notifier=notifier, portfolio=pf)
    decision = result["decision"]

    assert decision["action"] == "hold"
    assert decision["volume"] == 0
    assert decision["reason"] != ""
    # 观望不提醒
    assert len(notifier.sent) == 0
    # 回归：未达阈值不走买入风控（明细为空），但首次触发不得误判为幂等命中
    assert result["risk_detail"] == []
    assert result["idempotent_hit"] is False


# ---------------------------------------------------------------------------
# 需求 1.4：sell 结果路径（持仓且到出场条件）
# ---------------------------------------------------------------------------
def test_run_live_decision_sell(tmp_path, monkeypatch) -> None:
    _stub_io(monkeypatch, signal=0.30, price=10.0)
    store = DecisionStore(tmp_path)
    notifier = LogNotifier()
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=900)

    result = _run(store=store, notifier=notifier, portfolio=pf, should_exit=True)
    decision = result["decision"]

    assert decision["action"] == "sell"
    assert decision["volume"] == 900
    assert decision["reason"] != ""
    # sell 触发提醒一次
    assert len(notifier.sent) == 1
    # 回归：出场路径不走买入风控（明细为空），但首次触发不得误判为幂等命中
    assert result["risk_detail"] == []
    assert result["idempotent_hit"] is False


# ---------------------------------------------------------------------------
# 需求 3.2：仓位上限 → 缩量买入且风控明细通过
# ---------------------------------------------------------------------------
def test_run_live_decision_caps_buy_to_single_position_limit(tmp_path, monkeypatch) -> None:
    _stub_io(monkeypatch, signal=0.9, price=10.0)
    store = DecisionStore(tmp_path)
    notifier = LogNotifier()
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)
    # 单票上限仅 10% → 原目标 95000 会被裁剪到 10000。
    risk_config = RiskConfig(max_total_position_ratio=0.95, max_single_position_ratio=0.10)

    result = _run(store=store, notifier=notifier, portfolio=pf, risk_config=risk_config)
    decision = result["decision"]

    assert decision["action"] == "buy"
    assert decision["volume"] == 1000
    assert decision["reason"] == "概率达标且通过风控"
    # 明细中单票上限项按裁剪后的买入额通过。
    single_rec = next(
        r for r in result["risk_detail"] if r["check"] == "max_single_position"
    )
    assert single_rec["passed"] is True
    assert "拟新增后 10000" in single_rec["detail"]
    assert len(notifier.sent) == 1


# ---------------------------------------------------------------------------
# 需求 5.4：upload / pull 两分支取价路径汇合后行为一致
# ---------------------------------------------------------------------------
def test_upload_and_pull_paths_converge_identically(tmp_path, monkeypatch) -> None:
    """编排器不感知数据源；行情就绪后（同 signal/price）两分支决策完全一致。"""
    # upload 分支：数据经 /api/alpha/bar-data/import 就绪
    _stub_io(monkeypatch, signal=0.72, price=10.0)
    store_upload = DecisionStore(tmp_path / "upload")
    res_upload = _run(
        store=store_upload, notifier=LogNotifier(),
        portfolio=PortfolioSnapshot(portfolio_value=100000, current_position=0),
    )

    # pull 分支：数据经 datasource/Tushare 就绪（产出相同 signal/price）
    _stub_io(monkeypatch, signal=0.72, price=10.0)
    store_pull = DecisionStore(tmp_path / "pull")
    res_pull = _run(
        store=store_pull, notifier=LogNotifier(),
        portfolio=PortfolioSnapshot(portfolio_value=100000, current_position=0),
    )

    # 决策核心字段（除 created_at 时间戳外）一致
    du, dp = res_upload["decision"], res_pull["decision"]
    for field in ("signal_id", "action", "vt_symbol", "volume", "price", "signal", "reason"):
        assert du[field] == dp[field]
    assert res_upload["risk_detail"] == res_pull["risk_detail"]


# ---------------------------------------------------------------------------
# 需求 5.3：决策日行情缺失抛 ValueError
# ---------------------------------------------------------------------------
def test_missing_quote_raises_value_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        orchestrator,
        "predict_cnn_signals",
        lambda **kwargs: _signal_frame(0.72),
    )

    def _raise(vt_symbol, instant):
        raise ValueError(f"决策时刻 {instant.as_of.isoformat()} 之前的 {vt_symbol} 行情缺失")

    monkeypatch.setattr(orchestrator, "_load_close_price", _raise)
    store = DecisionStore(tmp_path)
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)

    with pytest.raises(ValueError, match="行情缺失"):
        _run(store=store, notifier=LogNotifier(), portfolio=pf)


def test_missing_signal_raises_value_error(tmp_path, monkeypatch) -> None:
    """推理结果不含决策日目标标的的信号时抛 ValueError（信号缺失）。"""
    # 推理返回空 DataFrame（无目标标的当日信号）
    empty = pl.DataFrame(
        {"datetime": [], "vt_symbol": [], "signal": []},
        schema={"datetime": pl.Datetime, "vt_symbol": pl.Utf8, "signal": pl.Float64},
    )
    monkeypatch.setattr(orchestrator, "predict_cnn_signals", lambda **kwargs: empty)
    monkeypatch.setattr(
        orchestrator, "_load_close_price",
        lambda vt_symbol, instant: (10.0, "d"),
    )
    store = DecisionStore(tmp_path)
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)

    with pytest.raises(ValueError, match="信号缺失"):
        _run(store=store, notifier=LogNotifier(), portfolio=pf)


# ---------------------------------------------------------------------------
# 需求 2.x（关联）：幂等命中时不重复提醒、idempotent_hit=True、risk_detail 为空
# ---------------------------------------------------------------------------
def test_idempotent_second_trigger(tmp_path, monkeypatch) -> None:
    _stub_io(monkeypatch, signal=0.72, price=10.0)
    store = DecisionStore(tmp_path)
    notifier = LogNotifier()
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)

    first = _run(store=store, notifier=notifier, portfolio=pf)
    assert first["idempotent_hit"] is False
    assert len(notifier.sent) == 1

    second = _run(store=store, notifier=notifier, portfolio=pf)
    # 同一 Decision 返回，未重新走风控，不再提醒
    assert second["decision"]["signal_id"] == first["decision"]["signal_id"]
    assert second["decision"]["action"] == first["decision"]["action"]
    assert second["idempotent_hit"] is True
    assert second["risk_detail"] == []
    assert len(notifier.sent) == 1


def test_idempotent_second_trigger_on_hold_path(tmp_path, monkeypatch) -> None:
    """hold 路径（不产生风控明细）下幂等判定仍正确：首次 False、二次 True。

    回归保护：幂等命中曾以 `inspector.records == []` 反推，导致所有非买入
    路径的首次决策被误标为幂等命中。
    """
    _stub_io(monkeypatch, signal=0.50, price=10.0)
    store = DecisionStore(tmp_path)
    notifier = LogNotifier()
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)

    first = _run(store=store, notifier=notifier, portfolio=pf)
    assert first["decision"]["action"] == "hold"
    assert first["idempotent_hit"] is False

    second = _run(store=store, notifier=notifier, portfolio=pf)
    assert second["decision"]["signal_id"] == first["decision"]["signal_id"]
    assert second["idempotent_hit"] is True


# ---------------------------------------------------------------------------
# 需求 7.1 / 7.2：无券商下单路径——任意结果均不调用任何下单接口
# ---------------------------------------------------------------------------
def test_no_broker_submission_path(tmp_path, monkeypatch) -> None:
    """编排模块不 import 任何券商网关/下单模块；产出仅限落盘 + 提醒。

    用 AST 只扫描 import 语句（而非全文，避免误伤安全声明文档串中的「不下单」negation）。
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(orchestrator))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported_modules.append(base)
            imported_modules.extend(f"{base}.{alias.name}" for alias in node.names)

    forbidden = ["broker", "submit_order", "place_order", "send_order", "gateway", "order"]
    for mod in imported_modules:
        lowered = mod.lower()
        for token in forbidden:
            assert token not in lowered, f"编排器不应 import 下单相关模块: {mod}"

    # 行为层面：一次 buy 决策只产生落盘 + 一次提醒，无其它副作用通道。
    _stub_io(monkeypatch, signal=0.72, price=10.0)
    store = DecisionStore(tmp_path)
    notifier = LogNotifier()
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)
    result = _run(store=store, notifier=notifier, portfolio=pf)

    assert result["decision"]["action"] == "buy"
    # 仅落盘 1 条 + 提醒 1 次
    assert len(store.list_ids()) == 1
    assert len(notifier.sent) == 1


# ---------------------------------------------------------------------------
# on_progress 透传：推理段 10~70%，编排 80%，完成 100%
# ---------------------------------------------------------------------------
def test_on_progress_callback_invoked(tmp_path, monkeypatch) -> None:
    _stub_io(monkeypatch, signal=0.72, price=10.0)
    store = DecisionStore(tmp_path)
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)
    events: list[tuple[float, str]] = []

    _run(
        store=store, notifier=LogNotifier(), portfolio=pf,
        on_progress=lambda p, m: events.append((p, m)),
    )

    pcts = [p for p, _ in events]
    assert pcts, "应至少有一次进度回调"
    assert max(pcts) == 100  # 完成
    assert any(p == 80 for p in pcts)  # 编排段
    assert all(0 <= p <= 100 for p in pcts)


# ---------------------------------------------------------------------------
# 任务 1.3：signal_fn 注入 — 自定义信号源参数化
# ---------------------------------------------------------------------------
def test_custom_signal_fn_is_called_and_predict_cnn_signals_is_not(
    tmp_path, monkeypatch
) -> None:
    """注入 signal_fn 时：自定义函数被调用，模块全局 predict_cnn_signals 不被调用。

    哨兵断言：把 orchestrator.predict_cnn_signals 替换为会报错的哨兵，确认注入路径
    完全绕过默认 CNN 推理。
    """
    monkeypatch.setattr(
        orchestrator,
        "_load_close_price",
        lambda vt_symbol, instant: (10.0, "d"),
    )

    # 哨兵：一旦被调用就 raise，保证注入路径不触碰全局 predict_cnn_signals。
    def _sentinel(**kwargs):
        raise AssertionError("注入 signal_fn 后不应调用全局 predict_cnn_signals")

    monkeypatch.setattr(orchestrator, "predict_cnn_signals", _sentinel)

    custom_called_with: list[dict] = []

    def _custom_signal_fn(**kwargs):
        custom_called_with.append(kwargs)
        return _signal_frame(0.75)

    store = DecisionStore(tmp_path)
    notifier = LogNotifier()
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)

    result = run_live_decision(
        model_name=MODEL,
        vt_symbol=VT_SYMBOL,
        scheme_name=SCHEME,
        instant=INSTANT,
        portfolio=pf,
        buy_threshold=0.6,
        risk_config=RiskConfig(max_total_position_ratio=0.95, max_single_position_ratio=0.95),
        store=store,
        notifier=notifier,
        model_version="v3",
        signal_fn=_custom_signal_fn,
    )

    # 自定义函数被调用一次，且以正确的 kwargs 调用（含 model_name/start/end）。
    assert len(custom_called_with) == 1
    call_kwargs = custom_called_with[0]
    assert call_kwargs["model_name"] == MODEL
    assert "start" in call_kwargs
    assert "end" in call_kwargs

    # 决策结果正常产出（signal=0.75 > threshold=0.6 → buy）。
    assert result["decision"]["action"] == "buy"
    assert result["decision"]["signal"] == 0.75


def test_default_path_unaffected_when_signal_fn_is_none(tmp_path, monkeypatch) -> None:
    """不注入 signal_fn（或显式传 None）时，默认路径（monkeypatched predict_cnn_signals）照常。

    确认：signal_fn=None 与不传 signal_fn 均走模块全局 predict_cnn_signals（已由
    monkeypatch 桩化），与任务 1.3 前的行为完全一致。
    """
    _stub_io(monkeypatch, signal=0.72, price=10.0)
    store = DecisionStore(tmp_path)
    notifier = LogNotifier()
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)

    # 显式传 signal_fn=None
    result = run_live_decision(
        model_name=MODEL,
        vt_symbol=VT_SYMBOL,
        scheme_name=SCHEME,
        instant=INSTANT,
        portfolio=pf,
        buy_threshold=0.6,
        risk_config=RiskConfig(max_total_position_ratio=0.95, max_single_position_ratio=0.95),
        store=store,
        notifier=notifier,
        model_version="v3",
        signal_fn=None,
    )
    assert result["decision"]["action"] == "buy"
    assert result["idempotent_hit"] is False
