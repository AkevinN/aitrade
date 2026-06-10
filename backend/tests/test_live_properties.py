"""
交易操作台后端属性测试（Hypothesis）。

每条正确性属性用单个属性测试实现，`@settings(max_examples=100)`，外部 I/O
（`predict_cnn_signals`、`_load_close_price`）全部桩化以注入确定的 signal/price，
`DecisionStore` 用临时目录隔离，`Notifier` 用 `LogNotifier` 记录提醒次数。

属性见 design.md「Correctness Properties」。
"""

from __future__ import annotations

import tempfile
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aitrade.live import orchestrator
from aitrade.live.decision import DecisionStore
from aitrade.live.decision_instant import DecisionInstant
from aitrade.live.notifier import LogNotifier
from aitrade.live.orchestrator import run_live_decision
from aitrade.live.risk import RiskConfig, RiskManager
from aitrade.live.signal_service import PortfolioSnapshot

import time

from fastapi.testclient import TestClient

from aitrade.api import live as live_api
from aitrade.main import create_app


TRADE_DATE = date(2026, 6, 9)
AS_OF = datetime(2026, 6, 9, 15, 5)  # 收盘后；Decision_Bar = 当日（日频等价）
INSTANT = DecisionInstant(AS_OF, "1d")
VT_SYMBOL = "000001.SZSE"
SCHEME = "eod_buy_v1"
MODEL = "测试"
MODEL_VERSION = "v3"


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
    """桩化外部 I/O：CNN 推理返回确定 signal、取价返回确定 price。"""
    monkeypatch.setattr(
        orchestrator, "predict_cnn_signals", lambda **kwargs: _signal_frame(signal)
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_close_price",
        lambda vt_symbol, instant: (float(price), "d"),
    )


# ---------------------------------------------------------------------------
# 生成器（见 design.md「生成器要点」）
# ---------------------------------------------------------------------------
_finite = dict(allow_nan=False, allow_infinity=False)

# 信号 ∈ [0,1]（分类概率），价格 > 0。
signals = st.floats(min_value=0.0, max_value=1.0, **_finite)
prices = st.floats(min_value=0.01, max_value=10000.0, **_finite)
thresholds = st.floats(min_value=0.0, max_value=1.0, **_finite)
ratios = st.floats(min_value=0.01, max_value=1.0, **_finite)


@st.composite
def portfolios(draw) -> PortfolioSnapshot:
    """组合快照：portfolio_value > 0，current_position ∈ {0, >0}，各市值非负。"""
    portfolio_value = draw(st.floats(min_value=1.0, max_value=1e9, **_finite))
    current_position = draw(st.integers(min_value=0, max_value=100000))
    total_position_value = draw(st.floats(min_value=0.0, max_value=1e9, **_finite))
    current_symbol_value = draw(st.floats(min_value=0.0, max_value=1e9, **_finite))
    return PortfolioSnapshot(
        portfolio_value=portfolio_value,
        total_position_value=total_position_value,
        current_position=current_position,
        current_symbol_value=current_symbol_value,
    )


@st.composite
def risk_configs(draw) -> RiskConfig:
    """风控配置：随机黑名单含/不含目标标的、随机 max_*_ratio、随机 allow_when_halted。"""
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


def _run(*, store, notifier, portfolio, risk_config, buy_threshold, should_exit, halted):
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
    )


# ---------------------------------------------------------------------------
# Property 1: 决策幂等往返
# ---------------------------------------------------------------------------
# Feature: trading-console, Property 1: 决策幂等往返
# 对任意一次有效决策请求，对同一 Signal_Id（日期+方案+模型版本）第二次触发，
# 系统返回的 Decision 与首次完全相等，且 DecisionStore 不发生新增写入、
# Notifier 不再发送提醒。
# Validates: Requirements 2.1, 2.2, 2.3, 2.4
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    signal=signals,
    price=prices,
    buy_threshold=thresholds,
    portfolio=portfolios(),
    risk_config=risk_configs(),
    should_exit=st.booleans(),
    halted=st.booleans(),
)
def test_property_1_decision_idempotent_roundtrip(
    monkeypatch,
    signal,
    price,
    buy_threshold,
    portfolio,
    risk_config,
    should_exit,
    halted,
):
    _stub_io(monkeypatch, signal=signal, price=price)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        notifier = LogNotifier()

        # 统计 DecisionStore 的写入次数（验证「不发生新增写入」）。
        save_calls = {"n": 0}
        original_save = store.save

        def _counting_save(decision):
            save_calls["n"] += 1
            return original_save(decision)

        monkeypatch.setattr(store, "save", _counting_save)

        # 首次触发
        first = _run(
            store=store,
            notifier=notifier,
            portfolio=portfolio,
            risk_config=risk_config,
            buy_threshold=buy_threshold,
            should_exit=should_exit,
            halted=halted,
        )
        first_decision = first["decision"]
        saves_after_first = save_calls["n"]
        notifications_after_first = len(notifier.sent)
        ids_after_first = store.list_ids()

        # 第二次触发（同一 signal_id）
        second = _run(
            store=store,
            notifier=notifier,
            portfolio=portfolio,
            risk_config=risk_config,
            buy_threshold=buy_threshold,
            should_exit=should_exit,
            halted=halted,
        )
        second_decision = second["decision"]

        # (Req 2.1/2.2) 第二次返回与首次完全相等的 Decision（含 created_at 等所有字段）。
        assert second_decision == first_decision

        # (Req 2.2) DecisionStore 不发生新增写入：写入次数未增长、id 集合未变。
        assert save_calls["n"] == saves_after_first
        assert store.list_ids() == ids_after_first

        # (Req 2.3) Notifier 不再发送提醒：提醒次数未增长。
        assert len(notifier.sent) == notifications_after_first

        # (Req 2.4) 存储往返一致：读回的 Decision 与首次返回的字段值相等。
        from dataclasses import asdict

        reloaded = store.get(first_decision["signal_id"])
        assert reloaded is not None
        assert asdict(reloaded) == first_decision

        # 第二次触发为幂等命中：未重新走风控明细。
        assert second["idempotent_hit"] is True
        assert second["risk_detail"] == []


# ---------------------------------------------------------------------------
# Property 2: 风控拦截则产出 hold 且 reason 含原因
# ---------------------------------------------------------------------------
#
# 买入候选生成器：保证「空仓 + 信号达标 + 资金足以买入最小手数」，即决策必然
# 走到 RiskManager.check_buy 这一步（不会被「资金不足最小手数」提前拦成 hold）。
# 然后按 block_kind 构造一个**确定会被某前置检查拦截**的场景，并记录该检查
# 应当出现在 reason 中的关键字（按 check_buy 短路顺序，保证目标检查是首个失败项）。
#
# check_buy 短路顺序：kill-switch/熔断 -> 黑名单 -> 停牌 -> 组合市值非正 ->
# 总仓位上限 -> 单票上限。
_BLOCK_KINDS = [
    "kill_switch",
    "circuit",
    "blacklist",
    "halted",
]


@st.composite
def blocked_buy_scenarios(draw):
    """生成必被风控拦截的买入候选场景。

    返回 dict，含信号/价格/阈值/组合/风控配置/停牌标志、kill-switch/熔断运行时
    状态，以及该拦截原因应包含的关键字 `expected`。
    """
    kind = draw(st.sampled_from(_BLOCK_KINDS))

    # 信号达标：阈值 <= 0.8 且信号 >= 0.85，保证 signal > buy_threshold。
    buy_threshold = draw(st.floats(min_value=0.0, max_value=0.8, **_finite))
    signal = draw(st.floats(min_value=0.85, max_value=1.0, **_finite))

    # 价格较小、组合市值很大 -> 估算手数远超最小手数，且 intended_value ≈ 0.95×组合市值。
    price = draw(st.floats(min_value=1.0, max_value=100.0, **_finite))
    portfolio_value = draw(st.floats(min_value=1e6, max_value=1e8, **_finite))

    # 默认：所有前置检查放行。
    blacklist: set[str] = set()
    max_total = 0.95
    max_single = 0.30
    allow_when_halted = False
    halted = False
    kill = False
    circuit = False
    current_total = 0.0
    current_symbol = 0.0

    if kind == "kill_switch":
        kill = True
        expected = "kill-switch"
    elif kind == "circuit":
        circuit = True
        expected = "熔断"
    elif kind == "blacklist":
        blacklist = {VT_SYMBOL}
        expected = "黑名单"
    elif kind == "halted":
        halted = True
        allow_when_halted = False
        expected = "停牌"
    else:
        raise AssertionError(f"未知拦截类型: {kind}")

    config = RiskConfig(
        blacklist=blacklist,
        max_total_position_ratio=max_total,
        max_single_position_ratio=max_single,
        allow_when_halted=allow_when_halted,
    )
    portfolio = PortfolioSnapshot(
        portfolio_value=portfolio_value,
        total_position_value=current_total,
        current_position=0,  # 空仓 -> 走入场分支
        current_symbol_value=current_symbol,
    )
    return {
        "kind": kind,
        "signal": signal,
        "price": price,
        "buy_threshold": buy_threshold,
        "portfolio": portfolio,
        "risk_config": config,
        "halted": halted,
        "kill": kill,
        "circuit": circuit,
        "expected": expected,
        "should_exit": draw(st.booleans()),
    }


def _patch_risk_manager(monkeypatch, *, kill: bool, circuit: bool) -> None:
    """让编排器构造的 RiskManager 携带指定的运行时 kill-switch / 熔断状态。

    `kill_switch` / `circuit_broken` 是 RiskManager 的运行时状态（非 RiskConfig
    字段），无法经 run_live_decision 的入参注入，故在此通过工厂在构造后置位。
    """
    real_cls = RiskManager

    def factory(config):
        rm = real_cls(config)
        if kill:
            rm.kill_switch = True
        if circuit:
            rm.circuit_broken = True
        return rm

    monkeypatch.setattr(orchestrator, "RiskManager", factory)


# Feature: trading-console, Property 2: 风控拦截则产出 hold 且 reason 含原因
# 对任意买入候选（空仓且信号达标），当 RiskManager 任一硬前置检查（黑名单/停牌/
# kill-switch/熔断）拦截时，返回的 Decision 的 action 必为 hold，且 reason 包含拦截原因。
# 总仓位/单票仓位上限属于 sizing 上限：SignalService 会先裁剪买入额，再做最终风控校验。
# Validates: Requirements 3.1, 3.2, 3.3, 3.5
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scenario=blocked_buy_scenarios())
def test_property_2_risk_block_yields_hold_with_reason(monkeypatch, scenario):
    _stub_io(monkeypatch, signal=scenario["signal"], price=scenario["price"])
    _patch_risk_manager(monkeypatch, kill=scenario["kill"], circuit=scenario["circuit"])

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        notifier = LogNotifier()

        result = _run(
            store=store,
            notifier=notifier,
            portfolio=scenario["portfolio"],
            risk_config=scenario["risk_config"],
            buy_threshold=scenario["buy_threshold"],
            should_exit=scenario["should_exit"],
            halted=scenario["halted"],
        )
        decision = result["decision"]

        # (Req 3.1/3.2/3.5) 风控拦截买入候选 -> action 必为 hold。
        assert decision["action"] == "hold", (
            f"block_kind={scenario['kind']} 期望 hold，实际 {decision['action']}: {decision}"
        )

        # (Req 3.2/3.3/3.5) reason 含「风控拦截」前缀与具体拦截原因关键字。
        assert "风控拦截" in decision["reason"], decision["reason"]
        assert scenario["expected"] in decision["reason"], (
            f"block_kind={scenario['kind']} 期望 reason 含 '{scenario['expected']}'，"
            f"实际: {decision['reason']}"
        )

        # 首次触发：风控明细已记录（非幂等命中），其中目标检查项标记为未通过。
        assert result["idempotent_hit"] is False
        assert result["risk_detail"]


# ---------------------------------------------------------------------------
# Property 3: 风控明细完整覆盖 5 项且结论与 RiskManager 一致
# ---------------------------------------------------------------------------
#
# 买入候选生成器：保证「空仓 + 信号达标 + 资金足以买入最小手数」，使决策必然
# 走到 RiskInspector.check_buy 这一步（不会被「资金不足最小手数」提前拦成 hold），
# 从而 risk_detail 被填充。风控配置/停牌/kill-switch/熔断随机（放行与拦截皆可），
# 用于验证：风控明细完整覆盖 5 项，且其综合放行结论与独立的 RiskManager.check_buy
# 权威判定一致。
#
# check_buy 入参取价路径与 SignalService 一致：
#   目标买入额 = min(position_ratio 目标仓位, 风控剩余额度)，再按 min_volume 向下取整。

# 风控明细应当出现的 5 个检查项（按 RiskInspector 记录顺序）。
_EXPECTED_RISK_CHECKS = [
    "kill_switch_or_circuit",
    "blacklist",
    "halted",
    "max_total_position",
    "max_single_position",
]


def _intended_value_after_capacity(
    portfolio: PortfolioSnapshot,
    risk_config: RiskConfig,
    price: float,
    halted: bool,
    kill: bool,
    circuit: bool,
) -> float:
    """复刻 SignalService 的目标仓位 + 风控额度裁剪 + 最小手数取整。"""
    import math

    position_ratio = 0.95
    min_volume = 100
    if price <= 0:
        return 0.0
    target_value = portfolio.portfolio_value * position_ratio
    manager = RiskManager(risk_config)
    manager.kill_switch = kill
    manager.circuit_broken = circuit
    capacity, _ = manager.buy_capacity(
        vt_symbol=VT_SYMBOL,
        portfolio_value=portfolio.portfolio_value,
        current_total_position_value=portfolio.total_position_value,
        current_symbol_value=portfolio.current_symbol_value,
        halted=halted,
    )
    clipped_value = min(target_value, max(0.0, capacity))
    volume = int(math.floor(clipped_value / price / min_volume)) * min_volume
    return volume * price


@st.composite
def buy_candidates(draw):
    """生成必然走到 check_buy 的买入候选（空仓 + 信号达标 + 资金充足）。

    风控配置 / 停牌 / kill-switch / 熔断随机，放行与拦截皆可能出现。
    """
    # 信号达标：阈值 <= 0.8 且信号 >= 0.85，保证 signal > buy_threshold。
    buy_threshold = draw(st.floats(min_value=0.0, max_value=0.8, **_finite))
    signal = draw(st.floats(min_value=0.85, max_value=1.0, **_finite))

    # 价格较小、组合市值很大 -> sized_volume 必然 >= 最小手数（100）。
    price = draw(st.floats(min_value=1.0, max_value=100.0, **_finite))
    portfolio_value = draw(st.floats(min_value=1e6, max_value=1e8, **_finite))
    total_position_value = draw(st.floats(min_value=0.0, max_value=1e8, **_finite))
    current_symbol_value = draw(st.floats(min_value=0.0, max_value=1e8, **_finite))

    portfolio = PortfolioSnapshot(
        portfolio_value=portfolio_value,
        total_position_value=total_position_value,
        current_position=0,  # 空仓 -> 走入场分支
        current_symbol_value=current_symbol_value,
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
    }


# Feature: trading-console, Property 3: 风控明细完整覆盖 5 项且结论与 RiskManager 一致
# 对任意首次触发的买入候选，响应中的 risk_detail 必包含全部前置检查项（kill-switch/
# 熔断、黑名单、停牌、总仓上限、单票上限），且每项含 passed 布尔与 detail 文本；其综合
# 放行结论与既有 RiskManager.check_buy 的权威判定一致。
# Validates: Requirements 3.4
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(candidate=buy_candidates())
def test_property_3_risk_detail_complete_and_consistent(monkeypatch, candidate):
    _stub_io(monkeypatch, signal=candidate["signal"], price=candidate["price"])
    _patch_risk_manager(monkeypatch, kill=candidate["kill"], circuit=candidate["circuit"])

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        notifier = LogNotifier()

        result = _run(
            store=store,
            notifier=notifier,
            portfolio=candidate["portfolio"],
            risk_config=candidate["risk_config"],
            buy_threshold=candidate["buy_threshold"],
            should_exit=candidate["should_exit"],
            halted=candidate["halted"],
        )
        risk_detail = result["risk_detail"]

        # 首次触发：必然走过风控（非幂等命中），明细被填充。
        assert result["idempotent_hit"] is False
        assert risk_detail, "首次触发的买入候选应产出非空 risk_detail"

        # (Req 3.4) 完整覆盖全部 5 个前置检查项，且顺序一致。
        assert [item["check"] for item in risk_detail] == _EXPECTED_RISK_CHECKS

        # (Req 3.4) 每项含 passed 布尔与 detail 文本。
        for item in risk_detail:
            assert isinstance(item["passed"], bool), item
            assert isinstance(item["detail"], str) and item["detail"], item

        # 综合放行结论：风控明细全部通过 <=> 放行。
        detail_conclusion = all(item["passed"] for item in risk_detail)

        # 独立调用权威 RiskManager.check_buy（同配置 + 同运行时状态 + 同入参）。
        authoritative = RiskManager(candidate["risk_config"])
        authoritative.kill_switch = candidate["kill"]
        authoritative.circuit_broken = candidate["circuit"]

        portfolio = candidate["portfolio"]
        intended_value = _intended_value_after_capacity(
            portfolio,
            candidate["risk_config"],
            candidate["price"],
            candidate["halted"],
            candidate["kill"],
            candidate["circuit"],
        )
        ok, _reason = authoritative.check_buy(
            vt_symbol=VT_SYMBOL,
            intended_value=intended_value,
            portfolio_value=portfolio.portfolio_value,
            current_total_position_value=portfolio.total_position_value,
            current_symbol_value=portfolio.current_symbol_value,
            halted=candidate["halted"],
        )

        # (Req 3.4) 明细综合结论与 RiskManager 权威判定一致。
        assert detail_conclusion == ok, (
            f"明细综合结论={detail_conclusion} 与 RiskManager.check_buy={ok} 不一致；"
            f"risk_detail={risk_detail}"
        )


# ---------------------------------------------------------------------------
# Property 4: kill-switch 已确定 buy 不被回溯篡改
# ---------------------------------------------------------------------------
#
# 生成器 `confirmed_buy_scenarios` 保证首次触发必然产出 buy：
#   - 空仓（current_position == 0）-> 走入场分支；
#   - 信号达标（signal >= 0.85 > buy_threshold <= 0.8）；
#   - 价格小、组合市值大 -> sized_volume >= 最小手数（100）；
#   - 宽松风控：黑名单不含目标标的、总仓/单票上限均放到 1.0、不停牌、初始无
#     kill-switch/熔断 -> check_buy 必放行 -> action == "buy"。
# 首次触发后将 kill-switch 置位（通过 _patch_risk_manager 重打桩，使编排器后续
# 构造的 RiskManager 携带 kill_switch=True），再对同一 signal_id 二次触发，并经
# DecisionStore.get 重新读回，断言返回的仍是原 buy Decision 且字段完全不变。


@st.composite
def confirmed_buy_scenarios(draw):
    """生成首次触发必然产出 buy 的场景（空仓 + 信号达标 + 资金充足 + 宽松风控）。"""
    buy_threshold = draw(st.floats(min_value=0.0, max_value=0.8, **_finite))
    signal = draw(st.floats(min_value=0.85, max_value=1.0, **_finite))

    # 价格较小、组合市值很大 -> sized_volume 必然 >= 最小手数（100）。
    price = draw(st.floats(min_value=1.0, max_value=100.0, **_finite))
    portfolio_value = draw(st.floats(min_value=1e6, max_value=1e8, **_finite))

    # 宽松风控：上限均放到 1.0，初始持仓为 0 -> check_buy 必放行。
    config = RiskConfig(
        blacklist=set(),
        max_total_position_ratio=1.0,
        max_single_position_ratio=1.0,
        allow_when_halted=False,
    )
    portfolio = PortfolioSnapshot(
        portfolio_value=portfolio_value,
        total_position_value=0.0,
        current_position=0,  # 空仓 -> 走入场分支
        current_symbol_value=0.0,
    )
    return {
        "signal": signal,
        "price": price,
        "buy_threshold": buy_threshold,
        "portfolio": portfolio,
        "risk_config": config,
    }


# Feature: trading-console, Property 4: kill-switch 已确定 buy 不被回溯篡改
# 对任意已被 DecisionStore 持久化为 buy 的 Decision，之后即使 kill-switch 触发，
# 对同一 Signal_Id 再次查询/触发仍返回原 buy Decision 不变。
# Validates: Requirements 3.6, 2.2
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scenario=confirmed_buy_scenarios())
def test_property_4_killswitch_does_not_retroactively_alter_buy(monkeypatch, scenario):
    _stub_io(monkeypatch, signal=scenario["signal"], price=scenario["price"])

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        notifier = LogNotifier()

        # 首次触发：无 kill-switch / 熔断 -> 应产出并持久化 buy。
        _patch_risk_manager(monkeypatch, kill=False, circuit=False)
        first = _run(
            store=store,
            notifier=notifier,
            portfolio=scenario["portfolio"],
            risk_config=scenario["risk_config"],
            buy_threshold=scenario["buy_threshold"],
            should_exit=False,
            halted=False,
        )
        first_decision = first["decision"]

        # 前置条件：首次触发确实产出了已持久化的 buy。
        assert first_decision["action"] == "buy", first_decision
        signal_id = first_decision["signal_id"]
        assert store.get(signal_id) is not None

        # (Req 3.6) kill-switch 在 buy 已确定之后才触发：让后续编排器构造的
        # RiskManager 携带 kill_switch=True，模拟 kill-switch 已触发。
        _patch_risk_manager(monkeypatch, kill=True, circuit=False)

        # 对同一 signal_id 再次触发：幂等短路在 check_buy 之前直接返回既有 Decision。
        second = _run(
            store=store,
            notifier=notifier,
            portfolio=scenario["portfolio"],
            risk_config=scenario["risk_config"],
            buy_threshold=scenario["buy_threshold"],
            should_exit=False,
            halted=False,
        )
        second_decision = second["decision"]

        # (Req 3.6/2.2) 即使 kill-switch 已触发，再次触发仍返回原 buy Decision 不变。
        assert second_decision == first_decision
        assert second_decision["action"] == "buy"
        assert second["idempotent_hit"] is True

        # (Req 2.2/3.6) 经 DecisionStore.get 重新读回，仍是原 buy，字段完全一致。
        from dataclasses import asdict

        reloaded = store.get(signal_id)
        assert reloaded is not None
        assert reloaded.action == "buy"
        assert asdict(reloaded) == first_decision


# ---------------------------------------------------------------------------
# Property 5: 默认决策日为当天
# ---------------------------------------------------------------------------
#
# 默认决策日逻辑（`req.trade_date or date.today()`）位于 Decision_API 路由层，
# 故本属性在 API 层（FastAPI TestClient）断言：当请求**未指定 trade_date** 时，
# 系统以当天作为 Decision_Date，并让其同时参与「推理区间」与「Signal_Id 生成」。
#
# 外部 I/O 桩化：predict_cnn_signals 被桩化为
#   (a) 捕获其接收到的推理区间 start/end（证明推理区间 = 当天）；
#   (b) 返回一根落在「当天」的 bar，使 _select_decision_bar 能取到 signal。
# _load_close_price 桩化为确定价位。DecisionStore 与模型库均用 tmp_path 隔离。
#
# 生成器在「不提供 trade_date」的前提下随机化其余请求字段（方案 / 模型版本 /
# 买入阈值 / 数据源 / 停牌 / 出场 / 组合快照），以覆盖「任意未指定 trade_date 的
# 决策请求」。model 与 vt_symbol 保持常量（桩输出按该 vt_symbol 构造）。


@st.composite
def portfolio_bodies(draw) -> dict:
    """生成 JSON 可序列化的组合快照请求体：portfolio_value > 0，各市值非负。"""
    return {
        "portfolio_value": draw(st.floats(min_value=1.0, max_value=1e9, **_finite)),
        "total_position_value": draw(st.floats(min_value=0.0, max_value=1e9, **_finite)),
        "current_position": draw(st.integers(min_value=0, max_value=100000)),
        "current_symbol_value": draw(st.floats(min_value=0.0, max_value=1e9, **_finite)),
    }


@st.composite
def no_date_requests(draw) -> dict:
    """生成「未指定 trade_date」的决策请求体，随机化其余字段。"""
    suffix = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8))
    body = {
        "model": MODEL,
        "vt_symbol": VT_SYMBOL,
        "scheme": f"scheme_{suffix}",
        # 关键：故意不放 trade_date（缺省 -> 路由默认当天）。
        "data_source": draw(st.sampled_from(["upload", "pull"])),
        "portfolio": draw(portfolio_bodies()),
        "buy_threshold": draw(thresholds),
        "model_version": draw(st.sampled_from(["", "v1", "v2", f"mv_{suffix}"])),
        "halted": draw(st.booleans()),
        "should_exit": draw(st.booleans()),
    }
    return body


@pytest.fixture
def prop5_client(tmp_path, monkeypatch):
    """隔离的 TestClient：tmp_path 决策存储 + 模型库；桩化外部 I/O 并捕获推理区间。

    返回 (test_client, store, captured)；captured 记录最近一次 predict_cnn_signals
    收到的推理区间 start/end，用于断言「推理区间 = 当天」。
    """
    store = DecisionStore(tmp_path / "decisions")
    monkeypatch.setattr(live_api, "_store", store)

    model_dir = tmp_path / "cnn_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"{MODEL}.pt").write_bytes(b"stub-model")
    monkeypatch.setattr(live_api, "CNN_MODEL_PATH", model_dir)

    captured: dict = {}

    def _stub_predict(**kwargs):
        # 捕获推理区间（证明默认 as_of 当天流入推理区间 end）。
        captured["start"] = kwargs.get("start")
        captured["end"] = kwargs.get("end")
        # 返回一根「确定已收盘」的 bar（end 的前一日，无论当前时刻都已收盘），
        # 使 select_decision_bar 必能取到 signal（避免依赖墙钟是否过收盘）。
        return _signal_frame_on(kwargs.get("end") - timedelta(days=1), signal=0.72)

    monkeypatch.setattr(orchestrator, "predict_cnn_signals", _stub_predict)
    monkeypatch.setattr(
        orchestrator,
        "_load_close_price",
        lambda vt_symbol, instant: (10.0, "d"),
    )

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client, store, captured


def _signal_frame_on(trade_date: date, *, signal: float) -> pl.DataFrame:
    """构造落在指定日期的桩信号帧（schema 同 predict_cnn_signals）。"""
    return pl.DataFrame(
        {
            "datetime": [datetime.combine(trade_date, datetime.min.time())],
            "vt_symbol": [VT_SYMBOL],
            "signal": [float(signal)],
        }
    )


def _poll_task(test_client: TestClient, task_id: str, timeout: float = 10.0) -> dict:
    """轮询任务直至 completed/failed，返回任务 dict。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = test_client.get(f"/api/alpha/tasks/{task_id}")
        assert resp.status_code == 200
        task = resp.json()
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.01)
    raise AssertionError(f"任务 {task_id} 未在 {timeout}s 内完成")


# Feature: decision-instant-unification, Property 5: 默认 as_of 为当前（当天）
# 对任意未指定 as_of 的决策请求，系统以当前时刻为 Decision_Instant.as_of（当天）参与
# 推理区间 end 与决策；bar_freq 默认 1d。
# Validates: Requirements 1.2
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(body=no_date_requests())
def test_property_5_default_as_of_is_today(prop5_client, body):
    test_client, _store, captured = prop5_client

    # 以「未指定 as_of」的请求触发决策；today 在断言时刻取，避免跨午夜竞态。
    resp = test_client.post("/api/live/decision", json=body)
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]

    task = _poll_task(test_client, task_id)
    today = date.today()

    assert task["status"] == "completed", f"任务消息: {task.get('message')}"

    # (Req 1.2) 默认 as_of=当前 → 推理区间 end == 当天。
    assert captured["end"] == today

    # (Req 1.2) 决策 as_of 落在当天；bar_freq 默认 1d。
    decision = task["result"]["decision"]
    assert decision["as_of"][:10] == today.isoformat()
    assert decision["bar_freq"] == "1d"


# ---------------------------------------------------------------------------
# Property 6: 决策动作合法（buy/sell/hold）且字段完整
# ---------------------------------------------------------------------------
#
# 生成 WIDE 范围的输入以同时覆盖三条 action 路径：
#   - sell：should_exit=True 且 current_position > 0（出场优先）；
#   - buy ：空仓 + 信号达标 + 资金充足 + 通过风控；
#   - hold：其余（信号未达标 / 资金不足最小手数 / 风控拦截 / 持有中未到出场条件）。
# 通过让 signal/buy_threshold ∈ [0,1]、price > 0、组合（空仓与持仓皆有）、随机风控
# 配置、随机 halted / should_exit、随机 kill-switch / 熔断运行时状态，使三条路径
# 在 100 次迭代中都被充分触发。无论落到哪条路径，断言 action 合法且字段完整。


@st.composite
def any_decision_scenarios(draw):
    """生成覆盖 buy/sell/hold 三路径的宽范围决策场景。"""
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
    }


# Feature: trading-console, Property 6: 决策动作合法且字段完整
# 对任意成功完成的决策任务，返回的 Decision 的 action 必为 buy/sell/hold 之一，
# 且包含 volume、price、signal、reason 字段。
# Validates: Requirements 1.4
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scenario=any_decision_scenarios())
def test_property_6_decision_action_valid_and_fields_complete(monkeypatch, scenario):
    _stub_io(monkeypatch, signal=scenario["signal"], price=scenario["price"])
    _patch_risk_manager(monkeypatch, kill=scenario["kill"], circuit=scenario["circuit"])

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        notifier = LogNotifier()

        result = _run(
            store=store,
            notifier=notifier,
            portfolio=scenario["portfolio"],
            risk_config=scenario["risk_config"],
            buy_threshold=scenario["buy_threshold"],
            should_exit=scenario["should_exit"],
            halted=scenario["halted"],
        )
        decision = result["decision"]

        # (Req 1.4) action 合法：必为 buy/sell/hold 之一。
        assert decision["action"] in ("buy", "sell", "hold"), decision

        # (Req 1.4) 字段完整：volume/price/signal/reason 均存在。
        for field_name in ("volume", "price", "signal", "reason"):
            assert field_name in decision, f"决策缺少字段 {field_name}: {decision}"

        # 字段类型/语义最小校验：volume 为非负整数，reason 为非空文本。
        assert isinstance(decision["volume"], int) and decision["volume"] >= 0, decision
        assert isinstance(decision["reason"], str) and decision["reason"], decision


# ---------------------------------------------------------------------------
# Property 7: 无券商下单路径（任意结果均不调用下单接口）
# ---------------------------------------------------------------------------
#
# 双重守护：
#   (a) 静态 AST 扫描：对编排器（orchestrator）、API 路由（api/live）、信号服务
#       （signal_service）三个模块的 import 语句（而非全文，避免误伤安全声明文档串
#       中的「不下单」negation）扫描禁用 token（broker/submit_order/place_order/
#       send_order/gateway/order）。任一模块 import 命中即视为存在下单调用路径。
#   (b) 运行时副作用检查：对任意决策执行路径（buy/sell/hold 任意结果），编排返回
#       结构仅含 {decision, risk_detail, idempotent_hit}，且唯一可观察的外部副作用
#       为 DecisionStore 写入与 Notifier 提醒——首次触发恰好落盘 1 条；提醒次数
#       严格等于 (action ∈ {buy, sell}) 时 1 次、否则 0 次。不存在任何其它副作用通道。
#
# 生成器复用 any_decision_scenarios（覆盖 buy/sell/hold 三路径的宽范围输入）。

import ast as _ast
import inspect as _inspect

from aitrade.api import live as _live_api
from aitrade.live import signal_service as _signal_service

# 单一属性 = 单一测试：AST 扫描的禁用 token 与待扫描模块。
_FORBIDDEN_ORDER_TOKENS = [
    "broker",
    "submit_order",
    "place_order",
    "send_order",
    "gateway",
    "order",
]
_NO_BROKER_MODULES = [orchestrator, _live_api, _signal_service]


def _imported_module_names(module) -> list[str]:
    """AST 解析模块源码，仅收集 import / from-import 的模块与名称（不扫全文）。"""
    tree = _ast.parse(_inspect.getsource(module))
    names: list[str] = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, _ast.ImportFrom):
            base = node.module or ""
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names)
    return names


# Feature: trading-console, Property 7: 无券商下单路径
# 对任意决策执行路径（buy/sell/hold 任意结果），系统不调用任何券商网关/下单接口；
# 产出仅限 Decision 持久化与 Notifier 提醒。
# Validates: Requirements 7.1, 7.2
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scenario=any_decision_scenarios())
def test_property_7_no_broker_order_path(monkeypatch, scenario):
    # (a) 静态 AST 守护：编排器 / API / 信号服务三模块的 import 均不含下单 token。
    for module in _NO_BROKER_MODULES:
        for mod_name in _imported_module_names(module):
            lowered = mod_name.lower()
            for token in _FORBIDDEN_ORDER_TOKENS:
                assert token not in lowered, (
                    f"{module.__name__} 不应 import 下单相关模块: {mod_name}"
                )

    _stub_io(monkeypatch, signal=scenario["signal"], price=scenario["price"])
    _patch_risk_manager(monkeypatch, kill=scenario["kill"], circuit=scenario["circuit"])

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)
        notifier = LogNotifier()

        # 仅统计「DecisionStore 写入」这一允许的副作用通道（验证不存在其它通道）。
        save_calls = {"n": 0}
        original_save = store.save

        def _counting_save(decision):
            save_calls["n"] += 1
            return original_save(decision)

        monkeypatch.setattr(store, "save", _counting_save)

        result = _run(
            store=store,
            notifier=notifier,
            portfolio=scenario["portfolio"],
            risk_config=scenario["risk_config"],
            buy_threshold=scenario["buy_threshold"],
            should_exit=scenario["should_exit"],
            halted=scenario["halted"],
        )

        # (b) 运行时副作用守护：返回结构仅含决策三元组，无任何下单/网关字段。
        assert set(result.keys()) == {"decision", "risk_detail", "idempotent_hit"}

        action = result["decision"]["action"]
        assert action in ("buy", "sell", "hold")

        # 首次触发：唯一落盘副作用为「DecisionStore 写入恰好 1 条」（无其它写入通道）。
        assert save_calls["n"] == 1
        assert len(store.list_ids()) == 1

        # 唯一外部提醒副作用为 Notifier：buy/sell 提醒 1 次、hold 不提醒。
        expected_sends = 1 if action in ("buy", "sell") else 0
        assert len(notifier.sent) == expected_sends, (
            f"action={action} 期望提醒 {expected_sends} 次，实际 {len(notifier.sent)} 次"
        )


# ---------------------------------------------------------------------------
# Property 8: 决策存储往返一致
# ---------------------------------------------------------------------------
#
# 直接针对 DecisionStore.save/get 验证往返一致：生成任意合法的 Decision（随机化
# 全部字段 signal_id / trade_date / scheme / action / vt_symbol / volume / price /
# signal / reason / created_at），保存后再读回，断言读回的 Decision 与保存时所有
# 字段值相等（asdict 相等）。DecisionStore 用临时目录隔离。
#
# 字段生成约束：
#   - signal_id：非空文本（含 ':' / '/' 以覆盖 _path 的文件名净化），作为幂等键与文件名；
#   - 浮点字段（price/signal）：有限值（排除 NaN/Inf，避免 JSON 往返后 NaN != NaN）；
#   - action ∈ {buy, sell, hold}；vt_symbol / price / signal 可为 None（Optional）。

from dataclasses import asdict as _asdict

from aitrade.live.decision import Decision

_safe_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.",
    min_size=1,
    max_size=24,
)
_reason_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz 中文测试0123456789_-.,",
    min_size=0,
    max_size=64,
)


@st.composite
def decisions(draw) -> Decision:
    """生成任意合法的 Decision：随机化全部字段。"""
    # signal_id 含 ':' / '/' 以覆盖文件名净化逻辑；保证非空。
    sid_parts = [draw(_safe_text), draw(_safe_text)]
    sep = draw(st.sampled_from([":", "/", ":", "_"]))
    signal_id = sep.join(sid_parts)

    bar_dt = draw(st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2100, 12, 31)))
    return Decision(
        signal_id=signal_id,
        decision_bar_dt=bar_dt.isoformat(),
        as_of=bar_dt.isoformat(),
        bar_freq="1d",
        scheme=draw(_safe_text),
        action=draw(st.sampled_from(["buy", "sell", "hold"])),
        vt_symbol=draw(st.one_of(st.none(), _safe_text)),
        volume=draw(st.integers(min_value=0, max_value=1_000_000)),
        price=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1e9, **_finite))),
        signal=draw(st.one_of(st.none(), st.floats(min_value=-1e6, max_value=1e6, **_finite))),
        reason=draw(_reason_text),
        created_at=draw(_safe_text),
    )


# Feature: trading-console, Property 8: 决策存储往返一致
# 对任意被保存的 Decision，经 DecisionStore 读取后所有字段值与保存时相等。
# Validates: Requirements 2.4
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(decision=decisions())
def test_property_8_decision_store_roundtrip(decision):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DecisionStore(tmpdir)

        store.save(decision)
        reloaded = store.get(decision.signal_id)

        # (Req 2.4) 读回的 Decision 非空，且全部字段值与保存时相等。
        assert reloaded is not None, f"保存后读回为 None: signal_id={decision.signal_id!r}"
        assert _asdict(reloaded) == _asdict(decision), (
            f"往返不一致:\n  saved={_asdict(decision)}\n  loaded={_asdict(reloaded)}"
        )
