"""
RiskInspector 单元测试（交易操作台特性，需求 3.1 / 3.4 / 3.5）。

验证两件事：
1. `records` 始终覆盖全部 5 项检查，且每项含 `check`/`passed`/`detail` 字段（需求 3.4）。
2. RiskInspector 的综合放行结论与权威 `RiskManager.check_buy` 完全一致，
   覆盖黑名单 / 停牌 / 超单票 / 超总仓 / kill-switch 各拦截分支（需求 3.1 / 3.5）。

全部不下单、不依赖外部网络。
"""

from __future__ import annotations

from aitrade.live.risk import RiskConfig, RiskManager
from aitrade.live.risk_inspector import RiskInspector


# RiskManager 同序的 5 项检查标识
EXPECTED_CHECKS = [
    "kill_switch_or_circuit",
    "blacklist",
    "halted",
    "max_total_position",
    "max_single_position",
]


def _inspector(config: RiskConfig | None = None) -> tuple[RiskInspector, RiskManager]:
    rm = RiskManager(config)
    return RiskInspector(rm), rm


def _buy_kwargs(**overrides) -> dict:
    base = dict(
        vt_symbol="X.SZSE",
        intended_value=20000.0,
        portfolio_value=100000.0,
        current_total_position_value=0.0,
        current_symbol_value=0.0,
        halted=False,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 需求 3.4：风控明细覆盖全部 5 项，每项含 passed/detail
# ---------------------------------------------------------------------------
def test_records_cover_all_five_checks_with_passed_and_detail() -> None:
    inspector, _ = _inspector(
        RiskConfig(max_total_position_ratio=0.95, max_single_position_ratio=0.30)
    )

    inspector.check_buy(**_buy_kwargs())

    # 恰好 5 项，且与 RiskManager 同序
    assert [r["check"] for r in inspector.records] == EXPECTED_CHECKS
    # 每项均含必需字段且类型正确
    for rec in inspector.records:
        assert set(rec.keys()) == {"check", "passed", "detail"}
        assert isinstance(rec["passed"], bool)
        assert isinstance(rec["detail"], str)
        assert rec["detail"] != ""


def test_records_present_even_when_all_checks_pass() -> None:
    inspector, _ = _inspector(
        RiskConfig(max_total_position_ratio=0.95, max_single_position_ratio=0.30)
    )

    ok, _ = inspector.check_buy(**_buy_kwargs(intended_value=10000.0))

    assert ok is True
    assert len(inspector.records) == 5
    assert all(r["passed"] for r in inspector.records)


# ---------------------------------------------------------------------------
# 需求 3.1 / 3.5：综合结论与 RiskManager.check_buy 权威结果一致（各拦截分支）
# ---------------------------------------------------------------------------
def test_conclusion_matches_risk_manager_blacklist() -> None:
    cfg = RiskConfig(blacklist={"BAD.SZSE"})
    inspector, rm = _inspector(cfg)
    kwargs = _buy_kwargs(vt_symbol="BAD.SZSE", intended_value=1000.0)

    insp_ok, insp_reason = inspector.check_buy(**kwargs)
    rm_ok, rm_reason = rm.check_buy(**kwargs)

    assert (insp_ok, insp_reason) == (rm_ok, rm_reason)
    assert insp_ok is False
    # 明细中黑名单项被标记为未通过
    blacklist_rec = next(r for r in inspector.records if r["check"] == "blacklist")
    assert blacklist_rec["passed"] is False


def test_conclusion_matches_risk_manager_halted() -> None:
    inspector, rm = _inspector(RiskConfig(allow_when_halted=False))
    kwargs = _buy_kwargs(intended_value=1000.0, halted=True)

    insp_ok, insp_reason = inspector.check_buy(**kwargs)
    rm_ok, rm_reason = rm.check_buy(**kwargs)

    assert (insp_ok, insp_reason) == (rm_ok, rm_reason)
    assert insp_ok is False and "停牌" in insp_reason
    halted_rec = next(r for r in inspector.records if r["check"] == "halted")
    assert halted_rec["passed"] is False


def test_conclusion_matches_risk_manager_over_single_position() -> None:
    cfg = RiskConfig(max_total_position_ratio=0.95, max_single_position_ratio=0.30)
    inspector, rm = _inspector(cfg)
    # 单票上限 30% → 拟买 40000 超限（但未超总仓 95%）
    kwargs = _buy_kwargs(intended_value=40000.0)

    insp_ok, insp_reason = inspector.check_buy(**kwargs)
    rm_ok, rm_reason = rm.check_buy(**kwargs)

    assert (insp_ok, insp_reason) == (rm_ok, rm_reason)
    assert insp_ok is False
    single_rec = next(r for r in inspector.records if r["check"] == "max_single_position")
    assert single_rec["passed"] is False


def test_conclusion_matches_risk_manager_over_total_position() -> None:
    cfg = RiskConfig(max_total_position_ratio=0.95, max_single_position_ratio=0.99)
    inspector, rm = _inspector(cfg)
    # 总仓上限 95% → 已有 90000，再拟买 20000 → 110000 超总仓（单票上限放宽到 99% 不触发）
    kwargs = _buy_kwargs(
        intended_value=20000.0,
        current_total_position_value=90000.0,
        current_symbol_value=0.0,
    )

    insp_ok, insp_reason = inspector.check_buy(**kwargs)
    rm_ok, rm_reason = rm.check_buy(**kwargs)

    assert (insp_ok, insp_reason) == (rm_ok, rm_reason)
    assert insp_ok is False
    total_rec = next(r for r in inspector.records if r["check"] == "max_total_position")
    assert total_rec["passed"] is False


def test_conclusion_matches_risk_manager_kill_switch() -> None:
    inspector, rm = _inspector(RiskConfig())
    rm.trip_kill_switch()
    kwargs = _buy_kwargs(intended_value=1000.0)

    insp_ok, insp_reason = inspector.check_buy(**kwargs)
    rm_ok, rm_reason = rm.check_buy(**kwargs)

    assert (insp_ok, insp_reason) == (rm_ok, rm_reason)
    assert insp_ok is False and "kill-switch" in insp_reason
    gate_rec = next(r for r in inspector.records if r["check"] == "kill_switch_or_circuit")
    assert gate_rec["passed"] is False


def test_conclusion_matches_risk_manager_circuit_breaker() -> None:
    inspector, rm = _inspector(RiskConfig(daily_loss_limit=0.05))
    assert rm.update_daily_pnl(-6000, 100000) is True  # 单日亏损达 5% → 熔断
    kwargs = _buy_kwargs(intended_value=1000.0)

    insp_ok, insp_reason = inspector.check_buy(**kwargs)
    rm_ok, rm_reason = rm.check_buy(**kwargs)

    assert (insp_ok, insp_reason) == (rm_ok, rm_reason)
    assert insp_ok is False and "熔断" in insp_reason
    gate_rec = next(r for r in inspector.records if r["check"] == "kill_switch_or_circuit")
    assert gate_rec["passed"] is False


def test_conclusion_matches_risk_manager_allowed_buy() -> None:
    cfg = RiskConfig(max_total_position_ratio=0.95, max_single_position_ratio=0.30)
    inspector, rm = _inspector(cfg)
    kwargs = _buy_kwargs(intended_value=20000.0)

    insp_ok, insp_reason = inspector.check_buy(**kwargs)
    rm_ok, rm_reason = rm.check_buy(**kwargs)

    assert (insp_ok, insp_reason) == (rm_ok, rm_reason)
    assert insp_ok is True


def test_can_trade_passthrough() -> None:
    inspector, rm = _inspector(RiskConfig())
    assert inspector.can_trade() == rm.can_trade()
    rm.trip_kill_switch()
    assert inspector.can_trade() == rm.can_trade()
    assert inspector.can_trade()[0] is False
