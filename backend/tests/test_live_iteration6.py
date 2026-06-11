"""
迭代 6 验收测试：半自动信号服务（交易日历/调度/风控/提醒/信号服务/决策持久化）。
全部不下单、不依赖外部网络。
"""

from __future__ import annotations

from datetime import date, datetime, time

import pytest

from aitrade.live.calendar import TradingCalendar
from aitrade.live.decision import Decision, DecisionStore
from aitrade.live.decision_instant import DecisionInstant, make_signal_id
from aitrade.live.notifier import LogNotifier, MultiNotifier, RetryNotifier, Notifier
from aitrade.live.risk import RiskConfig, RiskManager
from aitrade.live.signal_service import PortfolioSnapshot, SignalService


def _run(svc: SignalService, d: date, **kw) -> Decision:
    """以 1d 收盘后 as_of 触发某日决策（Decision_Bar = 当日）。"""
    return svc.run_for_instant(
        DecisionInstant(datetime.combine(d, time(15, 5)), "1d"),
        decision_bar_dt=datetime.combine(d, time(15, 0)),
        **kw,
    )


# ---------------------------------------------------------------------------
# 交易日历
# ---------------------------------------------------------------------------
def test_calendar_weekday_and_holiday() -> None:
    cal = TradingCalendar(holidays={date(2026, 6, 8)})  # 周一设为节假日
    assert cal.is_trading_day(date(2026, 6, 9)) is True    # 周二
    assert cal.is_trading_day(date(2026, 6, 8)) is False   # 节假日
    assert cal.is_trading_day(date(2026, 6, 7)) is False   # 周日


def test_calendar_explicit_trading_days() -> None:
    cal = TradingCalendar(trading_days={date(2026, 6, 8)})
    assert cal.is_trading_day(date(2026, 6, 8)) is True
    assert cal.is_trading_day(date(2026, 6, 9)) is False


# ---------------------------------------------------------------------------
# 提醒：多通道扇出 + 失败隔离 + 重试
# ---------------------------------------------------------------------------
class _FailNotifier:
    def send(self, title: str, message: str) -> bool:
        raise RuntimeError("通道故障")


def test_multi_notifier_isolates_failure() -> None:
    log = LogNotifier()
    multi = MultiNotifier([_FailNotifier(), log])
    assert multi.send("t", "m") is True   # 一个失败、一个成功 → 整体成功
    assert log.sent == [("t", "m")]


def test_retry_notifier_gives_up() -> None:
    assert RetryNotifier(_FailNotifier(), retries=2).send("t", "m") is False
    assert isinstance(LogNotifier(), Notifier)


# ---------------------------------------------------------------------------
# 风控
# ---------------------------------------------------------------------------
def test_risk_blacklist_and_limits() -> None:
    rm = RiskManager(RiskConfig(blacklist={"BAD.SZSE"}, max_total_position_ratio=0.95,
                                max_single_position_ratio=0.30))
    ok, _ = rm.check_buy("BAD.SZSE", 1000, 100000, 0)
    assert ok is False
    # 单票上限：30% → 拟买 40000 超限
    ok, reason = rm.check_buy("X.SZSE", 40000, 100000, 0, 0)
    assert ok is False and "单票" in reason
    # 合规买入
    ok, _ = rm.check_buy("X.SZSE", 20000, 100000, 0, 0)
    assert ok is True


def test_risk_halted_and_kill_switch_and_circuit() -> None:
    rm = RiskManager(RiskConfig(daily_loss_limit=0.05))
    ok, reason = rm.check_buy("X.SZSE", 1000, 100000, 0, halted=True)
    assert ok is False and "停牌" in reason

    rm.trip_kill_switch()
    ok, reason = rm.check_buy("X.SZSE", 1000, 100000, 0)
    assert ok is False and "kill-switch" in reason

    rm.reset()
    # 单日亏损达 5% → 熔断
    assert rm.update_daily_pnl(-6000, 100000) is True
    ok, reason = rm.check_buy("X.SZSE", 1000, 100000, 0)
    assert ok is False and "熔断" in reason


# ---------------------------------------------------------------------------
# 决策持久化（幂等）
# ---------------------------------------------------------------------------
def test_decision_store_roundtrip(tmp_path) -> None:
    store = DecisionStore(tmp_path)
    bar_dt = datetime(2026, 6, 9, 15, 0)
    sid = make_signal_id(bar_dt, "1d", "eod_buy_v1", "v3")
    assert store.exists(sid) is False
    d = Decision(signal_id=sid, decision_bar_dt=bar_dt.isoformat(), as_of=bar_dt.isoformat(),
                 bar_freq="1d", scheme="eod_buy_v1",
                 action="buy", vt_symbol="000001.SZSE", volume=900, price=12.3, signal=0.71)
    store.save(d)
    assert store.exists(sid) is True
    got = store.get(sid)
    assert got is not None and got.action == "buy" and got.volume == 900


# ---------------------------------------------------------------------------
# 信号服务（决策 + 风控 + 提醒 + 幂等）
# ---------------------------------------------------------------------------
def _service(tmp_path, **kw) -> tuple[SignalService, LogNotifier, DecisionStore]:
    notifier = LogNotifier()
    store = DecisionStore(tmp_path)
    risk = RiskManager(RiskConfig(max_total_position_ratio=0.95, max_single_position_ratio=1.0))
    svc = SignalService("eod_buy_v1", buy_threshold=0.6, risk=risk, store=store,
                        notifier=notifier, model_version="v3", **kw)
    return svc, notifier, store


def test_signal_service_buy_and_idempotent(tmp_path) -> None:
    svc, notifier, store = _service(tmp_path)
    pf = PortfolioSnapshot(portfolio_value=100000, total_position_value=0, current_position=0)
    d1 = _run(svc, date(2026, 6, 9), signal=0.72, price=10.0, portfolio=pf,
              vt_symbol="000001.SZSE")
    assert d1.action == "buy" and d1.volume == 9500   # 95% * 100000 / 10 = 9500
    assert len(notifier.sent) == 1

    # 再次运行同日 → 幂等返回，不重复提醒
    d2 = _run(svc, date(2026, 6, 9), signal=0.72, price=10.0, portfolio=pf,
              vt_symbol="000001.SZSE")
    assert d2.signal_id == d1.signal_id
    assert len(notifier.sent) == 1


def test_signal_service_hold_when_below_threshold(tmp_path) -> None:
    svc, notifier, _ = _service(tmp_path)
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)
    d = _run(svc, date(2026, 6, 9), signal=0.5, price=10.0, portfolio=pf,
             vt_symbol="000001.SZSE")
    assert d.action == "hold"
    assert len(notifier.sent) == 0   # 观望不提醒


def test_signal_service_exit_signal(tmp_path) -> None:
    svc, notifier, _ = _service(tmp_path)
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=900)
    d = _run(svc, date(2026, 6, 10), signal=0.3, price=10.0, portfolio=pf,
             vt_symbol="000001.SZSE", should_exit=True)
    assert d.action == "sell" and d.volume == 900
    assert len(notifier.sent) == 1


def test_signal_service_caps_buy_to_single_position_limit(tmp_path) -> None:
    notifier = LogNotifier()
    store = DecisionStore(tmp_path)
    risk = RiskManager(RiskConfig(max_single_position_ratio=0.10))  # 单票仅 10%
    svc = SignalService("s", 0.6, risk, store, notifier, model_version="v1")
    pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)
    d = _run(svc, date(2026, 6, 9), signal=0.9, price=10.0, portfolio=pf,
             vt_symbol="000001.SZSE")
    assert d.action == "buy"
    assert d.volume == 1000  # 10% * 100000 / 10 = 1000 股
    assert d.reason == "概率达标且通过风控"
