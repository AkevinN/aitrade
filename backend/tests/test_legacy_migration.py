"""
遗留数据迁移属性测试（Decision Instant Unification，Wave 3 迁移片段，纯新增）。

Property DI-6：旧 Decision/Plan JSON 迁移后关键语义一致，且不含 trade_date /
data_basis / decision_time(s) 旧字段；迁移幂等。
"""

from __future__ import annotations

from datetime import date, datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.live.decision_instant import SESSION_CLOSE
from aitrade.live.legacy_migration import migrate_decision, migrate_plan

_dates = st.dates(min_value=date(2024, 1, 1), max_value=date(2027, 12, 31))
_hhmm = st.builds(
    lambda h, m: f"{h:02d}:{m:02d}",
    st.integers(min_value=0, max_value=23),
    st.integers(min_value=0, max_value=59),
)
_OLD_KEYS = ("trade_date", "data_basis", "decision_time", "decision_times")


# ---------------------------------------------------------------------------
# Property DI-6（Decision）: 旧 Decision 迁移零残留且语义一致
# ---------------------------------------------------------------------------
# Feature: decision-instant-unification, Property DI-6: 迁移往返零残留（Decision）
# 对任意旧 Decision（含 trade_date），迁移后 decision_bar_dt 落在原 trade_date 当日、
# bar_freq=1d，且不含 trade_date；scheme/action 等语义保留；迁移幂等。
# Validates: Requirements 5.1, 5.3
@settings(max_examples=100)
@given(d=_dates, scheme=st.text(min_size=1, max_size=8), action=st.sampled_from(["buy", "sell", "hold"]))
def test_property_di6_decision_migration(d, scheme, action):
    old = {
        "signal_id": f"{d.isoformat()}:{scheme}",
        "trade_date": d.isoformat(),
        "scheme": scheme,
        "action": action,
        "vt_symbol": "000001.SZSE",
        "volume": 100,
        "price": 10.0,
        "signal": 0.7,
        "reason": "x",
    }
    new = migrate_decision(old)
    assert "trade_date" not in new
    assert new["bar_freq"] == "1d"
    assert datetime.fromisoformat(new["decision_bar_dt"]).date() == d
    assert datetime.fromisoformat(new["as_of"]).time() == SESSION_CLOSE
    assert new["scheme"] == scheme and new["action"] == action
    # 幂等
    assert migrate_decision(new) == new


# ---------------------------------------------------------------------------
# Property DI-6（Plan）: 旧 Plan 迁移零残留且生效时刻集合一致
# ---------------------------------------------------------------------------
# Feature: decision-instant-unification, Property DI-6: 迁移往返零残留（Plan）
# 对任意旧 Plan（data_basis/decision_time(s)），迁移后 trigger_times = 旧生效时点
# 去重升序、bar_freq=1d，且不含旧字段；迁移幂等。
# Validates: Requirements 5.2, 5.3
@settings(max_examples=100)
@given(
    legacy_single=_hhmm,
    multi=st.lists(_hhmm, max_size=4),
    data_basis=st.sampled_from(["closed_t", "prev_close"]),
)
def test_property_di6_plan_migration(legacy_single, multi, data_basis):
    old = {
        "plan_id": "p1",
        "name": "计划",
        "decision_time": legacy_single,
        "decision_times": multi,
        "data_basis": data_basis,
        "model": "m",
        "vt_symbol": "000001.SZSE",
        "scheme": "s",
    }
    new = migrate_plan(old)
    for key in _OLD_KEYS:
        assert key not in new
    assert new["bar_freq"] == "1d"
    expected = sorted({t for t in (multi or [legacy_single]) if t}) or ["15:05"]
    assert new["trigger_times"] == expected
    # 幂等
    assert migrate_plan(new) == new
