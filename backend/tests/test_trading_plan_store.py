"""TradingPlanStore 存储单元测试（交易计划自动化 任务 3）。"""

from __future__ import annotations

import json

from aitrade.live.trading_plan import TradingPlan, TradingPlanStore


def _make_plan(plan_id: str = "p1") -> TradingPlan:
    return TradingPlan(
        plan_id=plan_id,
        name="测试计划",
        model="m1",
        vt_symbol="000001.SZSE",
        scheme="eod_buy_v1",
        portfolio={"portfolio_value": 1000000},
        risk={"max_total_position_ratio": 0.95},
        notify_channels=["dingtalk"],
        enabled=True,
    )


def test_save_get_roundtrip(tmp_path) -> None:
    store = TradingPlanStore(tmp_path)
    plan = _make_plan()
    store.save(plan)
    loaded = store.get("p1")
    assert loaded == plan  # dataclass 字段逐项相等


def test_get_missing_returns_none(tmp_path) -> None:
    store = TradingPlanStore(tmp_path)
    assert store.get("nope") is None


def test_delete(tmp_path) -> None:
    store = TradingPlanStore(tmp_path)
    store.save(_make_plan())
    assert store.delete("p1") is True
    assert store.get("p1") is None
    assert store.delete("p1") is False  # 再删返回 False


def test_list_all(tmp_path) -> None:
    store = TradingPlanStore(tmp_path)
    store.save(_make_plan("p1"))
    store.save(_make_plan("p2"))
    ids = {p.plan_id for p in store.list_all()}
    assert ids == {"p1", "p2"}


def test_plan_json_has_no_credentials(tmp_path) -> None:
    store = TradingPlanStore(tmp_path)
    store.save(_make_plan())
    raw = (tmp_path / "p1.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    # notify_channels 仅通道名
    assert payload["notify_channels"] == ["dingtalk"]
    # 不含任何 webhook/secret/token 字段
    assert "webhook" not in raw.lower()
    assert "secret" not in raw.lower()
    assert "token" not in raw.lower()
