"""
RebalanceStore 单元测试（Phase 3 M2）。

覆盖：实体往返 / save_if_absent 幂等 / 归档式删除 / update_status 原子性。
"""

from __future__ import annotations

import json

from aitrade.live.rebalance_decision import (
    RebalanceDecision,
    RebalanceItem,
    RebalanceStore,
)


def _make_item(sym: str = "000001.SZSE", action: str = "buy", volume: int = 1000) -> RebalanceItem:
    return RebalanceItem(
        vt_symbol=sym,
        action=action,
        volume=volume,
        price=10.5,
        signal=0.8,
        reason="测试",
    )


def _make_decision(signal_id: str = "rule_etf_20260101_1d_p1") -> RebalanceDecision:
    return RebalanceDecision(
        signal_id=signal_id,
        decision_bar_dt="2026-01-01T15:00:00",
        as_of="2026-01-01T15:05:00",
        bar_freq="1d",
        scheme="rule:etf_momentum",
        portfolio_id="p1",
        items=[_make_item("000001.SZSE", "buy", 1000), _make_item("510300.SSE", "sell", 500)],
        target_portfolio={"000001.SZSE": 1000},
        risk_summary=[{"check": "max_position", "passed": True, "detail": "OK"}],
    )


# ---------------------------------------------------------------------------
# 实体往返
# ---------------------------------------------------------------------------


def test_save_get_roundtrip(tmp_path) -> None:
    store = RebalanceStore(tmp_path)
    d = _make_decision()
    store.save(d)
    loaded = store.get(d.signal_id)
    assert loaded is not None
    assert loaded.signal_id == d.signal_id
    assert loaded.scheme == d.scheme
    assert loaded.portfolio_id == d.portfolio_id
    assert len(loaded.items) == 2
    # items 嵌套还原
    assert loaded.items[0].vt_symbol == "000001.SZSE"
    assert loaded.items[0].action == "buy"
    assert loaded.items[1].vt_symbol == "510300.SSE"
    assert loaded.items[1].action == "sell"


def test_get_missing_returns_none(tmp_path) -> None:
    store = RebalanceStore(tmp_path)
    assert store.get("not_exist") is None


def test_list_ids_and_list_all(tmp_path) -> None:
    store = RebalanceStore(tmp_path)
    d1 = _make_decision("id_a")
    d2 = _make_decision("id_b")
    store.save(d1)
    store.save(d2)
    ids = store.list_ids()
    assert set(ids) == {"id_a", "id_b"}
    all_decisions = store.list_all()
    assert {d.signal_id for d in all_decisions} == {"id_a", "id_b"}


def test_json_file_has_correct_structure(tmp_path) -> None:
    """落盘 JSON 应含必要字段且 items 保持列表结构。"""
    store = RebalanceStore(tmp_path)
    d = _make_decision()
    store.save(d)
    raw = json.loads((tmp_path / "rule_etf_20260101_1d_p1.json").read_text(encoding="utf-8"))
    assert raw["signal_id"] == d.signal_id
    assert raw["status"] == "proposed"
    assert isinstance(raw["items"], list)
    assert raw["items"][0]["vt_symbol"] == "000001.SZSE"


# ---------------------------------------------------------------------------
# save_if_absent 幂等语义
# ---------------------------------------------------------------------------


def test_save_if_absent_first_call_writes(tmp_path) -> None:
    store = RebalanceStore(tmp_path)
    d = _make_decision()
    saved, result = store.save_if_absent(d)
    assert saved is True
    assert result.signal_id == d.signal_id
    assert store.exists(d.signal_id)


def test_save_if_absent_second_call_skips(tmp_path) -> None:
    store = RebalanceStore(tmp_path)
    d = _make_decision()
    store.save_if_absent(d)

    # 修改 status，再次调用不应覆盖
    d2 = _make_decision()
    d2.status = "confirmed"
    saved, result = store.save_if_absent(d2)
    assert saved is False
    # 返回的是旧版本（proposed），不是新版本（confirmed）
    assert result.status == "proposed"


def test_save_if_absent_returns_existing_on_duplicate(tmp_path) -> None:
    store = RebalanceStore(tmp_path)
    d = _make_decision()
    store.save(d)

    d_new = _make_decision()
    saved, existing = store.save_if_absent(d_new)
    assert saved is False
    assert existing.signal_id == d.signal_id


# ---------------------------------------------------------------------------
# 归档式删除
# ---------------------------------------------------------------------------


def test_delete_archives_file(tmp_path) -> None:
    store = RebalanceStore(tmp_path)
    d = _make_decision()
    store.save(d)
    archived = store.delete(d.signal_id)
    assert archived is not None
    assert archived.exists()
    # 原文件不再存在
    assert not store.exists(d.signal_id)
    # archive/ 子目录
    assert archived.parent.name == "archive"


def test_delete_missing_returns_none(tmp_path) -> None:
    store = RebalanceStore(tmp_path)
    assert store.delete("not_exist") is None


def test_delete_releases_idempotency_slot(tmp_path) -> None:
    """归档后同一 signal_id 可重新 save_if_absent 写入（幂等占位解除）。"""
    store = RebalanceStore(tmp_path)
    d = _make_decision()
    store.save(d)
    store.delete(d.signal_id)

    # 再次写入（幂等占位已解除）
    d2 = _make_decision()
    saved, result = store.save_if_absent(d2)
    assert saved is True


def test_list_ids_excludes_archived(tmp_path) -> None:
    store = RebalanceStore(tmp_path)
    d = _make_decision("id_to_archive")
    store.save(d)
    store.delete("id_to_archive")
    assert "id_to_archive" not in store.list_ids()


# ---------------------------------------------------------------------------
# update_status 原子性
# ---------------------------------------------------------------------------


def test_update_status_changes_status(tmp_path) -> None:
    store = RebalanceStore(tmp_path)
    d = _make_decision()
    store.save(d)
    updated = store.update_status(d.signal_id, "confirmed", "2026-01-01T15:10:00")
    assert updated is not None
    assert updated.status == "confirmed"
    assert updated.confirmed_at == "2026-01-01T15:10:00"


def test_update_status_persists_to_disk(tmp_path) -> None:
    """update_status 写磁盘后，get 读回应看到新状态。"""
    store = RebalanceStore(tmp_path)
    d = _make_decision()
    store.save(d)
    store.update_status(d.signal_id, "confirmed", "2026-01-01T15:10:00")
    reloaded = store.get(d.signal_id)
    assert reloaded is not None
    assert reloaded.status == "confirmed"
    assert reloaded.confirmed_at == "2026-01-01T15:10:00"


def test_update_status_missing_returns_none(tmp_path) -> None:
    store = RebalanceStore(tmp_path)
    assert store.update_status("not_exist", "confirmed") is None


def test_update_status_no_tmp_file_left(tmp_path) -> None:
    """原子写完成后，.tmp 临时文件应不存在。"""
    store = RebalanceStore(tmp_path)
    d = _make_decision()
    store.save(d)
    store.update_status(d.signal_id, "confirmed")
    tmp_files = list(tmp_path.glob("*.tmp.json"))
    assert tmp_files == []


def test_update_status_invalid_value_raises_valueerror(tmp_path) -> None:
    """非法 status 值应抛出 ValueError 且含中文说明。"""
    import pytest

    store = RebalanceStore(tmp_path)
    d = _make_decision()
    store.save(d)
    with pytest.raises(ValueError, match="非法 status 值"):
        store.update_status(d.signal_id, "invalid_status")
