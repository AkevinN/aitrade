"""
PositionBook 单元测试（Phase 3 M2）。

覆盖：apply 数学 / 卖超拒绝且账本不变（属性测试）/ 防重复确认 / 空账本加载 / 原子写。
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aitrade.live.position_book import PortfolioState, PositionBook
from aitrade.live.rebalance_decision import RebalanceDecision, RebalanceItem


# ---------------------------------------------------------------------------
# 辅助工厂
# ---------------------------------------------------------------------------


def _make_decision(
    signal_id: str = "sig_001",
    portfolio_id: str = "p1",
    items: list[RebalanceItem] | None = None,
) -> RebalanceDecision:
    if items is None:
        items = []
    return RebalanceDecision(
        signal_id=signal_id,
        decision_bar_dt="2026-01-01T15:00:00",
        as_of="2026-01-01T15:05:00",
        bar_freq="1d",
        scheme="rule:etf",
        portfolio_id=portfolio_id,
        items=items,
        target_portfolio={},
    )


def _item(sym: str, action: str, volume: int) -> RebalanceItem:
    return RebalanceItem(vt_symbol=sym, action=action, volume=volume)


# ---------------------------------------------------------------------------
# 空账本加载
# ---------------------------------------------------------------------------


def test_load_missing_returns_empty(tmp_path) -> None:
    book = PositionBook(tmp_path)
    state = book.load("p_new")
    assert state.portfolio_id == "p_new"
    assert state.positions == {}
    assert state.last_signal_id == ""


def test_load_saved_state(tmp_path) -> None:
    book = PositionBook(tmp_path)
    state = PortfolioState(portfolio_id="p1", positions={"000001.SZSE": 1000})
    book.save(state)
    loaded = book.load("p1")
    assert loaded.positions == {"000001.SZSE": 1000}


# ---------------------------------------------------------------------------
# apply_rebalance 数学正确性
# ---------------------------------------------------------------------------


def test_apply_buy_increases_position(tmp_path) -> None:
    book = PositionBook(tmp_path)
    d = _make_decision(items=[_item("000001.SZSE", "buy", 1000)])
    state = book.apply_rebalance("p1", d)
    assert state.positions["000001.SZSE"] == 1000
    assert state.last_signal_id == "sig_001"
    assert state.updated_at != ""


def test_apply_sell_decreases_position(tmp_path) -> None:
    book = PositionBook(tmp_path)
    # 先给账本打底
    init_state = PortfolioState(portfolio_id="p1", positions={"000001.SZSE": 2000})
    book.save(init_state)

    d = _make_decision(items=[_item("000001.SZSE", "sell", 500)])
    state = book.apply_rebalance("p1", d)
    assert state.positions["000001.SZSE"] == 1500


def test_apply_sell_to_zero_removes_symbol(tmp_path) -> None:
    """卖至归零的 symbol 从 dict 移除。"""
    book = PositionBook(tmp_path)
    init_state = PortfolioState(portfolio_id="p1", positions={"000001.SZSE": 500})
    book.save(init_state)

    d = _make_decision(items=[_item("000001.SZSE", "sell", 500)])
    state = book.apply_rebalance("p1", d)
    assert "000001.SZSE" not in state.positions


def test_apply_multiple_items(tmp_path) -> None:
    book = PositionBook(tmp_path)
    init_state = PortfolioState(portfolio_id="p1", positions={"510300.SSE": 1000})
    book.save(init_state)

    d = _make_decision(
        items=[
            _item("000001.SZSE", "buy", 2000),
            _item("510300.SSE", "sell", 300),
        ]
    )
    state = book.apply_rebalance("p1", d)
    assert state.positions["000001.SZSE"] == 2000
    assert state.positions["510300.SSE"] == 700


def test_apply_persists_to_disk(tmp_path) -> None:
    """apply_rebalance 成功后，重新 load 应看到新持仓。"""
    book = PositionBook(tmp_path)
    d = _make_decision(items=[_item("000001.SZSE", "buy", 1000)])
    book.apply_rebalance("p1", d)
    reloaded = book.load("p1")
    assert reloaded.positions["000001.SZSE"] == 1000


# ---------------------------------------------------------------------------
# 卖超拒绝（整笔原子性）
# ---------------------------------------------------------------------------


def test_oversell_raises_valueerror(tmp_path) -> None:
    book = PositionBook(tmp_path)
    init_state = PortfolioState(portfolio_id="p1", positions={"000001.SZSE": 100})
    book.save(init_state)

    d = _make_decision(items=[_item("000001.SZSE", "sell", 200)])
    with pytest.raises(ValueError, match="超过当前持仓"):
        book.apply_rebalance("p1", d)


def test_oversell_leaves_book_unchanged(tmp_path) -> None:
    """超卖失败时账本不变（原子性）。"""
    book = PositionBook(tmp_path)
    init_positions = {"000001.SZSE": 100, "510300.SSE": 500}
    init_state = PortfolioState(portfolio_id="p1", positions=dict(init_positions))
    book.save(init_state)

    # 买 000001 正常，但卖 510300 超额 → 整笔应拒绝
    d = _make_decision(
        items=[
            _item("000001.SZSE", "buy", 500),   # 正常
            _item("510300.SSE", "sell", 600),    # 超额
        ]
    )
    with pytest.raises(ValueError):
        book.apply_rebalance("p1", d)

    # 账本应保持原样（buy 也不应该被部分应用）
    after = book.load("p1")
    assert after.positions == init_positions


def test_oversell_on_zero_position_raises(tmp_path) -> None:
    """空账本中尝试卖出 → 超卖拒绝。"""
    book = PositionBook(tmp_path)
    d = _make_decision(items=[_item("000001.SZSE", "sell", 100)])
    with pytest.raises(ValueError, match="超过当前持仓"):
        book.apply_rebalance("p1", d)


# ---------------------------------------------------------------------------
# 卖超拒绝 + 账本不变（属性测试：随机 items 序列）
# ---------------------------------------------------------------------------

_VT_SYMBOLS = ["000001.SZSE", "510300.SSE", "510500.SSE"]

_item_strategy = st.builds(
    RebalanceItem,
    vt_symbol=st.sampled_from(_VT_SYMBOLS),
    action=st.sampled_from(["buy", "sell"]),
    volume=st.integers(min_value=100, max_value=2000),
    price=st.none(),
    signal=st.none(),
    reason=st.just(""),
)


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(items=st.lists(_item_strategy, min_size=1, max_size=8))
def test_property_oversell_does_not_partial_apply(tmp_path, items) -> None:
    """属性：apply_rebalance 失败时账本与失败前完全一致。

    每次 hypothesis 生成新 items 时，用独立临时目录隔离账本，避免 function-scoped
    fixture 不重置的副作用（每次都新建账本 + 固定初始持仓）。
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        book = PositionBook(Path(tmp_dir) / "book")
        init_positions = {sym: 1000 for sym in _VT_SYMBOLS}
        init_state = PortfolioState(portfolio_id="prop_p", positions=dict(init_positions))
        book.save(init_state)
        snapshot = copy.deepcopy(init_positions)

        d = _make_decision(signal_id="prop_sig", portfolio_id="prop_p", items=items)
        try:
            book.apply_rebalance("prop_p", d)
        except ValueError:
            after = book.load("prop_p")
            assert after.positions == snapshot, (
                f"失败后账本被部分修改！before={snapshot}, after={after.positions}"
            )


# ---------------------------------------------------------------------------
# 防重复确认
# ---------------------------------------------------------------------------


def test_duplicate_confirm_raises(tmp_path) -> None:
    book = PositionBook(tmp_path)
    d = _make_decision(items=[_item("000001.SZSE", "buy", 100)])
    book.apply_rebalance("p1", d)

    # 再次用同一 signal_id 确认
    with pytest.raises(ValueError, match="已确认过"):
        book.apply_rebalance("p1", d)


def test_different_signal_id_allowed(tmp_path) -> None:
    """不同 signal_id 不触发防重复逻辑。"""
    book = PositionBook(tmp_path)
    d1 = _make_decision(signal_id="sig_001", items=[_item("000001.SZSE", "buy", 100)])
    d2 = _make_decision(signal_id="sig_002", items=[_item("000001.SZSE", "buy", 200)])
    book.apply_rebalance("p1", d1)
    state = book.apply_rebalance("p1", d2)
    assert state.positions["000001.SZSE"] == 300
    assert state.last_signal_id == "sig_002"


# ---------------------------------------------------------------------------
# 原子写（无残余临时文件）
# ---------------------------------------------------------------------------


def test_save_leaves_no_tmp_files(tmp_path) -> None:
    book = PositionBook(tmp_path)
    state = PortfolioState(portfolio_id="p1", positions={"X.SSE": 100})
    book.save(state)
    tmp_files = list(tmp_path.glob("*.tmp.json"))
    assert tmp_files == []
