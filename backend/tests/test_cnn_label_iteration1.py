"""
迭代 1 验收测试：CNN label 可配置化（计价口径 price_ref / 去噪阈值 / 无前视）。

覆盖：
1. _compute_label_return：close（收盘到收盘）与 next_open（次开盘到次开盘）口径，及越界返回 None。
2. _label_from_return：去噪 dead-zone 与中性样本策略（drop / negative）。
3. _label_future_index：next_bar / horizon_bars / next_session_close 的未来索引定位。
4. _normalize_label_spec：默认值与非法值兜底。
5. build_dataset 集成（monkeypatch 合成行情）：
   - price_ref 真正改变标签（close 与 next_open 在构造数据下给出相反标签）；
   - next_open 因需要 anchor+1 / future+1 开盘价，样本数不多于 close（无前视的边界体现）；
   - info 正确回传 label_spec / price_ref；
   - 阈值去噪：drop 全中性→报错，negative→并入下跌类。

纯函数测试不依赖任何数据；集成测试用 monkeypatch 注入合成行情。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from aitrade.cnn import dataset as cnn_dataset
from aitrade.cnn.dataset import (
    _compute_label_return,
    _label_from_return,
    _label_future_index,
    _normalize_label_spec,
    build_dataset,
)


# ---------------------------------------------------------------------------
# 1. _compute_label_return
# ---------------------------------------------------------------------------
def test_compute_label_return_close_ref() -> None:
    close = np.array([100.0, 110.0, 121.0])
    open_ = np.array([99.0, 100.0, 100.0])
    # close 口径：close[future]/close[anchor]-1
    r = _compute_label_return(0, 1, "close", open_, close, total_steps=3)
    assert r == pytest.approx((110.0 - 100.0) / 100.0)


def test_compute_label_return_next_open_ref() -> None:
    # next_open 口径：entry=open[anchor+1], exit=open[future+1]
    open_ = np.array([99.0, 100.0, 105.0, 110.0])
    close = np.array([100.0, 101.0, 102.0, 103.0])
    # anchor=0, future=1 → entry=open[1]=100, exit=open[2]=105
    r = _compute_label_return(0, 1, "next_open", open_, close, total_steps=4)
    assert r == pytest.approx((105.0 - 100.0) / 100.0)


def test_compute_label_return_next_open_out_of_range_returns_none() -> None:
    open_ = np.array([99.0, 100.0, 105.0])
    close = np.array([100.0, 101.0, 102.0])
    # future=2 → exit_index=3 >= total_steps(3) → None（无次日开盘，避免前视/越界）
    assert _compute_label_return(1, 2, "next_open", open_, close, total_steps=3) is None


def test_compute_label_return_next_close_ref() -> None:
    # next_close 口径：entry=close[anchor+1], exit=close[future+1]
    open_ = np.array([99.0, 100.0, 105.0, 110.0])
    close = np.array([100.0, 101.0, 110.0, 121.0])
    # anchor=0, future=1 → entry=close[1]=101, exit=close[2]=110
    r = _compute_label_return(0, 1, "next_close", open_, close, total_steps=4)
    assert r == pytest.approx((110.0 - 101.0) / 101.0)


def test_compute_label_return_next_close_out_of_range_returns_none() -> None:
    open_ = np.array([99.0, 100.0, 105.0])
    close = np.array([100.0, 101.0, 102.0])
    # future=2 → exit_index=3 >= total_steps(3) → None
    assert _compute_label_return(1, 2, "next_close", open_, close, total_steps=3) is None


def test_compute_label_return_next_vwap_ref() -> None:
    # next_vwap 口径：entry=vwap[anchor+1], exit=vwap[future+1]
    open_ = np.array([99.0, 100.0, 105.0, 110.0])
    close = np.array([100.0, 101.0, 102.0, 103.0])
    vwap = np.array([100.0, 104.0, 120.0, 130.0])
    # anchor=0, future=1 → entry=vwap[1]=104, exit=vwap[2]=120
    r = _compute_label_return(0, 1, "next_vwap", open_, close, total_steps=4, vwap_series=vwap)
    assert r == pytest.approx((120.0 - 104.0) / 104.0)


def test_compute_label_return_next_vwap_falls_back_to_close_when_missing() -> None:
    # 未提供 vwap_series 时回退到 close 序列（与 build_dataset 的缺失回退一致）
    open_ = np.array([99.0, 100.0, 105.0, 110.0])
    close = np.array([100.0, 101.0, 110.0, 121.0])
    r = _compute_label_return(0, 1, "next_vwap", open_, close, total_steps=4)
    assert r == pytest.approx((110.0 - 101.0) / 101.0)


# ---------------------------------------------------------------------------
# 2. _label_from_return
# ---------------------------------------------------------------------------
def test_label_from_return_no_threshold_is_sign() -> None:
    assert _label_from_return(0.01, 0.0, "drop") == 1.0
    assert _label_from_return(-0.01, 0.0, "drop") == 0.0
    assert _label_from_return(0.0, 0.0, "drop") == 0.0


def test_label_from_return_deadzone_drop() -> None:
    # |收益| <= 阈值 → drop 返回 None
    assert _label_from_return(0.005, 0.01, "drop") is None
    assert _label_from_return(0.02, 0.01, "drop") == 1.0
    assert _label_from_return(-0.02, 0.01, "drop") == 0.0


def test_label_from_return_deadzone_negative() -> None:
    # 中性样本并入下跌类
    assert _label_from_return(0.005, 0.01, "negative") == 0.0


# ---------------------------------------------------------------------------
# 3. _label_future_index
# ---------------------------------------------------------------------------
def _days(n: int) -> list[datetime]:
    base = datetime(2026, 1, 5)
    return [base + timedelta(days=i) for i in range(n)]


def test_label_future_index_next_bar() -> None:
    days = _days(5)
    assert _label_future_index(0, days, {"mode": "next_bar"}, input_interval="d") == 1
    # 末根无下一根 → None
    assert _label_future_index(4, days, {"mode": "next_bar"}, input_interval="d") is None


def test_label_future_index_horizon_bars() -> None:
    days = _days(6)
    assert _label_future_index(1, days, {"mode": "horizon_bars", "horizon": 3}, input_interval="d") == 4
    assert _label_future_index(4, days, {"mode": "horizon_bars", "horizon": 3}, input_interval="d") is None


def test_label_future_index_next_session_close_daily() -> None:
    # 日线下每天一根 bar：次日收盘 = anchor+1
    days = _days(4)
    assert _label_future_index(0, days, {"mode": "next_session_close"}, input_interval="d") == 1
    # 最后一天无次日 → None
    assert _label_future_index(3, days, {"mode": "next_session_close"}, input_interval="d") is None


# ---------------------------------------------------------------------------
# 4. _normalize_label_spec
# ---------------------------------------------------------------------------
def test_normalize_label_spec_defaults_and_fallback() -> None:
    spec = _normalize_label_spec(None)
    assert spec["mode"] == "next_bar"
    assert spec["threshold"] == 0.0
    assert spec["neutral_policy"] == "drop"
    assert spec["price_ref"] == "close"

    # 非法值兜底
    bad = _normalize_label_spec({"price_ref": "xxx", "neutral_policy": "yyy", "mode": "horizon_bars"})
    assert bad["price_ref"] == "close"
    assert bad["neutral_policy"] == "drop"
    assert bad["horizon"] == 1  # horizon_bars 缺省补 1

    # 新增可执行口径合法透传
    for ref in ("next_open", "next_close", "next_vwap"):
        assert _normalize_label_spec({"price_ref": ref})["price_ref"] == ref


# ---------------------------------------------------------------------------
# 5. build_dataset 集成（合成行情）
# ---------------------------------------------------------------------------
def _frame(opens: list[float], closes: list[float]) -> pl.DataFrame:
    start = datetime(2026, 1, 5)
    rows = []
    for i, (o, c) in enumerate(zip(opens, closes)):
        rows.append({
            "datetime": start + timedelta(days=i),
            "open": o,
            "high": max(o, c) + 1.0,
            "low": min(o, c) - 1.0,
            "close": c,
            "volume": 1_000 + i,
            "turnover": (1_000 + i) * c,
            "open_interest": float(i),
        })
    return pl.DataFrame(rows)


def _patch_loader(monkeypatch, frame: pl.DataFrame) -> None:
    def fake_loader(vt_symbol, start, end, *, input_data_kind, input_interval):
        return frame

    monkeypatch.setattr(cnn_dataset, "_load_market_frame", fake_loader)


def test_price_ref_changes_labels(monkeypatch) -> None:
    n = 30
    # close 单调上行、open 单调下行 → 两种口径给出相反方向标签
    closes = [100.0 + i for i in range(n)]
    opens = [100.0 - i for i in range(n)]
    frame = _frame(opens, closes)
    _patch_loader(monkeypatch, frame)

    common = dict(
        vt_symbols=["AAA.SSE"], start=date(2026, 1, 5), end=date(2026, 3, 31),
        lookback=5, target_symbol="AAA.SSE", input_data_kind="bar", input_interval="d",
    )

    _, y_close, _, info_close = build_dataset(label_spec={"mode": "next_bar", "price_ref": "close"}, **common)
    _, y_open, _, info_open = build_dataset(label_spec={"mode": "next_bar", "price_ref": "next_open"}, **common)

    # close 口径：close 单调上行 → 全为上涨
    assert y_close.mean() == 1.0
    # next_open 口径：open 单调下行 → 全为下跌
    assert y_open.mean() == 0.0
    # info 正确回传口径
    assert info_close["price_ref"] == "close"
    assert info_open["price_ref"] == "next_open"
    # next_open 需要 future+1 的开盘价，尾部边界更紧 → 样本数不多于 close
    assert len(y_open) <= len(y_close)


def test_threshold_deadzone_drop_raises_when_all_neutral(monkeypatch) -> None:
    n = 30
    # 极小涨幅（每步 +0.01%），阈值 1% → 全部落入 dead-zone
    closes = [100.0 * (1.0001 ** i) for i in range(n)]
    opens = closes
    _patch_loader(monkeypatch, _frame(opens, closes))

    with pytest.raises(ValueError, match="没有生成任何有效样本"):
        build_dataset(
            vt_symbols=["AAA.SSE"], start=date(2026, 1, 5), end=date(2026, 3, 31),
            lookback=5, target_symbol="AAA.SSE", input_data_kind="bar", input_interval="d",
            label_spec={"mode": "next_bar", "price_ref": "close", "threshold": 0.01, "neutral_policy": "drop"},
        )


def test_threshold_deadzone_negative_keeps_samples(monkeypatch) -> None:
    n = 30
    closes = [100.0 * (1.0001 ** i) for i in range(n)]
    opens = closes
    _patch_loader(monkeypatch, _frame(opens, closes))

    _, y, _, info = build_dataset(
        vt_symbols=["AAA.SSE"], start=date(2026, 1, 5), end=date(2026, 3, 31),
        lookback=5, target_symbol="AAA.SSE", input_data_kind="bar", input_interval="d",
        label_spec={"mode": "next_bar", "price_ref": "close", "threshold": 0.01, "neutral_policy": "negative"},
    )
    # 中性并入下跌类 → 全 0，且不丢样本
    assert y.mean() == 0.0
    assert info["skipped_for_neutral"] == 0


def test_next_vwap_uses_turnover_over_volume(monkeypatch) -> None:
    # 构造 vwap(=turnover/volume) 单调上行、close 单调下行 → next_vwap 与 next_close 标签相反，
    # 证明 next_vwap 真正用了均价口径而非收盘价。
    n = 30
    start = datetime(2026, 1, 5)
    rows = []
    for i in range(n):
        close = 200.0 - i          # 收盘单调下行
        vwap = 100.0 + i           # 均价单调上行
        volume = 1_000.0
        rows.append({
            "datetime": start + timedelta(days=i),
            "open": close,
            "high": max(close, vwap) + 1.0,
            "low": min(close, vwap) - 1.0,
            "close": close,
            "volume": volume,
            "turnover": vwap * volume,   # turnover/volume == vwap
            "open_interest": float(i),
        })
    frame = pl.DataFrame(rows)
    _patch_loader(monkeypatch, frame)

    common = dict(
        vt_symbols=["AAA.SSE"], start=date(2026, 1, 5), end=date(2026, 3, 31),
        lookback=5, target_symbol="AAA.SSE", input_data_kind="bar", input_interval="d",
    )
    _, y_vwap, _, info_vwap = build_dataset(label_spec={"mode": "next_bar", "price_ref": "next_vwap"}, **common)
    _, y_close, _, _ = build_dataset(label_spec={"mode": "next_bar", "price_ref": "next_close"}, **common)

    # vwap 单调上行 → 全涨；next_close 单调下行 → 全跌
    assert y_vwap.mean() == 1.0
    assert y_close.mean() == 0.0
    assert info_vwap["price_ref"] == "next_vwap"


def test_label_spec_persisted_in_info(monkeypatch) -> None:
    n = 25
    closes = [100.0 + i for i in range(n)]
    opens = [c - 0.5 for c in closes]
    _patch_loader(monkeypatch, _frame(opens, closes))

    _, _, _, info = build_dataset(
        vt_symbols=["AAA.SSE"], start=date(2026, 1, 5), end=date(2026, 3, 31),
        lookback=5, target_symbol="AAA.SSE", input_data_kind="bar", input_interval="d",
        label_spec={"mode": "horizon_bars", "horizon": 2, "price_ref": "next_open"},
    )
    spec = info["label_spec"]
    assert spec["mode"] == "horizon_bars"
    assert spec["horizon"] == 2
    assert spec["price_ref"] == "next_open"
