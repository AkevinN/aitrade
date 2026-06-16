"""
cb_double_low 信号源测试（Task 4.3）。

覆盖：
1. 双低分手算对照（signal = -(close + premium_rate*100)）
2. 价格过滤（max_price）：高价转债被剔除
3. 评级过滤（min_rating）：低评级转债被剔除
4. 规模过滤（min_issue_scale）：小规模转债被剔除
5. 上市天数过滤（min_list_days）：新券被剔除
6. 溢价率历史缺失时使用快照值回退并发 logger.warning
7. 全空时抛 RuntimeError（中文，含 cb-terms/refresh 提示）
8. 评级序比较边界：AA- 含等于（既通过 AA-，也拒绝 A+）
9. 输出 schema 三列 + (datetime, vt_symbol) 升序
10. 通过 build_signal_source("cb_double_low") 全链路可构造
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import aitrade.rules  # noqa: F401  触发 "cb_double_low" 自注册

from aitrade.alpha.lab import AlphaLab
from aitrade.backtest.registry import build_signal_source


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------


def _make_bar_frame(prices: list[float], base_date: date | None = None) -> pl.DataFrame:
    """构造最小化日线 DataFrame（datetime / close + 其他必须列）。"""
    if base_date is None:
        base_date = date(2024, 1, 2)
    rows = []
    for i, p in enumerate(prices):
        dt = base_date + timedelta(days=i)
        rows.append(
            {
                "datetime": datetime(dt.year, dt.month, dt.day, 9, 30),
                "open": p,
                "high": p + 0.5,
                "low": p - 0.5,
                "close": p,
                "volume": 100_000.0,
                "turnover": p * 100_000.0,
                "open_interest": 0.0,
            }
        )
    return pl.DataFrame(rows)


def _save_bar(lab: AlphaLab, vt_symbol: str, prices: list[float], base_date: date | None = None) -> None:
    df = _make_bar_frame(prices, base_date=base_date)
    lab.save_bar_frame(vt_symbol, "d", df)


def _make_snapshot(
    entries: list[dict],
) -> pl.DataFrame:
    """构造 bond_zh_cov 风格的快照 DataFrame。

    每个 entry 可含：code / name / rating / issue_scale / premium_rate / list_date。
    """
    rows = []
    for e in entries:
        rows.append(
            {
                "债券代码": e.get("code", ""),
                "债券简称": e.get("name", ""),
                "信用评级": e.get("rating", "AA"),
                "发行规模": float(e.get("issue_scale", 5.0)),
                "转股溢价率": float(e.get("premium_rate", 10.0)),
                "上市时间": e.get("list_date", "2023-01-01"),
            }
        )
    return pl.DataFrame(rows)


def _make_premium_hist(dates: list[str], premiums: list[float]) -> pl.DataFrame:
    """构造 value_analysis 风格的溢价率历史 DataFrame（百分比形式）。"""
    return pl.DataFrame(
        {
            "日期": dates,
            "收盘价": [100.0] * len(dates),
            "转股溢价率": premiums,
        }
    )


class _FakeTermsStore:
    """测试用：内存中的 CBTermsStore 替代品。"""

    def __init__(
        self,
        snapshot: pl.DataFrame | None = None,
        premium_map: dict[str, pl.DataFrame] | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._premium_map: dict[str, pl.DataFrame] = premium_map or {}

    def load_snapshot(self) -> pl.DataFrame | None:
        return self._snapshot

    def load_premium_history(self, vt_symbol: str) -> pl.DataFrame | None:
        return self._premium_map.get(vt_symbol)


def _build(
    lab: AlphaLab,
    terms_store: _FakeTermsStore,
    **kwargs,
) -> object:
    """快捷方式：构造 cb_double_low 信号源（注入 _lab 和 _terms_store）。"""
    return build_signal_source(
        "cb_double_low",
        {"_lab": lab, "_terms_store": terms_store, **kwargs},
    )


# ---------------------------------------------------------------------------
# 测试 1：双低分手算对照
# ---------------------------------------------------------------------------


def test_signal_value_matches_manual_calculation(tmp_path: Path) -> None:
    """signal = -(close + premium_rate*100)，手算应相等。

    转债 A：close=108.0，premium_rate=12.5%（溢价率历史值 12.5，/100=0.125）
      → signal = -(108.0 + 0.125 * 100) = -(108.0 + 12.5) = -120.5

    转债 B：close=95.0，premium_rate=8.0%（溢价率历史值 8.0，/100=0.08）
      → signal = -(95.0 + 0.08 * 100) = -(95.0 + 8.0) = -103.0

    B 双低分更高（-103.0 > -120.5），TopK 降序优先选 B。
    """
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)

    _save_bar(lab, "113050.SSE", [108.0], base_date=base)  # 转债 A
    _save_bar(lab, "128093.SZSE", [95.0], base_date=base)  # 转债 B

    snapshot = _make_snapshot(
        [
            {"code": "113050", "rating": "AA", "issue_scale": 5.0, "premium_rate": 12.5, "list_date": "2023-01-01"},
            {"code": "128093", "rating": "AA", "issue_scale": 5.0, "premium_rate": 8.0, "list_date": "2023-01-01"},
        ]
    )
    premium_a = _make_premium_hist(["2024-01-02"], [12.5])  # 百分比 12.5 → /100 = 0.125
    premium_b = _make_premium_hist(["2024-01-02"], [8.0])   # 百分比 8.0 → /100 = 0.08

    terms_store = _FakeTermsStore(
        snapshot=snapshot,
        premium_map={"113050.SSE": premium_a, "128093.SZSE": premium_b},
    )

    provider = _build(lab, terms_store, max_price=130.0, min_rating="A")
    df = provider.predict(base, base)

    assert df.height == 2

    row_a = df.filter(pl.col("vt_symbol") == "113050.SSE")
    row_b = df.filter(pl.col("vt_symbol") == "128093.SZSE")

    assert row_a.height == 1
    assert row_b.height == 1

    # 手算验证（premium history 百分比值 /100 后再 *100 = 原值）
    assert float(row_a["signal"][0]) == pytest.approx(-(108.0 + 12.5), rel=1e-6)
    assert float(row_b["signal"][0]) == pytest.approx(-(95.0 + 8.0), rel=1e-6)

    # B 的 signal 更高（更双低）
    assert float(row_b["signal"][0]) > float(row_a["signal"][0])


# ---------------------------------------------------------------------------
# 测试 2：价格过滤（max_price）
# ---------------------------------------------------------------------------


def test_max_price_filter_positive(tmp_path: Path) -> None:
    """close < max_price 的转债应出现在结果中。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)
    _save_bar(lab, "113050.SSE", [120.0], base_date=base)

    snapshot = _make_snapshot([{"code": "113050", "premium_rate": 10.0}])
    premium = _make_premium_hist(["2024-01-02"], [10.0])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={"113050.SSE": premium})

    provider = _build(lab, terms_store, max_price=130.0)
    df = provider.predict(base, base)
    assert df.height == 1


def test_max_price_filter_negative(tmp_path: Path) -> None:
    """close > max_price 的转债应被剔除。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)
    _save_bar(lab, "113050.SSE", [135.0], base_date=base)  # 超过 max_price=130

    snapshot = _make_snapshot([{"code": "113050", "premium_rate": 5.0}])
    premium = _make_premium_hist(["2024-01-02"], [5.0])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={"113050.SSE": premium})

    provider = _build(lab, terms_store, max_price=130.0)
    with pytest.raises(RuntimeError, match="cb_double_low"):
        provider.predict(base, base)


# ---------------------------------------------------------------------------
# 测试 3：评级过滤（min_rating）
# ---------------------------------------------------------------------------


def test_rating_filter_positive(tmp_path: Path) -> None:
    """rating=AA 在 min_rating=AA- 时应通过。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)
    _save_bar(lab, "113050.SSE", [108.0], base_date=base)

    snapshot = _make_snapshot([{"code": "113050", "rating": "AA", "premium_rate": 10.0}])
    premium = _make_premium_hist(["2024-01-02"], [10.0])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={"113050.SSE": premium})

    provider = _build(lab, terms_store, min_rating="AA-", max_price=130.0)
    df = provider.predict(base, base)
    assert df.height == 1


def test_rating_filter_negative(tmp_path: Path) -> None:
    """rating=A+ 在 min_rating=AA- 时应被剔除。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)
    _save_bar(lab, "113050.SSE", [108.0], base_date=base)

    snapshot = _make_snapshot([{"code": "113050", "rating": "A+", "premium_rate": 10.0}])
    premium = _make_premium_hist(["2024-01-02"], [10.0])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={"113050.SSE": premium})

    provider = _build(lab, terms_store, min_rating="AA-", max_price=130.0)
    with pytest.raises(RuntimeError, match="cb_double_low"):
        provider.predict(base, base)


# ---------------------------------------------------------------------------
# 测试 4：规模过滤（min_issue_scale）
# ---------------------------------------------------------------------------


def test_issue_scale_filter_positive(tmp_path: Path) -> None:
    """issue_scale=5.0 在 min_issue_scale=3.0 时应通过。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)
    _save_bar(lab, "113050.SSE", [108.0], base_date=base)

    snapshot = _make_snapshot([{"code": "113050", "issue_scale": 5.0, "premium_rate": 10.0}])
    premium = _make_premium_hist(["2024-01-02"], [10.0])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={"113050.SSE": premium})

    provider = _build(lab, terms_store, min_issue_scale=3.0, max_price=130.0)
    df = provider.predict(base, base)
    assert df.height == 1


def test_issue_scale_filter_negative(tmp_path: Path) -> None:
    """issue_scale=2.0 在 min_issue_scale=3.0 时应被剔除。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)
    _save_bar(lab, "113050.SSE", [108.0], base_date=base)

    snapshot = _make_snapshot([{"code": "113050", "issue_scale": 2.0, "premium_rate": 10.0}])
    premium = _make_premium_hist(["2024-01-02"], [10.0])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={"113050.SSE": premium})

    provider = _build(lab, terms_store, min_issue_scale=3.0, max_price=130.0)
    with pytest.raises(RuntimeError, match="cb_double_low"):
        provider.predict(base, base)


# ---------------------------------------------------------------------------
# 测试 5：上市天数过滤（min_list_days）
# ---------------------------------------------------------------------------


def test_min_list_days_filter_positive(tmp_path: Path) -> None:
    """上市超过 min_list_days 天的转债应通过。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 6, 1)
    _save_bar(lab, "113050.SSE", [108.0], base_date=base)

    # list_date 早于 base 30 天，满足 min_list_days=5
    list_date = (base - timedelta(days=30)).isoformat()
    snapshot = _make_snapshot([{"code": "113050", "premium_rate": 10.0, "list_date": list_date}])
    premium = _make_premium_hist([base.isoformat()], [10.0])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={"113050.SSE": premium})

    provider = _build(lab, terms_store, min_list_days=5, max_price=130.0)
    df = provider.predict(base, base)
    assert df.height == 1


def test_min_list_days_filter_negative(tmp_path: Path) -> None:
    """上市不足 min_list_days 天的新券应被剔除。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 6, 1)
    _save_bar(lab, "113050.SSE", [108.0], base_date=base)

    # list_date = base - 2 天，不满足 min_list_days=5
    list_date = (base - timedelta(days=2)).isoformat()
    snapshot = _make_snapshot([{"code": "113050", "premium_rate": 10.0, "list_date": list_date}])
    premium = _make_premium_hist([base.isoformat()], [10.0])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={"113050.SSE": premium})

    provider = _build(lab, terms_store, min_list_days=5, max_price=130.0)
    with pytest.raises(RuntimeError, match="cb_double_low"):
        provider.predict(base, base)


# ---------------------------------------------------------------------------
# 测试 6：溢价率回退 + warning
# ---------------------------------------------------------------------------


def test_premium_fallback_to_snapshot_warns(tmp_path: Path, caplog) -> None:
    """缺少历史溢价率时应使用快照值回退并发 warning。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)
    _save_bar(lab, "113050.SSE", [108.0], base_date=base)

    # 快照溢价率 15.0（百分比形式），无历史 premium_hist
    snapshot = _make_snapshot([{"code": "113050", "premium_rate": 15.0, "list_date": "2023-01-01"}])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={})  # 无历史数据

    provider = _build(lab, terms_store, max_price=130.0)

    with caplog.at_level(logging.WARNING, logger="aitrade.rules.cb_double_low"):
        df = provider.predict(base, base)

    # 应有 warning
    assert any("回退" in r.message or "快照" in r.message for r in caplog.records)

    # 信号值应基于快照溢价率计算
    assert df.height == 1
    # 快照值 15.0（百分比形式）→ /100 = 0.15 → signal = -(108.0 + 0.15*100) = -123.0
    expected_signal = -(108.0 + 15.0)
    assert float(df["signal"][0]) == pytest.approx(expected_signal, rel=1e-4)


def test_premium_fallback_low_snapshot_no_heuristic(tmp_path: Path, caplog) -> None:
    """快照溢价率 < 10（低溢价双低优质标的）时应与高溢价路径一致，均无条件 /100。

    修复前：abs(3.5) < 10 → 直接返回 3.5（当作小数），signal = -(108.0 + 3.5*100) = -458.0
    修复后：3.5 / 100.0 = 0.035，signal = -(108.0 + 0.035*100) = -(108.0 + 3.5) = -111.5
    """
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)
    close = 108.0
    _save_bar(lab, "113050.SSE", [close], base_date=base)

    # 快照溢价率 3.5（百分比形式，< 10），无历史数据
    snapshot = _make_snapshot([{"code": "113050", "premium_rate": 3.5, "list_date": "2023-01-01"}])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={})

    provider = _build(lab, terms_store, max_price=130.0)

    with caplog.at_level(logging.WARNING, logger="aitrade.rules.cb_double_low"):
        df = provider.predict(base, base)

    assert any("回退" in r.message or "快照" in r.message for r in caplog.records)
    assert df.height == 1

    # 手算：3.5（百分比）/100 = 0.035，signal = -(108.0 + 0.035*100) = -111.5
    expected_signal = -(close + 3.5)
    assert float(df["signal"][0]) == pytest.approx(expected_signal, rel=1e-4)


# ---------------------------------------------------------------------------
# 测试 7：全空时抛 RuntimeError（中文，含提示）
# ---------------------------------------------------------------------------


def test_all_empty_raises_runtime_error(tmp_path: Path) -> None:
    """无任何输出时应抛 RuntimeError，消息含 cb_double_low 和刷新提示。"""
    lab = AlphaLab(tmp_path)
    # 不保存任何行情数据
    snapshot = _make_snapshot([{"code": "113050", "premium_rate": 10.0}])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={})

    provider = _build(lab, terms_store, max_price=130.0)
    with pytest.raises(RuntimeError, match="cb_double_low"):
        provider.predict(date(2024, 1, 2), date(2024, 1, 5))


def test_no_snapshot_raises_runtime_error(tmp_path: Path) -> None:
    """快照不存在时应抛 RuntimeError，提示先刷新。"""
    lab = AlphaLab(tmp_path)
    terms_store = _FakeTermsStore(snapshot=None)

    provider = _build(lab, terms_store, max_price=130.0)
    with pytest.raises(RuntimeError, match="刷新"):
        provider.predict(date(2024, 1, 2), date(2024, 1, 5))


# ---------------------------------------------------------------------------
# 测试 8：评级序比较边界（AA- 含等于，A+ 被拒）
# ---------------------------------------------------------------------------


def test_rating_boundary_aa_minus_passes(tmp_path: Path) -> None:
    """评级恰好等于 AA-（min_rating=AA-）时应通过（>=）。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)
    _save_bar(lab, "113050.SSE", [108.0], base_date=base)

    snapshot = _make_snapshot([{"code": "113050", "rating": "AA-", "premium_rate": 10.0}])
    premium = _make_premium_hist(["2024-01-02"], [10.0])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={"113050.SSE": premium})

    provider = _build(lab, terms_store, min_rating="AA-", max_price=130.0)
    df = provider.predict(base, base)
    assert df.height == 1, "AA- 等于 min_rating=AA- 应通过"


def test_rating_boundary_a_plus_rejected(tmp_path: Path) -> None:
    """评级 A+（低于 AA-）在 min_rating=AA- 时应被拒绝。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)
    _save_bar(lab, "113050.SSE", [108.0], base_date=base)

    snapshot = _make_snapshot([{"code": "113050", "rating": "A+", "premium_rate": 10.0}])
    premium = _make_premium_hist(["2024-01-02"], [10.0])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={"113050.SSE": premium})

    provider = _build(lab, terms_store, min_rating="AA-", max_price=130.0)
    with pytest.raises(RuntimeError, match="cb_double_low"):
        provider.predict(base, base)


# ---------------------------------------------------------------------------
# 测试 9：输出 schema 三列 + 排序
# ---------------------------------------------------------------------------


def test_output_schema_and_sort(tmp_path: Path) -> None:
    """输出 DataFrame 应有三列，按 (datetime, vt_symbol) 升序。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)

    # 两只转债，各两天
    _save_bar(lab, "113050.SSE", [108.0, 109.0], base_date=base)
    _save_bar(lab, "128093.SZSE", [100.0, 101.0], base_date=base)

    snapshot = _make_snapshot(
        [
            {"code": "113050", "premium_rate": 12.0, "list_date": "2023-01-01"},
            {"code": "128093", "premium_rate": 8.0, "list_date": "2023-01-01"},
        ]
    )
    premium_a = _make_premium_hist(
        ["2024-01-02", "2024-01-03"], [12.0, 12.5]
    )
    premium_b = _make_premium_hist(
        ["2024-01-02", "2024-01-03"], [8.0, 7.5]
    )
    terms_store = _FakeTermsStore(
        snapshot=snapshot,
        premium_map={"113050.SSE": premium_a, "128093.SZSE": premium_b},
    )

    provider = _build(lab, terms_store, max_price=130.0)
    end = base + timedelta(days=1)
    df = provider.predict(base, end)

    # schema
    assert list(df.columns) == ["datetime", "vt_symbol", "signal"]

    # 排序：(datetime, vt_symbol) 升序
    if df.height > 1:
        for i in range(df.height - 1):
            dt_i = df["datetime"][i]
            dt_j = df["datetime"][i + 1]
            sym_i = df["vt_symbol"][i]
            sym_j = df["vt_symbol"][i + 1]
            assert (dt_i, sym_i) <= (dt_j, sym_j)


# ---------------------------------------------------------------------------
# 测试 10：全链路 build_signal_source
# ---------------------------------------------------------------------------


def test_build_signal_source_full_chain(tmp_path: Path) -> None:
    """经 build_signal_source('cb_double_low', {...}) 构造的对象可正常 predict。"""
    lab = AlphaLab(tmp_path)
    base = date(2024, 1, 2)
    _save_bar(lab, "113050.SSE", [108.0], base_date=base)

    snapshot = _make_snapshot([{"code": "113050", "premium_rate": 10.0, "list_date": "2023-01-01"}])
    premium = _make_premium_hist(["2024-01-02"], [10.0])
    terms_store = _FakeTermsStore(snapshot=snapshot, premium_map={"113050.SSE": premium})

    provider = build_signal_source(
        "cb_double_low",
        {"_lab": lab, "_terms_store": terms_store, "max_price": 130.0, "min_rating": "A"},
    )
    df = provider.predict(base, base)

    assert df.height >= 1
    assert set(df.columns) >= {"datetime", "vt_symbol", "signal"}


# ---------------------------------------------------------------------------
# 测试 11：cb_double_low 注册后出现在 list_signal_sources
# ---------------------------------------------------------------------------


def test_cb_double_low_registered() -> None:
    """import aitrade.rules 后 'cb_double_low' 应在注册表并有描述与 param_spec。"""
    import aitrade.backtest.registry as _reg  # noqa: PLC0415

    sources = {s["name"]: s for s in _reg.list_signal_sources()}
    assert "cb_double_low" in sources

    meta = sources["cb_double_low"]
    assert meta["description"]
    assert isinstance(meta["param_spec"], dict)
    for key in ("top_k", "max_price", "min_rating", "min_issue_scale", "min_list_days", "interval"):
        assert key in meta["param_spec"], f"param_spec 缺少键：{key}"


# ---------------------------------------------------------------------------
# 测试 12：不引入 torch
# ---------------------------------------------------------------------------


def test_cb_double_low_no_torch_import() -> None:
    """cb_double_low 不应在模块级引入 torch。"""
    import sys

    mod = sys.modules.get("aitrade.rules.cb_double_low")
    assert mod is not None, "aitrade.rules.cb_double_low 应已被 import"
    assert "torch" not in vars(mod), "cb_double_low 不应在模块级引入 torch"
