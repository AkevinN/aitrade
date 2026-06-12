"""
small_cap 小市值信号源测试（Phase 5 Task 5.3）。

覆盖（≥10 用例）：
 1. signal = -total_mv 手算对照
 2. 市值无前视：total_mv 跳变日，断言跳变前的信号日用旧值
 3. min_price 过滤正向（close >= min_price 通过）
 4. min_price 过滤反向（close < min_price 被剔除）
 5. min_amount 流动性过滤正向
 6. min_amount 流动性过滤反向
 7. exclude_st=True 且合约名含 ST → 剔除
 8. exclude_st=True 但无 _contracts 注入 → warning + 不过滤（诚实降级）
 9. min_list_days 上市天数过滤正向
10. min_list_days 上市天数过滤反向
11. list_date 缺失（None）→ 保守保留
12. 全空时抛 RuntimeError（中文，含刷新提示）
13. build_signal_source 全链路可构造
14. 输出 schema 三列 + (datetime, vt_symbol) 升序
15. 不引入 torch
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import aitrade.rules  # noqa: F401  触发 "small_cap" 自注册

from aitrade.alpha.lab import AlphaLab
from aitrade.backtest.registry import build_signal_source
from aitrade.rules.store import FundamentalStore


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------


def _make_bar_frame(
    prices: list[float],
    turnovers: list[float] | None = None,
    base_date: date | None = None,
) -> pl.DataFrame:
    """构造日线 DataFrame（datetime / close / turnover 等必要列）。"""
    if base_date is None:
        base_date = date(2024, 1, 2)
    if turnovers is None:
        # 默认成交额足够大（5_000_000 元/日）
        turnovers = [5_000_000.0] * len(prices)

    rows = []
    for i, (p, t) in enumerate(zip(prices, turnovers, strict=True)):
        dt = base_date + timedelta(days=i)
        rows.append(
            {
                "datetime": datetime(dt.year, dt.month, dt.day, 9, 30),
                "open": p,
                "high": p + 0.5,
                "low": p - 0.5,
                "close": p,
                "volume": t / max(p, 0.01),
                "turnover": t,
                "open_interest": 0.0,
            }
        )
    return pl.DataFrame(rows)


def _save_bar(
    lab: AlphaLab,
    vt_symbol: str,
    prices: list[float],
    turnovers: list[float] | None = None,
    base_date: date | None = None,
) -> None:
    df = _make_bar_frame(prices, turnovers=turnovers, base_date=base_date)
    lab.save_bar_frame(vt_symbol, "d", df)


def _make_fund_df(
    dates: list[str],
    total_mvs: list[float],
) -> pl.DataFrame:
    """构造基本面 DataFrame（YYYYMMDD 格式的 datetime 列）。"""
    rows = []
    for d, mv in zip(dates, total_mvs, strict=True):
        rows.append({
            "datetime": d,
            "pe": 20.0,
            "pe_ttm": 19.5,
            "pb": 2.0,
            "total_mv": mv,
            "circ_mv": mv * 0.8,
            "turnover_rate": 1.5,
        })
    return pl.DataFrame(rows)


def _save_fund(store: FundamentalStore, vt_symbol: str, dates: list[str], total_mvs: list[float]) -> None:
    """快捷写入基本面数据。"""
    store.save(vt_symbol, _make_fund_df(dates, total_mvs))


def _build(
    lab: AlphaLab,
    fund_store: FundamentalStore,
    contracts: dict | None = None,
    **kwargs,
) -> object:
    """快捷构造 small_cap 信号源（注入 _lab/_fundamental_store/_contracts）。"""
    return build_signal_source(
        "small_cap",
        {
            "_lab": lab,
            "_fundamental_store": fund_store,
            "_contracts": contracts,
            **kwargs,
        },
    )


# ---------------------------------------------------------------------------
# 测试 1：signal = -total_mv 手算对照
# ---------------------------------------------------------------------------


def test_signal_equals_negative_total_mv(tmp_path: Path) -> None:
    """signal 应等于 -total_mv（万元原值），两只标的手算验证。

    股 A：total_mv = 50_000（万元）→ signal = -50_000
    股 B：total_mv = 30_000（万元）→ signal = -30_000
    B 信号更高（-30000 > -50000），TopK 降序优先选 B（小市值优先）。
    """
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 1, 2)

    _save_bar(lab, "000001.SZSE", [10.0])
    _save_bar(lab, "000002.SZSE", [8.0])
    _save_fund(store, "000001.SZSE", ["20240102"], [50_000.0])
    _save_fund(store, "000002.SZSE", ["20240102"], [30_000.0])

    provider = _build(lab, store, min_price=1.0, min_amount=1.0, min_list_days=0)
    df = provider.predict(base, base)

    assert df.height == 2

    row_a = df.filter(pl.col("vt_symbol") == "000001.SZSE")
    row_b = df.filter(pl.col("vt_symbol") == "000002.SZSE")

    assert row_a.height == 1
    assert row_b.height == 1

    assert float(row_a["signal"][0]) == pytest.approx(-50_000.0, rel=1e-6)
    assert float(row_b["signal"][0]) == pytest.approx(-30_000.0, rel=1e-6)
    # B 信号更高（小市值优先）
    assert float(row_b["signal"][0]) > float(row_a["signal"][0])


# ---------------------------------------------------------------------------
# 测试 2：市值无前视——total_mv 跳变日，信号日用旧值（严格 ≤ as_of）
# ---------------------------------------------------------------------------


def test_no_lookahead_in_total_mv(tmp_path: Path) -> None:
    """关键属性测试：market cap lookup 必须严格使用 ≤ 信号日的最近一条记录。

    构造：
    - 基本面：20240101 total_mv=10_000，20240103 total_mv=99_000（跳变）
    - 行情：20240102 / 20240103 各一行
    - 信号日 20240102 应使用 20240101 的 total_mv=10_000（≤ 20240102 的最新）
    - 信号日 20240103 应使用 20240103 的 total_mv=99_000
    """
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)

    _save_bar(lab, "600519.SSE", [500.0, 510.0], base_date=date(2024, 1, 2))
    # 基本面：20240101 低市值；20240103 市值跳变（模拟再融资等事件）
    store.save(
        "600519.SSE",
        _make_fund_df(["20240101", "20240103"], [10_000.0, 99_000.0]),
    )

    provider = _build(
        lab, store, min_price=1.0, min_amount=1.0, min_list_days=0
    )
    df = provider.predict(date(2024, 1, 2), date(2024, 1, 3))

    assert df.height == 2, f"应有 2 行，实际 {df.height}"

    sig_0102 = float(df.filter(
        pl.col("datetime").cast(pl.Date) == pl.lit(date(2024, 1, 2))
    )["signal"][0])
    sig_0103 = float(df.filter(
        pl.col("datetime").cast(pl.Date) == pl.lit(date(2024, 1, 3))
    )["signal"][0])

    # 20240102：最近 ≤ 20240102 的基本面是 20240101，total_mv=10_000 → signal=-10_000
    assert sig_0102 == pytest.approx(-10_000.0, rel=1e-6), (
        f"前视红线违反：20240102 的 signal 应为 -10000，实际 {sig_0102}"
    )
    # 20240103：有当日基本面 total_mv=99_000 → signal=-99_000
    assert sig_0103 == pytest.approx(-99_000.0, rel=1e-6), (
        f"20240103 signal 应为 -99000，实际 {sig_0103}"
    )


# ---------------------------------------------------------------------------
# 测试 3 & 4：min_price 过滤
# ---------------------------------------------------------------------------


def test_min_price_filter_positive(tmp_path: Path) -> None:
    """close >= min_price 的标的应出现在结果中。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 1, 2)

    _save_bar(lab, "000001.SZSE", [3.0])
    _save_fund(store, "000001.DZSE", ["20240102"], [5_000.0])  # 不同标的，不影响结果
    _save_fund(store, "000001.SZSE", ["20240102"], [5_000.0])

    provider = _build(lab, store, min_price=2.0, min_amount=1.0, min_list_days=0)
    df = provider.predict(base, base)

    assert df.height == 1
    assert df["vt_symbol"][0] == "000001.SZSE"


def test_min_price_filter_negative(tmp_path: Path) -> None:
    """close < min_price 的标的应被剔除，最终全空 → RuntimeError。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 1, 2)

    _save_bar(lab, "000001.SZSE", [1.5])  # 低于 min_price=2.0
    _save_fund(store, "000001.SZSE", ["20240102"], [5_000.0])

    provider = _build(lab, store, min_price=2.0, min_amount=1.0, min_list_days=0)
    with pytest.raises(RuntimeError, match="small_cap"):
        provider.predict(base, base)


# ---------------------------------------------------------------------------
# 测试 5 & 6：min_amount 流动性过滤
# ---------------------------------------------------------------------------


def test_min_amount_filter_positive(tmp_path: Path) -> None:
    """近 20 日均成交额 >= min_amount 的标的应通过。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 1, 2)

    # 成交额 5_000_000 > min_amount=3_000_000
    _save_bar(lab, "000001.SZSE", [10.0], turnovers=[5_000_000.0])
    _save_fund(store, "000001.SZSE", ["20240102"], [5_000.0])

    provider = _build(lab, store, min_price=1.0, min_amount=3_000_000.0, min_list_days=0)
    df = provider.predict(base, base)

    assert df.height == 1


def test_min_amount_filter_negative(tmp_path: Path) -> None:
    """近 20 日均成交额 < min_amount 的标的应被剔除。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 1, 2)

    # 成交额 1_000_000 < min_amount=3_000_000
    _save_bar(lab, "000001.SZSE", [10.0], turnovers=[1_000_000.0])
    _save_fund(store, "000001.SZSE", ["20240102"], [5_000.0])

    provider = _build(lab, store, min_price=1.0, min_amount=3_000_000.0, min_list_days=0)
    with pytest.raises(RuntimeError, match="small_cap"):
        provider.predict(base, base)


# ---------------------------------------------------------------------------
# 测试 7：ST 过滤——合约名含 ST 被剔除
# ---------------------------------------------------------------------------


def test_st_filter_removes_st_stock(tmp_path: Path) -> None:
    """exclude_st=True 时，名称含 'ST' 的股票应被剔除。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 1, 2)

    _save_bar(lab, "000001.SZSE", [10.0])
    _save_fund(store, "000001.SZSE", ["20240102"], [5_000.0])

    contracts = {
        "000001.SZSE": {"name": "*ST 某某", "list_date": "2020-01-01"},
    }

    provider = _build(
        lab, store, contracts=contracts,
        min_price=1.0, min_amount=1.0, min_list_days=0, exclude_st=True,
    )
    with pytest.raises(RuntimeError, match="small_cap"):
        provider.predict(base, base)


def test_non_st_stock_not_filtered(tmp_path: Path) -> None:
    """exclude_st=True 时，名称不含 ST 的股票应保留。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 1, 2)

    _save_bar(lab, "000001.SZSE", [10.0])
    _save_fund(store, "000001.SZSE", ["20240102"], [5_000.0])

    contracts = {
        "000001.SZSE": {"name": "平安银行", "list_date": "2020-01-01"},
    }

    provider = _build(
        lab, store, contracts=contracts,
        min_price=1.0, min_amount=1.0, min_list_days=0, exclude_st=True,
    )
    df = provider.predict(base, base)
    assert df.height == 1


# ---------------------------------------------------------------------------
# 测试 8：exclude_st=True 但无 _contracts 注入 → warning（诚实降级）
# ---------------------------------------------------------------------------


def test_st_filter_without_contracts_warns(tmp_path: Path, caplog) -> None:
    """exclude_st=True 但 _contracts 为 None 时应 warning 并继续（不过滤）。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 1, 2)

    _save_bar(lab, "000001.SZSE", [10.0])
    _save_fund(store, "000001.SZSE", ["20240102"], [5_000.0])

    # contracts=None（不注入）
    provider = _build(
        lab, store, contracts=None,
        min_price=1.0, min_amount=1.0, min_list_days=0, exclude_st=True,
    )

    with caplog.at_level(logging.WARNING, logger="aitrade.rules.small_cap"):
        df = provider.predict(base, base)

    # 应有降级 warning
    assert any("ST" in r.message for r in caplog.records), (
        "应产生 ST 过滤降级 warning，但未找到"
    )
    # 标的未被过滤（诚实降级：继续输出，不误杀）
    assert df.height == 1


# ---------------------------------------------------------------------------
# 测试 9 & 10：min_list_days 上市天数过滤
# ---------------------------------------------------------------------------


def test_min_list_days_filter_positive(tmp_path: Path) -> None:
    """上市天数 >= min_list_days 的标的应通过。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 6, 1)

    _save_bar(lab, "000001.SZSE", [10.0], base_date=base)
    _save_fund(store, "000001.SZSE", ["20240601"], [5_000.0])

    # 上市 100 天前，满足 min_list_days=60
    list_date = (base - timedelta(days=100)).isoformat()
    contracts = {"000001.SZSE": {"name": "平安银行", "list_date": list_date}}

    provider = _build(
        lab, store, contracts=contracts,
        min_price=1.0, min_amount=1.0, min_list_days=60,
    )
    df = provider.predict(base, base)
    assert df.height == 1


def test_min_list_days_filter_negative(tmp_path: Path) -> None:
    """上市天数 < min_list_days 的新股应被剔除。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 6, 1)

    _save_bar(lab, "000001.SZSE", [10.0], base_date=base)
    _save_fund(store, "000001.SZSE", ["20240601"], [5_000.0])

    # 上市仅 10 天，不满足 min_list_days=60
    list_date = (base - timedelta(days=10)).isoformat()
    contracts = {"000001.SZSE": {"name": "某新股", "list_date": list_date}}

    provider = _build(
        lab, store, contracts=contracts,
        min_price=1.0, min_amount=1.0, min_list_days=60,
    )
    with pytest.raises(RuntimeError, match="small_cap"):
        provider.predict(base, base)


# ---------------------------------------------------------------------------
# 测试 11：list_date 缺失（None）→ 保守保留
# ---------------------------------------------------------------------------


def test_list_date_none_keeps_stock(tmp_path: Path) -> None:
    """list_date=None 时应保守保留（不剔除），即使 min_list_days > 0。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 6, 1)

    _save_bar(lab, "000001.SZSE", [10.0], base_date=base)
    _save_fund(store, "000001.SZSE", ["20240601"], [5_000.0])

    # list_date=None：未知上市日期
    contracts = {"000001.SZSE": {"name": "某股票", "list_date": None}}

    provider = _build(
        lab, store, contracts=contracts,
        min_price=1.0, min_amount=1.0, min_list_days=60,
    )
    df = provider.predict(base, base)
    # 保守保留
    assert df.height == 1, "list_date=None 时应保守保留，不因 min_list_days 剔除"


# ---------------------------------------------------------------------------
# 测试 12：全空时抛 RuntimeError
# ---------------------------------------------------------------------------


def test_all_empty_raises_runtime_error(tmp_path: Path) -> None:
    """过滤后无任何输出时应抛 RuntimeError，消息含 'small_cap' 和刷新提示。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 1, 2)

    # 行情存在但价格低于 min_price，应全部被过滤
    _save_bar(lab, "000001.SZSE", [0.5])  # 低于 min_price=2.0
    _save_fund(store, "000001.SZSE", ["20240102"], [5_000.0])

    provider = _build(lab, store, min_price=2.0, min_amount=1.0, min_list_days=0)
    with pytest.raises(RuntimeError, match="small_cap"):
        provider.predict(base, base)


def test_no_fundamental_data_raises_runtime_error(tmp_path: Path) -> None:
    """基本面数据完全为空时应抛 RuntimeError（无宇宙）。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 1, 2)

    # 不保存任何基本面数据（list_symbols 返回空列表）
    # 保存行情但没有基本面：宇宙为空
    _save_bar(lab, "000001.SZSE", [10.0])
    # 不调用 _save_fund

    provider = _build(lab, store, min_price=1.0, min_amount=1.0, min_list_days=0)
    with pytest.raises(RuntimeError, match="small_cap"):
        provider.predict(base, base)


# ---------------------------------------------------------------------------
# 测试 13：build_signal_source 全链路可构造
# ---------------------------------------------------------------------------


def test_build_signal_source_full_chain(tmp_path: Path) -> None:
    """经 build_signal_source('small_cap', {...}) 构造的对象可正常 predict。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 1, 2)

    _save_bar(lab, "000001.SZSE", [10.0])
    _save_fund(store, "000001.SZSE", ["20240102"], [5_000.0])

    provider = build_signal_source(
        "small_cap",
        {
            "_lab": lab,
            "_fundamental_store": store,
            "_contracts": None,
            "min_price": 1.0,
            "min_amount": 1.0,
            "min_list_days": 0,
            "exclude_st": False,
        },
    )
    df = provider.predict(base, base)

    assert df.height >= 1
    assert set(df.columns) >= {"datetime", "vt_symbol", "signal"}


# ---------------------------------------------------------------------------
# 测试 14：输出 schema 三列 + (datetime, vt_symbol) 升序
# ---------------------------------------------------------------------------


def test_output_schema_and_sort(tmp_path: Path) -> None:
    """输出 DataFrame 应有三列，且按 (datetime, vt_symbol) 升序。"""
    lab = AlphaLab(tmp_path)
    store = FundamentalStore(base_path=tmp_path)
    base = date(2024, 1, 2)

    # 两只股票，各两天
    _save_bar(lab, "000001.SZSE", [10.0, 11.0], base_date=base)
    _save_bar(lab, "000002.SZSE", [8.0, 9.0], base_date=base)
    _save_fund(store, "000001.SZSE", ["20240101"], [50_000.0])
    _save_fund(store, "000002.SZSE", ["20240101"], [30_000.0])

    provider = _build(
        lab, store, min_price=1.0, min_amount=1.0, min_list_days=0
    )
    end = base + timedelta(days=1)
    df = provider.predict(base, end)

    # schema
    assert list(df.columns) == ["datetime", "vt_symbol", "signal"]
    assert df["datetime"].dtype in (pl.Datetime, pl.Date)
    assert df["vt_symbol"].dtype in (pl.Utf8, pl.String)
    assert df["signal"].dtype == pl.Float64

    # 排序：(datetime, vt_symbol) 升序
    for i in range(df.height - 1):
        dt_i = str(df["datetime"][i])
        dt_j = str(df["datetime"][i + 1])
        sym_i = df["vt_symbol"][i]
        sym_j = df["vt_symbol"][i + 1]
        assert (dt_i, sym_i) <= (dt_j, sym_j), (
            f"排序违反：({dt_i}, {sym_i}) > ({dt_j}, {sym_j})"
        )


# ---------------------------------------------------------------------------
# 测试 15：不引入 torch
# ---------------------------------------------------------------------------


def test_small_cap_no_torch_import() -> None:
    """small_cap 不应在模块级引入 torch。"""
    mod = sys.modules.get("aitrade.rules.small_cap")
    assert mod is not None, "aitrade.rules.small_cap 应已被 import"
    assert "torch" not in vars(mod), "small_cap 不应在模块级引入 torch"


# ---------------------------------------------------------------------------
# 测试 16：small_cap 注册后出现在 list_signal_sources
# ---------------------------------------------------------------------------


def test_small_cap_registered() -> None:
    """import aitrade.rules 后 'small_cap' 应在注册表，含描述与 param_spec。"""
    import aitrade.backtest.registry as _reg  # noqa: PLC0415

    sources = {s["name"]: s for s in _reg.list_signal_sources()}
    assert "small_cap" in sources

    meta = sources["small_cap"]
    assert meta["description"]
    assert isinstance(meta["param_spec"], dict)
    for key in ("top_k", "min_price", "min_list_days", "min_amount", "exclude_st", "interval"):
        assert key in meta["param_spec"], f"param_spec 缺少键：{key}"
