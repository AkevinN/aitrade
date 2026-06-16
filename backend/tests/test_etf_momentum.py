"""
etf_momentum 信号源验收测试（Task 2.2）。

覆盖范围
--------
1. 两标的不同动量 → 信号值与手算一致（确定性价格序列）
2. 负动量标的被过滤；全体负动量日整天无行（空仓防御）
3. 预热正确：start 当日即有信号（前提数据足够）
4. 个别标的缺数据 → warning + 其余标的正常输出
5. 全部缺数据 → RuntimeError 中文错误
6. lookback 校验：< 1 抛 ValueError；类型错误抛 ValueError
7. 经 build_signal_source("etf_momentum", {...}) 全链路构造可用
8. 输出 schema 三列（datetime / vt_symbol / signal）与排序
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import polars as pl
import pytest

import aitrade.rules  # noqa: F401  触发 "etf_momentum" 自注册

from aitrade.backtest.registry import build_signal_source
from aitrade.alpha.lab import AlphaLab


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------


def _make_bar_frame(
    prices: list[float],
    base_date: date | None = None,
) -> pl.DataFrame:
    """构造仅含 datetime / close 的日线 DataFrame，用于写入 AlphaLab。

    datetime 从 base_date（默认 2024-01-02）起每日递增。
    """
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
                "volume": 1_000_000.0,
                "turnover": p * 1_000_000.0,
                "open_interest": 0.0,
            }
        )
    return pl.DataFrame(rows)


def _save(lab: AlphaLab, vt_symbol: str, prices: list[float], base_date: date | None = None) -> None:
    """快捷方式：把价格序列写入 lab（日线）。"""
    df = _make_bar_frame(prices, base_date=base_date)
    lab.save_bar_frame(vt_symbol, "d", df)


def _build(lab: AlphaLab, **kwargs) -> object:
    """快捷方式：构造 etf_momentum 信号源（注入 _lab）。"""
    return build_signal_source("etf_momentum", {"_lab": lab, **kwargs})


# ---------------------------------------------------------------------------
# 测试 1：信号值与手算一致
# ---------------------------------------------------------------------------


def test_signal_values_match_manual_calculation(tmp_path):
    """两标的不同动量，信号值应等于 close/close.shift(lookback)-1 的手算结果。"""
    lab = AlphaLab(tmp_path)
    lookback = 3

    # A：第 0~2 日为预热，第 3 日 close=110，close[0]=100 → momentum=0.10
    prices_a = [100.0, 102.0, 105.0, 110.0]
    # B：第 0~2 日为预热，第 3 日 close=95，close[0]=100 → momentum=-0.05
    prices_b = [100.0, 101.0, 98.0, 95.0]

    base = date(2024, 1, 2)
    _save(lab, "A.SSE", prices_a, base_date=base)
    _save(lab, "B.SSE", prices_b, base_date=base)

    # start=第3日，end=第3日，min_momentum=-1.0（不过滤任何）
    start = base + timedelta(days=3)
    end = start

    provider = _build(
        lab,
        universe=["A.SSE", "B.SSE"],
        lookback=lookback,
        min_momentum=-1.0,
    )
    df = provider.predict(start, end)

    # 应有两行
    assert df.height == 2

    row_a = df.filter(pl.col("vt_symbol") == "A.SSE")
    row_b = df.filter(pl.col("vt_symbol") == "B.SSE")

    assert row_a.height == 1
    assert row_b.height == 1

    assert float(row_a["signal"][0]) == pytest.approx(0.10, rel=1e-6)
    assert float(row_b["signal"][0]) == pytest.approx(-0.05, rel=1e-6)


# ---------------------------------------------------------------------------
# 测试 2：负动量过滤 + 全体负动量日无行
# ---------------------------------------------------------------------------


def test_negative_momentum_filtered_default(tmp_path):
    """默认 min_momentum=0.0，负动量标的被过滤；若所有标的该日均负，整天无行。"""
    lab = AlphaLab(tmp_path)
    lookback = 2

    # A：最后一日收于起点以下 → momentum < 0
    prices_a = [100.0, 99.0, 98.0]
    # B：最后一日收于起点以下 → momentum < 0
    prices_b = [100.0, 99.5, 97.0]

    base = date(2024, 1, 2)
    _save(lab, "A.SSE", prices_a, base_date=base)
    _save(lab, "B.SSE", prices_b, base_date=base)

    start = base + timedelta(days=lookback)  # 最后一日
    end = start

    provider = _build(lab, universe=["A.SSE", "B.SSE"], lookback=lookback, min_momentum=0.0)
    df = provider.predict(start, end)

    # 全体负动量 → 整天无行
    assert df.is_empty(), f"期望空 DataFrame，实际得到 {df}"


def test_partial_negative_momentum_filtered(tmp_path):
    """A 正动量、B 负动量时，仅 A 的行出现在结果中。"""
    lab = AlphaLab(tmp_path)
    lookback = 2

    prices_a = [100.0, 102.0, 108.0]   # momentum = 0.08 > 0
    prices_b = [100.0, 101.0, 96.0]    # momentum = -0.04 < 0

    base = date(2024, 1, 2)
    _save(lab, "A.SSE", prices_a, base_date=base)
    _save(lab, "B.SSE", prices_b, base_date=base)

    start = base + timedelta(days=lookback)
    end = start

    provider = _build(lab, universe=["A.SSE", "B.SSE"], lookback=lookback, min_momentum=0.0)
    df = provider.predict(start, end)

    assert df.height == 1
    assert df["vt_symbol"][0] == "A.SSE"
    assert float(df["signal"][0]) == pytest.approx(0.08, rel=1e-6)


# ---------------------------------------------------------------------------
# 测试 3：预热正确 —— start 当日即有信号
# ---------------------------------------------------------------------------


def test_warmup_start_day_has_signal(tmp_path):
    """数据量充足时，predict(start=第lookback日, end=...) 在 start 当日应有信号。"""
    lab = AlphaLab(tmp_path)
    lookback = 5

    # 构造 lookback+2 根 bar（预热足够）
    # 价格线性上涨，最终动量必然为正
    n_bars = lookback + 2
    prices = [100.0 + i for i in range(n_bars)]

    base = date(2024, 1, 2)
    _save(lab, "ONLY.SSE", prices, base_date=base)

    # start 设为能产生动量的最早日（base + lookback）
    start = base + timedelta(days=lookback)
    end = base + timedelta(days=n_bars - 1)

    provider = _build(lab, universe=["ONLY.SSE"], lookback=lookback, min_momentum=-1.0)
    df = provider.predict(start, end)

    # start 日 (datetime 等于 start 的 9:30) 必须出现在结果中
    start_dt = datetime(start.year, start.month, start.day, 9, 30)
    rows_at_start = df.filter(pl.col("datetime") == start_dt)

    assert rows_at_start.height >= 1, (
        f"start 当日 {start} 应有信号，但结果中仅含：{df['datetime'].to_list()}"
    )


# ---------------------------------------------------------------------------
# 测试 4：个别标的缺数据 → warning + 其余正常
# ---------------------------------------------------------------------------


def test_missing_single_symbol_warns_and_others_ok(tmp_path, caplog):
    """一个标的无数据时，应 warning 并继续输出其余标的的信号。"""
    lab = AlphaLab(tmp_path)
    lookback = 2

    prices_good = [100.0, 104.0, 110.0]  # momentum > 0
    base = date(2024, 1, 2)
    _save(lab, "GOOD.SSE", prices_good, base_date=base)
    # "MISSING.SSE" 故意不写入任何数据

    start = base + timedelta(days=lookback)
    end = start

    with caplog.at_level(logging.WARNING, logger="aitrade.rules.etf_momentum"):
        provider = _build(
            lab,
            universe=["GOOD.SSE", "MISSING.SSE"],
            lookback=lookback,
            min_momentum=-1.0,
        )
        df = provider.predict(start, end)

    # 应有 MISSING.SSE 相关的 warning
    assert any("MISSING.SSE" in r.message for r in caplog.records)

    # GOOD.SSE 的信号应正常输出
    assert df.height == 1
    assert df["vt_symbol"][0] == "GOOD.SSE"


# ---------------------------------------------------------------------------
# 测试 5：全部缺数据 → RuntimeError 中文
# ---------------------------------------------------------------------------


def test_all_missing_raises_runtime_error(tmp_path):
    """universe 全部无本地行情时，predict 应抛 RuntimeError，信息为中文。"""
    lab = AlphaLab(tmp_path)

    provider = _build(
        lab,
        universe=["NO_DATA_A.SSE", "NO_DATA_B.SSE"],
        lookback=5,
        min_momentum=-1.0,
    )

    with pytest.raises(RuntimeError, match="universe 中所有标的均无本地行情"):
        provider.predict(date(2024, 1, 10), date(2024, 1, 20))


# ---------------------------------------------------------------------------
# 测试 6：lookback 校验
# ---------------------------------------------------------------------------


def test_lookback_zero_raises_value_error(tmp_path):
    """lookback=0 应在构造期抛 ValueError。"""
    lab = AlphaLab(tmp_path)
    with pytest.raises(ValueError, match="lookback"):
        _build(lab, lookback=0)


def test_lookback_negative_raises_value_error(tmp_path):
    """lookback=-5 应在构造期抛 ValueError。"""
    lab = AlphaLab(tmp_path)
    with pytest.raises(ValueError, match="lookback"):
        _build(lab, lookback=-5)


def test_lookback_wrong_type_raises_value_error(tmp_path):
    """lookback 传浮点数（如 5.0）应在构造期抛 ValueError（严格类型）。"""
    lab = AlphaLab(tmp_path)
    with pytest.raises(ValueError, match="lookback"):
        _build(lab, lookback=5.0)


def test_universe_wrong_type_raises_value_error(tmp_path):
    """universe 不是列表时应抛 ValueError。"""
    lab = AlphaLab(tmp_path)
    with pytest.raises(ValueError, match="universe"):
        _build(lab, universe="510300.SSE")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 测试 7：build_signal_source 全链路
# ---------------------------------------------------------------------------


def test_build_signal_source_full_chain(tmp_path):
    """经 build_signal_source('etf_momentum', {...}) 构造出的对象是 SignalProvider。"""
    lab = AlphaLab(tmp_path)
    lookback = 2
    prices = [100.0, 105.0, 112.0]
    base = date(2024, 1, 2)
    _save(lab, "ETF.SSE", prices, base_date=base)

    # 不依赖内部符号，走注册表公开 API
    provider = build_signal_source(
        "etf_momentum",
        {
            "_lab": lab,
            "universe": ["ETF.SSE"],
            "lookback": lookback,
            "min_momentum": -1.0,
        },
    )

    start = base + timedelta(days=lookback)
    end = start
    df = provider.predict(start, end)

    assert df.height >= 1
    assert set(df.columns) >= {"datetime", "vt_symbol", "signal"}


# ---------------------------------------------------------------------------
# 测试 8：输出 schema + 排序
# ---------------------------------------------------------------------------


def test_output_schema_and_sort_order(tmp_path):
    """输出 DataFrame 应恰好有三列，且按 (datetime, vt_symbol) 升序排列。"""
    lab = AlphaLab(tmp_path)
    lookback = 2

    # 两标的各两天，共 4 行（2 天 × 2 标的）
    base = date(2024, 1, 2)
    _save(lab, "AAA.SSE", [100.0, 102.0, 110.0, 115.0], base_date=base)
    _save(lab, "BBB.SSE", [100.0, 103.0, 112.0, 118.0], base_date=base)

    start = base + timedelta(days=lookback)       # 第3日
    end = base + timedelta(days=lookback + 1)     # 第4日

    provider = _build(
        lab,
        universe=["AAA.SSE", "BBB.SSE"],
        lookback=lookback,
        min_momentum=-1.0,
    )
    df = provider.predict(start, end)

    # schema：恰好三列
    assert list(df.columns) == ["datetime", "vt_symbol", "signal"]

    # 排序：(datetime, vt_symbol) 升序
    if df.height > 1:
        for i in range(df.height - 1):
            dt_i = df["datetime"][i]
            dt_j = df["datetime"][i + 1]
            sym_i = df["vt_symbol"][i]
            sym_j = df["vt_symbol"][i + 1]
            assert (dt_i, sym_i) <= (dt_j, sym_j), (
                f"排序异常：第{i}行 ({dt_i},{sym_i}) > 第{i+1}行 ({dt_j},{sym_j})"
            )


# ---------------------------------------------------------------------------
# 测试 9：注册后 list_signal_sources 中包含 etf_momentum 条目
# ---------------------------------------------------------------------------


def test_etf_momentum_registered():
    """import aitrade.rules 后 'etf_momentum' 应出现在信号源注册表并有描述与 param_spec。"""
    import aitrade.backtest.registry as _reg  # noqa: PLC0415

    sources = {s["name"]: s for s in _reg.list_signal_sources()}
    assert "etf_momentum" in sources

    meta = sources["etf_momentum"]
    assert meta["description"]  # 非空
    assert isinstance(meta["param_spec"], dict)
    # 检查 param_spec 的关键键存在
    for key in ("universe", "lookback", "min_momentum", "interval"):
        assert key in meta["param_spec"], f"param_spec 缺少键：{key}"


# ---------------------------------------------------------------------------
# 测试 10：import rules 不应新引入 torch
# ---------------------------------------------------------------------------


def test_etf_momentum_no_torch_import():
    """etf_momentum 模块不应在模块级引入 torch（规则信号源红线）。

    策略：直接检查已加载的 aitrade.rules.etf_momentum 模块的全局命名空间，
    确认 "torch" 不存在其中。这比移除 sys.modules 再重 import 更安全，
    不会造成策略注册表的跨测试污染（registry 内类 id 错位）。
    """
    import sys

    # etf_momentum 模块此时应已被 `import aitrade.rules` 加载（文件顶部已 import）
    mod = sys.modules.get("aitrade.rules.etf_momentum")
    assert mod is not None, "aitrade.rules.etf_momentum 应已被 import"

    # 模块全局命名空间不应含 torch
    assert "torch" not in vars(mod), (
        "etf_momentum 不应在模块级引入 torch；请检查模块顶层 import"
    )
