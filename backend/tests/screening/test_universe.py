"""
CNN 选股 universe.discover_universe 测试。

全部测试使用伪造的 FakeLab（只实现 list_data_resources()，返回合成数据），
**不依赖真实行情文件**，可在 CI / 离线环境零配置运行。

覆盖场景：
- 默认发现（interval 过滤，按字典序排序）
- 交易所过滤（只保留匹配，其余入 excluded）
- 最小历史 bar 数过滤（不足的入 excluded）
- include_symbols 显式清单 + 本地无数据 → excluded "无本地数据"
- exclude_symbols 显式排除
- interval 不匹配被排除
- 空结果不抛异常（Property 10）
- 代码规范化：sz000001 / 纯六位数字 → 标准 vt_symbol

Feature: cnn-stock-screening, Property 10: 批量鲁棒性（空池结构化上报）
"""

from __future__ import annotations

from typing import Any

import pytest

from aitrade.screening.universe import discover_universe


# ---------------------------------------------------------------------------
# 测试辅助：FakeLab
# ---------------------------------------------------------------------------

class FakeLab:
    """最小化伪 AlphaLab，只暴露 list_data_resources() 返回合成数据。

    Args:
        raw_bars: 注入到 ``raw_bars`` 列表的摘要条目。
        derived_bars: 注入到 ``derived_bars`` 列表的摘要条目，默认空列表。
    """

    def __init__(
        self,
        raw_bars: list[dict[str, Any]],
        derived_bars: list[dict[str, Any]] | None = None,
    ) -> None:
        self._raw_bars = raw_bars
        self._derived_bars = derived_bars or []

    def list_data_resources(self) -> dict[str, Any]:
        """返回合成数据资源摘要，格式与真实 AlphaLab 完全一致。

        Returns:
            含 ``raw_bars``、``derived_bars`` 等键的字典。
        """
        return {
            "raw_bars": self._raw_bars,
            "raw_ticks": [],
            "raw_bar_batches": [],
            "raw_tick_batches": [],
            "derived_bars": self._derived_bars,
            "raw_bar_intervals": [],
            "derived_intervals": [],
        }


def _bar(
    vt_symbol: str,
    interval: str,
    row_count: int,
    section: str = "raw_bars",
) -> dict[str, Any]:
    """构造一条最小 bar 摘要条目（辅助工厂）。

    Args:
        vt_symbol: 标的代码，如 ``"000001.SZSE"``。
        interval: K 线周期，如 ``"d"``、``"1m"``。
        row_count: 数据行数。
        section: 所属分组名（仅用于注释，实际放入由 FakeLab 决定）。

    Returns:
        与 ``_resource_summary_from_file`` 返回值结构兼容的字典。
    """
    return {
        "vt_symbol": vt_symbol,
        "interval": interval,
        "row_count": row_count,
        "start": "2023-01-01",
        "end": "2025-12-31",
        "kind": "raw_bar" if section == "raw_bars" else "derived_bar",
    }


# ---------------------------------------------------------------------------
# 测试：默认发现（全量 + 按字典序排序）
# ---------------------------------------------------------------------------

def test_default_discovery_returns_all_symbols_for_interval() -> None:
    """默认模式：返回指定 interval 下本地全部标的，按字典序升序。"""
    lab = FakeLab(
        raw_bars=[
            _bar("600519.SSE", "d", 500),
            _bar("000001.SZSE", "d", 300),
            _bar("300750.SZSE", "1m", 1000),  # 不同 interval，应被排除
            _bar("000002.SZSE", "d", 400),
        ]
    )
    universe, excluded = discover_universe(lab, interval="d", min_bar_count=1)

    assert universe == ["000001.SZSE", "000002.SZSE", "600519.SSE"]
    # 300750.SZSE 的 1m 数据不在 "d" interval 下，不应出现在 excluded
    # （它从未进入候选集，interval 不匹配是在 available_map 构造阶段就过滤掉的）
    assert all(e["vt_symbol"] != "300750.SZSE" for e in excluded)


def test_default_discovery_sorted_deterministically() -> None:
    """universe 列表按字典序确定性排序，与数据输入顺序无关。"""
    lab = FakeLab(
        raw_bars=[
            _bar("600030.SSE", "d", 500),
            _bar("000001.SZSE", "d", 300),
            _bar("000002.SZSE", "d", 300),
            _bar("300760.SZSE", "d", 300),
        ]
    )
    universe, _ = discover_universe(lab, interval="d", min_bar_count=1)

    assert universe == sorted(universe), "universe 应当按字典序升序排列"


# ---------------------------------------------------------------------------
# 测试：interval 不匹配被过滤
# ---------------------------------------------------------------------------

def test_interval_mismatch_excluded_from_universe() -> None:
    """本地只有 1m 数据的标的，请求 d 线时不出现在 universe 中。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 300),
            _bar("600000.SSE", "1m", 50000),  # 只有分钟线，无日线
        ]
    )
    universe, excluded = discover_universe(lab, interval="d", min_bar_count=1)

    assert "600000.SSE" not in universe
    # 600000.SSE 未进入候选集，excluded 中也不应包含它
    assert all(e["vt_symbol"] != "600000.SSE" for e in excluded)
    assert universe == ["000001.SZSE"]


# ---------------------------------------------------------------------------
# 测试：交易所过滤
# ---------------------------------------------------------------------------

def test_exchange_filter_keeps_only_matching() -> None:
    """exchange 过滤：只保留指定交易所，其余标的进入 excluded 并附原因。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 300),
            _bar("000002.SZSE", "d", 400),
            _bar("600519.SSE", "d", 500),
            _bar("430047.BSE", "d", 200),
        ]
    )
    universe, excluded = discover_universe(
        lab, interval="d", min_bar_count=1, exchange="SZSE"
    )

    assert universe == ["000001.SZSE", "000002.SZSE"]
    excluded_symbols = {e["vt_symbol"] for e in excluded}
    assert "600519.SSE" in excluded_symbols
    assert "430047.BSE" in excluded_symbols
    # 原因包含"期望 SZSE"
    for e in excluded:
        if e["vt_symbol"] in ("600519.SSE", "430047.BSE"):
            assert "SZSE" in e["reason"]


def test_exchange_filter_none_does_not_filter() -> None:
    """exchange=None 时不按交易所过滤，保留全部标的。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 300),
            _bar("600519.SSE", "d", 500),
        ]
    )
    universe, excluded = discover_universe(
        lab, interval="d", min_bar_count=1, exchange=None
    )

    assert len(universe) == 2
    assert len(excluded) == 0


# ---------------------------------------------------------------------------
# 测试：min_bar_count 过滤
# ---------------------------------------------------------------------------

def test_min_bar_count_drops_short_history() -> None:
    """历史不足 min_bar_count 的标的进入 excluded，原因含实际 row_count 与阈值。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 300),
            _bar("000002.SZSE", "d", 100),  # 不足 250
            _bar("600519.SSE", "d", 500),
        ]
    )
    universe, excluded = discover_universe(
        lab, interval="d", min_bar_count=250
    )

    assert "000002.SZSE" not in universe
    assert "000001.SZSE" in universe
    assert "600519.SSE" in universe

    short_excluded = [e for e in excluded if e["vt_symbol"] == "000002.SZSE"]
    assert len(short_excluded) == 1
    reason = short_excluded[0]["reason"]
    assert "100" in reason   # 实际 row_count
    assert "250" in reason   # min_bar_count


def test_min_bar_count_exactly_at_threshold_passes() -> None:
    """row_count 恰好等于 min_bar_count（边界值）时应通过过滤，不被排除。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 250),
        ]
    )
    universe, excluded = discover_universe(lab, interval="d", min_bar_count=250)

    assert "000001.SZSE" in universe
    assert len(excluded) == 0


# ---------------------------------------------------------------------------
# 测试：include_symbols 显式清单
# ---------------------------------------------------------------------------

def test_include_symbols_restricts_to_given_list() -> None:
    """include_symbols 提供时，候选集限定为该清单，不在清单内的标的不进入 universe。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 300),
            _bar("000002.SZSE", "d", 400),
            _bar("600519.SSE", "d", 500),
        ]
    )
    universe, excluded = discover_universe(
        lab,
        interval="d",
        min_bar_count=1,
        include_symbols=["000001.SZSE", "600519.SSE"],
    )

    assert universe == ["000001.SZSE", "600519.SSE"]
    assert all(e["vt_symbol"] != "000002.SZSE" for e in excluded)


def test_include_symbol_with_no_local_data_excluded_with_reason() -> None:
    """include_symbols 中的标的在本地无 interval 数据时，进入 excluded，原因为"无本地数据"。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 300),
            # 000002.SZSE 根本不在数据库中
        ]
    )
    universe, excluded = discover_universe(
        lab,
        interval="d",
        min_bar_count=1,
        include_symbols=["000001.SZSE", "000002.SZSE"],
    )

    assert "000002.SZSE" not in universe
    no_data = [e for e in excluded if e["vt_symbol"] == "000002.SZSE"]
    assert len(no_data) == 1
    assert "无本地数据" in no_data[0]["reason"]


# ---------------------------------------------------------------------------
# 测试：exclude_symbols 显式排除
# ---------------------------------------------------------------------------

def test_exclude_symbols_removes_matching() -> None:
    """exclude_symbols 中的标的被排除，原因为"用户显式排除"。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 300),
            _bar("000002.SZSE", "d", 400),
            _bar("600519.SSE", "d", 500),
        ]
    )
    universe, excluded = discover_universe(
        lab,
        interval="d",
        min_bar_count=1,
        exclude_symbols=["000002.SZSE"],
    )

    assert "000002.SZSE" not in universe
    excluded_reasons = {e["vt_symbol"]: e["reason"] for e in excluded}
    assert "000002.SZSE" in excluded_reasons
    assert "用户显式排除" in excluded_reasons["000002.SZSE"]


def test_exclude_symbols_empty_list_no_effect() -> None:
    """exclude_symbols 为空列表时，不排除任何标的。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 300),
            _bar("600519.SSE", "d", 500),
        ]
    )
    universe, excluded = discover_universe(
        lab, interval="d", min_bar_count=1, exclude_symbols=[]
    )

    assert len(universe) == 2
    assert len(excluded) == 0


def test_exclude_symbols_with_include_symbols() -> None:
    """include_symbols + exclude_symbols 同时出现：清单入场后仍受 exclude_symbols 限制。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 300),
            _bar("000002.SZSE", "d", 400),
        ]
    )
    universe, excluded = discover_universe(
        lab,
        interval="d",
        min_bar_count=1,
        include_symbols=["000001.SZSE", "000002.SZSE"],
        exclude_symbols=["000002.SZSE"],
    )

    assert universe == ["000001.SZSE"]
    removed = [e for e in excluded if e["vt_symbol"] == "000002.SZSE"]
    assert len(removed) == 1
    assert "用户显式排除" in removed[0]["reason"]


# ---------------------------------------------------------------------------
# 测试：空结果不抛异常（Property 10 / R1.5）
# ---------------------------------------------------------------------------

def test_empty_result_returns_empty_list_not_exception() -> None:
    """过滤后 universe 为空时，返回 ([], excluded)，不抛异常。

    # Feature: cnn-stock-screening, Property 10: 批量鲁棒性（空池结构化上报）
    """
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 50),   # 历史不足 250
            _bar("000002.SZSE", "d", 80),   # 历史不足 250
        ]
    )
    result = discover_universe(lab, interval="d", min_bar_count=250)

    assert isinstance(result, tuple), "应当返回 tuple"
    universe, excluded = result
    assert universe == [], "universe 应为空列表"
    assert len(excluded) == 2, "两只标的均应在 excluded 中"


def test_empty_local_db_returns_empty() -> None:
    """本地无任何数据时，返回 ([], [])，不抛异常。

    # Feature: cnn-stock-screening, Property 10: 批量鲁棒性（空池结构化上报）
    """
    lab = FakeLab(raw_bars=[])
    universe, excluded = discover_universe(lab, interval="d", min_bar_count=1)

    assert universe == []
    assert excluded == []


def test_no_matching_exchange_returns_empty() -> None:
    """全部标的交易所均不匹配时，universe 为空，不抛异常。

    # Feature: cnn-stock-screening, Property 10: 批量鲁棒性（空池结构化上报）
    """
    lab = FakeLab(
        raw_bars=[
            _bar("600519.SSE", "d", 500),
            _bar("600030.SSE", "d", 400),
        ]
    )
    universe, excluded = discover_universe(
        lab, interval="d", min_bar_count=1, exchange="SZSE"
    )

    assert universe == []
    assert len(excluded) == 2


# ---------------------------------------------------------------------------
# 测试：代码规范化
# ---------------------------------------------------------------------------

def test_include_symbols_raw_prefix_normalizes_and_matches() -> None:
    """include_symbols 中的 ``sz000001`` 规范化为 ``000001.SZSE`` 后匹配本地数据。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 300),
        ]
    )
    universe, excluded = discover_universe(
        lab,
        interval="d",
        min_bar_count=1,
        include_symbols=["sz000001"],
    )

    assert "000001.SZSE" in universe
    assert len(excluded) == 0


def test_include_symbols_plain_digits_normalizes_and_matches() -> None:
    """include_symbols 中的纯六位数字（如 ``000001``）规范化后匹配本地数据。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 300),
        ]
    )
    universe, excluded = discover_universe(
        lab,
        interval="d",
        min_bar_count=1,
        include_symbols=["000001"],
    )

    assert "000001.SZSE" in universe
    assert len(excluded) == 0


def test_include_symbols_sse_prefix_normalizes() -> None:
    """``sh600519`` 规范化为 ``600519.SSE`` 后匹配本地数据。"""
    lab = FakeLab(
        raw_bars=[
            _bar("600519.SSE", "d", 500),
        ]
    )
    universe, excluded = discover_universe(
        lab,
        interval="d",
        min_bar_count=1,
        include_symbols=["sh600519"],
    )

    assert "600519.SSE" in universe
    assert len(excluded) == 0


# ---------------------------------------------------------------------------
# 测试：derived_bars 补充
# ---------------------------------------------------------------------------

def test_derived_bars_counted_when_raw_absent() -> None:
    """当 raw_bars 无该标的数据、但 derived_bars 有时，标的可进入候选集。"""
    lab = FakeLab(
        raw_bars=[],
        derived_bars=[
            _bar("000001.SZSE", "d", 300, section="derived_bars"),
        ],
    )
    universe, excluded = discover_universe(lab, interval="d", min_bar_count=1)

    assert "000001.SZSE" in universe


def test_raw_and_derived_same_symbol_takes_max_row_count() -> None:
    """同一标的在 raw_bars 和 derived_bars 均有数据时，row_count 取较大值（影响 min_bar_count 过滤）。"""
    lab = FakeLab(
        raw_bars=[
            _bar("000001.SZSE", "d", 100),    # 不足 250
        ],
        derived_bars=[
            _bar("000001.SZSE", "d", 300, section="derived_bars"),  # 足够
        ],
    )
    universe, excluded = discover_universe(lab, interval="d", min_bar_count=250)

    # derived_bars 的 300 > 100，应通过过滤
    assert "000001.SZSE" in universe
    assert len(excluded) == 0


# ---------------------------------------------------------------------------
# 测试：只读性（不调用任何写方法）
# ---------------------------------------------------------------------------

def test_read_only_only_calls_list_data_resources() -> None:
    """discover_universe 仅调用 list_data_resources()，不调用其他 lab 方法。"""
    call_log: list[str] = []

    class SpyLab:
        def list_data_resources(self) -> dict[str, Any]:
            call_log.append("list_data_resources")
            return {"raw_bars": [], "derived_bars": []}

        def __getattr__(self, name: str) -> Any:
            def _forbidden(*args: Any, **kwargs: Any) -> None:
                call_log.append(name)
            return _forbidden

    lab = SpyLab()
    discover_universe(lab, interval="d", min_bar_count=1)

    assert call_log == ["list_data_resources"], (
        f"只应调用 list_data_resources，实际调用了：{call_log}"
    )
