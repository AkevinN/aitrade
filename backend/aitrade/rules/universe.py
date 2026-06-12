"""
标的宇宙时点过滤与覆盖率统计（Phase 5 Task 5.2）。

提供两个公开 API：
- filter_by_listing  ：纯函数，按上市/退市日期过滤标的（时点过滤）
- UniverseCoverage   ：数据类，记录回测宇宙覆盖率统计

设计原则
--------
- filter_by_listing 无 I/O 副作用，list_dates / delist_dates 由调用方传入；
  上市/退市日期不可得时（None）保守保留标的，并由调用方记录 warning。
- 覆盖率 < 0.8 时调用方应在结果中附 warning 文案。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class UniverseCoverage:
    """回测宇宙覆盖率统计。

    Attributes:
        requested: 请求标的总数。
        with_bars: 有行情数据的标的数。
        with_fundamental: 有基本面数据的标的数（不需要基本面的策略 = with_bars）。
        excluded_not_listed: 时点过滤剔除数（区间内未上市/已退市）。
        coverage_ratio: with_bars / max(1, requested - excluded_not_listed)。
        warnings: 覆盖率告警或数据可用性说明列表。
    """

    requested: int
    with_bars: int
    with_fundamental: int
    excluded_not_listed: int
    coverage_ratio: float
    warnings: list[str] = field(default_factory=list)


def filter_by_listing(
    vt_symbols: list[str],
    as_of: date,
    *,
    list_dates: dict[str, date | None],
    delist_dates: dict[str, date | None],
    min_list_days: int = 0,
) -> tuple[list[str], list[str]]:
    """时点过滤纯函数：按上市/退市日期筛选标的。

    规则
    ----
    1. list_date 已知且 list_date > as_of → 尚未上市 → 剔除。
    2. delist_date 已知且 delist_date < as_of → 已退市 → 剔除。
    3. min_list_days > 0 且 list_date 已知且 (as_of - list_date).days < min_list_days
       → 上市天数不足 → 剔除。
    4. list_date / delist_date 为 None（未知）→ 保守保留（不剔除）。

    Args:
        vt_symbols: 待过滤的标的列表。
        as_of: 评估时点（通常为回测结束日）。
        list_dates: {vt_symbol: 上市日期 | None}，None 表示日期不可得。
        delist_dates: {vt_symbol: 退市日期 | None}，None 表示日期不可得。
        min_list_days: 最短上市天数要求（0 表示不限）。

    Returns:
        (保留标的列表, 被剔除标的列表)
    """
    kept: list[str] = []
    excluded: list[str] = []

    for sym in vt_symbols:
        list_date = list_dates.get(sym)
        delist_date = delist_dates.get(sym)

        # 规则 1：尚未上市
        if list_date is not None and list_date > as_of:
            excluded.append(sym)
            continue

        # 规则 2：已退市
        if delist_date is not None and delist_date < as_of:
            excluded.append(sym)
            continue

        # 规则 3：上市天数不足
        if (
            min_list_days > 0
            and list_date is not None
            and (as_of - list_date).days < min_list_days
        ):
            excluded.append(sym)
            continue

        kept.append(sym)

    return kept, excluded


def compute_coverage(
    requested: list[str],
    symbols_with_bars: list[str] | set[str],
    symbols_with_fundamental: list[str] | set[str] | None,
    excluded: list[str],
    *,
    coverage_warnings: list[str] | None = None,
) -> UniverseCoverage:
    """根据实际覆盖情况计算 UniverseCoverage。

    Args:
        requested: 原始请求标的列表。
        symbols_with_bars: 有行情数据的标的集合。
        symbols_with_fundamental: 有基本面数据的标的集合，None 表示策略不需要基本面
                                  （此时 with_fundamental = with_bars）。
        excluded: filter_by_listing 返回的被剔除标的列表。
        coverage_warnings: 附加告警（如上市日期不可用）。

    Returns:
        UniverseCoverage 实例，覆盖率 < 0.8 时自动附 warning 文案。
    """
    bars_set = set(symbols_with_bars)
    n_requested = len(requested)
    n_with_bars = sum(1 for s in requested if s in bars_set)
    n_excluded = len(excluded)

    if symbols_with_fundamental is None:
        n_with_fundamental = n_with_bars
    else:
        fund_set = set(symbols_with_fundamental)
        n_with_fundamental = sum(1 for s in requested if s in fund_set)

    effective = max(1, n_requested - n_excluded)
    ratio = n_with_bars / effective

    warnings: list[str] = list(coverage_warnings or [])
    if ratio < 0.8:
        warnings.append(
            f"宇宙覆盖率偏低（{ratio:.1%}），"
            f"请求 {n_requested} 只标的，"
            f"时点过滤剔除 {n_excluded} 只，"
            f"有行情数据 {n_with_bars} 只，"
            "请检查数据是否已下载或回测区间是否合理。"
        )

    return UniverseCoverage(
        requested=n_requested,
        with_bars=n_with_bars,
        with_fundamental=n_with_fundamental,
        excluded_not_listed=n_excluded,
        coverage_ratio=ratio,
        warnings=warnings,
    )
