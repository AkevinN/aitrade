"""
universe.py 单元测试（Phase 5 Task 5.2）。

覆盖：
filter_by_listing
  1. 基本保留（list_date <= as_of，delist_date > as_of）
  2. 区间前已退市（delist_date < as_of）→ 剔除
  3. 区间后才上市（list_date > as_of）→ 剔除
  4. None 保守保留（list_date=None / delist_date=None）
  5. min_list_days：上市天数不足 → 剔除
  6. min_list_days：上市天数足够 → 保留
  7. 混合场景：一只保留、一只剔除

compute_coverage
  8. 全覆盖：coverage_ratio = 1.0
  9. 部分覆盖：ratio = with_bars / (requested - excluded)
  10. 覆盖率 < 0.8 → warnings 含告警文案
  11. 覆盖率 >= 0.8 → warnings 无覆盖率告警（可有其他 warning）
  12. excluded 全部剔除（分母保底为 1）：ratio = with_bars / 1
  13. symbols_with_fundamental=None → with_fundamental = with_bars
"""

from __future__ import annotations

from datetime import date

import pytest

from aitrade.rules.universe import compute_coverage, filter_by_listing

# ============================================================================
# 辅助
# ============================================================================

AS_OF = date(2024, 6, 30)  # 评估时点

SYM_A = "600519.SSE"
SYM_B = "000001.SZSE"
SYM_C = "688001.SSE"
SYM_D = "301001.SZSE"


# ============================================================================
# filter_by_listing 测试
# ============================================================================


def test_filter_keeps_listed_active() -> None:
    """上市日 <= as_of，退市日 > as_of → 应保留。"""
    kept, excluded = filter_by_listing(
        [SYM_A],
        as_of=AS_OF,
        list_dates={SYM_A: date(2001, 7, 18)},
        delist_dates={SYM_A: date(2099, 12, 31)},
    )
    assert SYM_A in kept
    assert SYM_A not in excluded


def test_filter_excludes_delisted_before_as_of() -> None:
    """delist_date < as_of → 已退市 → 剔除。"""
    delist = date(2024, 1, 15)  # 早于 AS_OF
    kept, excluded = filter_by_listing(
        [SYM_A],
        as_of=AS_OF,
        list_dates={SYM_A: date(2001, 7, 18)},
        delist_dates={SYM_A: delist},
    )
    assert SYM_A in excluded
    assert SYM_A not in kept


def test_filter_excludes_not_yet_listed() -> None:
    """list_date > as_of → 尚未上市 → 剔除。"""
    future_list = date(2025, 3, 1)
    kept, excluded = filter_by_listing(
        [SYM_C],
        as_of=AS_OF,
        list_dates={SYM_C: future_list},
        delist_dates={SYM_C: None},
    )
    assert SYM_C in excluded
    assert SYM_C not in kept


def test_filter_none_list_date_keeps_conservatively() -> None:
    """list_date=None → 保守保留（不剔除）。"""
    kept, excluded = filter_by_listing(
        [SYM_B],
        as_of=AS_OF,
        list_dates={SYM_B: None},
        delist_dates={SYM_B: None},
    )
    assert SYM_B in kept
    assert SYM_B not in excluded


def test_filter_none_delist_date_keeps_conservatively() -> None:
    """delist_date=None → 保守保留（不视为已退市）。"""
    kept, excluded = filter_by_listing(
        [SYM_A],
        as_of=AS_OF,
        list_dates={SYM_A: date(2001, 7, 18)},
        delist_dates={SYM_A: None},
    )
    assert SYM_A in kept


def test_filter_min_list_days_excludes_insufficient() -> None:
    """上市天数不足 min_list_days → 剔除。"""
    # as_of = 2024-06-30，list_date = 2024-06-25，天数 = 5
    recent_list = date(2024, 6, 25)
    kept, excluded = filter_by_listing(
        [SYM_D],
        as_of=AS_OF,
        list_dates={SYM_D: recent_list},
        delist_dates={SYM_D: None},
        min_list_days=30,
    )
    assert SYM_D in excluded


def test_filter_min_list_days_keeps_sufficient() -> None:
    """上市天数 >= min_list_days → 保留。"""
    old_list = date(2020, 1, 1)
    kept, excluded = filter_by_listing(
        [SYM_A],
        as_of=AS_OF,
        list_dates={SYM_A: old_list},
        delist_dates={SYM_A: None},
        min_list_days=30,
    )
    assert SYM_A in kept


def test_filter_mixed_scenario() -> None:
    """混合场景：SYM_A 保留，SYM_B 已退市剔除。"""
    kept, excluded = filter_by_listing(
        [SYM_A, SYM_B],
        as_of=AS_OF,
        list_dates={SYM_A: date(2001, 7, 18), SYM_B: date(2005, 1, 10)},
        delist_dates={SYM_A: None, SYM_B: date(2024, 3, 1)},
    )
    assert SYM_A in kept
    assert SYM_B in excluded
    assert len(kept) == 1
    assert len(excluded) == 1


def test_filter_symbol_not_in_dicts_keeps_conservatively() -> None:
    """标的不在 list_dates / delist_dates dict 中（.get 返回 None）→ 保守保留。"""
    kept, excluded = filter_by_listing(
        [SYM_C],
        as_of=AS_OF,
        list_dates={},
        delist_dates={},
    )
    assert SYM_C in kept


# ============================================================================
# compute_coverage 测试
# ============================================================================


def test_coverage_full() -> None:
    """全量有数据 → coverage_ratio = 1.0。"""
    cov = compute_coverage(
        requested=[SYM_A, SYM_B],
        symbols_with_bars=[SYM_A, SYM_B],
        symbols_with_fundamental=None,
        excluded=[],
    )
    assert cov.coverage_ratio == pytest.approx(1.0)
    assert cov.with_bars == 2
    assert cov.requested == 2
    assert cov.excluded_not_listed == 0
    # 无覆盖率告警（ratio >= 0.8）
    coverage_warns = [w for w in cov.warnings if "覆盖率偏低" in w]
    assert len(coverage_warns) == 0


def test_coverage_partial() -> None:
    """部分有数据：ratio = with_bars / (requested - excluded)。"""
    # requested=4，excluded=1，with_bars=2 → ratio = 2/3
    syms = [SYM_A, SYM_B, SYM_C, SYM_D]
    cov = compute_coverage(
        requested=syms,
        symbols_with_bars=[SYM_A, SYM_B],
        symbols_with_fundamental=None,
        excluded=[SYM_D],
    )
    assert cov.requested == 4
    assert cov.excluded_not_listed == 1
    assert cov.with_bars == 2
    expected_ratio = 2 / 3
    assert cov.coverage_ratio == pytest.approx(expected_ratio)


def test_coverage_low_triggers_warning() -> None:
    """覆盖率 < 0.8 → warnings 含覆盖率偏低提示。"""
    syms = [SYM_A, SYM_B, SYM_C, SYM_D]
    cov = compute_coverage(
        requested=syms,
        symbols_with_bars=[SYM_A],  # 1/4 = 0.25
        symbols_with_fundamental=None,
        excluded=[],
    )
    assert cov.coverage_ratio == pytest.approx(0.25)
    coverage_warns = [w for w in cov.warnings if "覆盖率偏低" in w]
    assert len(coverage_warns) >= 1


def test_coverage_high_no_coverage_warning() -> None:
    """覆盖率 >= 0.8 → warnings 中无 '覆盖率偏低'。"""
    cov = compute_coverage(
        requested=[SYM_A, SYM_B],
        symbols_with_bars=[SYM_A, SYM_B],
        symbols_with_fundamental=None,
        excluded=[],
    )
    coverage_warns = [w for w in cov.warnings if "覆盖率偏低" in w]
    assert len(coverage_warns) == 0


def test_coverage_all_excluded_denominator_clamp() -> None:
    """全部剔除时分母保底为 1，避免 ZeroDivisionError。"""
    syms = [SYM_A, SYM_B]
    cov = compute_coverage(
        requested=syms,
        symbols_with_bars=[],
        symbols_with_fundamental=None,
        excluded=syms,  # 全剔除
    )
    # with_bars=0, excluded=2, effective=max(1, 2-2)=1, ratio=0/1=0
    assert cov.coverage_ratio == pytest.approx(0.0)


def test_coverage_fundamental_none_equals_with_bars() -> None:
    """symbols_with_fundamental=None 时 with_fundamental = with_bars。"""
    cov = compute_coverage(
        requested=[SYM_A, SYM_B],
        symbols_with_bars=[SYM_A],
        symbols_with_fundamental=None,
        excluded=[],
    )
    assert cov.with_fundamental == cov.with_bars


def test_coverage_fundamental_given() -> None:
    """显式传入 symbols_with_fundamental 时独立计算。"""
    cov = compute_coverage(
        requested=[SYM_A, SYM_B, SYM_C],
        symbols_with_bars=[SYM_A, SYM_B, SYM_C],
        symbols_with_fundamental=[SYM_A],  # 只有 A 有基本面
        excluded=[],
    )
    assert cov.with_fundamental == 1
    assert cov.with_bars == 3


def test_coverage_extra_warnings_passed_through() -> None:
    """传入的 coverage_warnings 应出现在结果 warnings 中。"""
    cov = compute_coverage(
        requested=[SYM_A],
        symbols_with_bars=[SYM_A],
        symbols_with_fundamental=None,
        excluded=[],
        coverage_warnings=["上市/退市日期数据不可用，时点过滤未生效（v1 降级）"],
    )
    assert any("时点过滤未生效" in w for w in cov.warnings)
