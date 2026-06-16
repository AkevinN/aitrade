"""
回测验证工具（迭代 5）：成本敏感性 + 样本外 / walk-forward 切分。

这些工具用于上线前质量闭环：
- cost_sensitivity_table：在「基准 / 佣金×2 / 滑点+5bp」等情景下对比关键指标，
  判断策略是否经得起成本恶化（高换手策略常在此归零）。
- time_series_holdout / walk_forward_windows：按时间顺序切分训练/留出集，
  保证样本外验证不混入未来信息（不打乱）。
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Callable, Optional

from .scheme import CostConfig


# ---------------------------------------------------------------------------
# 成本敏感性
# ---------------------------------------------------------------------------
def default_cost_scenarios(base: CostConfig) -> list[tuple[str, CostConfig]]:
    """生成三种默认成本压测情景：基准、佣金×2、滑点+5bp。

    Args:
        base: 基准成本配置。

    Returns:
        每项为 (情景名称, CostConfig) 的元组列表：
        - ``"基准"``：原始成本；
        - ``"佣金×2"``：commission_rate 翻倍；
        - ``"滑点+5bp"``：slippage 增加 0.0005。
    """
    return [
        ("基准", base),
        ("佣金×2", base.model_copy(update={"commission_rate": base.commission_rate * 2})),
        ("滑点+5bp", base.model_copy(update={"slippage": base.slippage + 0.0005})),
    ]


def cost_sensitivity_table(
    run_with_cost: Callable[[CostConfig], dict[str, Any]],
    base_cost: CostConfig,
    scenarios: Optional[list[tuple[str, CostConfig]]] = None,
) -> list[dict[str, Any]]:
    """对每个成本情景跑一次回测，汇总关键指标。

    Args:
        run_with_cost: 给定 CostConfig，返回该情景下的 statistics dict。
        base_cost: 基准成本。
        scenarios: 情景列表 [(名称, CostConfig)]；缺省用 default_cost_scenarios。

    Returns:
        每情景一行：scenario / total_return / sharpe_ratio / max_ddpercent /
        total_net_pnl / total_commission。
    """
    scenarios = scenarios or default_cost_scenarios(base_cost)
    rows: list[dict[str, Any]] = []
    for name, cost in scenarios:
        stats = run_with_cost(cost) or {}
        rows.append({
            "scenario": name,
            "total_return": float(stats.get("total_return", 0.0)),
            "sharpe_ratio": float(stats.get("sharpe_ratio", 0.0)),
            "max_ddpercent": float(stats.get("max_ddpercent", 0.0)),
            "total_net_pnl": float(stats.get("total_net_pnl", 0.0)),
            "total_commission": float(stats.get("total_commission", 0.0)),
        })
    return rows


# ---------------------------------------------------------------------------
# 样本外 / walk-forward 切分（按时间顺序，不打乱）
# ---------------------------------------------------------------------------
def time_series_holdout(items: list[Any], train_ratio: float) -> tuple[list[Any], list[Any]]:
    """按时间顺序切分为 (训练集, 留出集)，保持原始顺序，不打乱。

    Args:
        items: 按时间顺序排列的样本列表（如日期列表或 bar 列表）。
        train_ratio: 训练集比例，必须在 (0, 1) 之间（不含端点）。

    Returns:
        (train, holdout) 元组，两个子列表均保持原始相对顺序，合并等于 items。

    Raises:
        ValueError: train_ratio 不在 (0, 1) 开区间内时抛出。

    Example:
        >>> train, test = time_series_holdout(list(range(10)), 0.8)
        >>> len(train), len(test)
        (8, 2)
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio 必须在 (0,1) 之间，当前 {train_ratio}")
    split = int(len(items) * train_ratio)
    return items[:split], items[split:]


def walk_forward_windows(
    start: date,
    end: date,
    train_days: int,
    test_days: int,
    step_days: Optional[int] = None,
) -> list[dict[str, tuple[date, date]]]:
    """生成滚动 walk-forward 时间窗口列表（不含未来信息）。

    每个窗口的 test 区间紧接 train 区间之后（test_start == train_end），
    保证样本外验证不混入训练期数据。step_days 控制游标每轮向前移动的天数，
    缺省等于 test_days（无重叠滚动）；设为小于 test_days 可生成重叠测试窗口。

    Args:
        start: 整体数据起始日期，首个 train 窗口从此日期开始。
        end: 整体数据截止日期，test_end > end 时停止生成。
        train_days: 每个训练窗口的日历天数（必须为正整数）。
        test_days: 每个测试窗口的日历天数（必须为正整数）。
        step_days: 游标每轮步进天数；None 时等于 test_days（不重叠）。

    Returns:
        每元素为 ``{"train": (train_start, train_end), "test": (test_start, test_end)}``
        的字典列表；若整个区间内无法生成任何完整窗口则返回空列表。

    Raises:
        ValueError: train_days 或 test_days 不为正数时抛出。

    Example:
        >>> from datetime import date
        >>> windows = walk_forward_windows(date(2020,1,1), date(2021,12,31), 180, 60)
        >>> len(windows) > 0
        True
    """
    if train_days <= 0 or test_days <= 0:
        raise ValueError("train_days / test_days 必须为正")
    step = step_days or test_days
    windows: list[dict[str, tuple[date, date]]] = []
    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        if test_end > end:
            break
        windows.append({"train": (train_start, train_end), "test": (test_start, test_end)})
        cursor = cursor + timedelta(days=step)
    return windows
