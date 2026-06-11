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
    """默认压测情景：基准、佣金×2、滑点+5bp。"""
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
    """按时间顺序切分为 (训练, 留出)，保持原始顺序，不打乱。"""
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
    """生成滚动 walk-forward 窗口。

    每个窗口 test 区间紧接 train 区间之后（test_start == train_end），
    保证样本外不含训练期信息。step_days 缺省等于 test_days（不重叠滚动）。
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
