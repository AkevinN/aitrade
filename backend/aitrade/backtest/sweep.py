"""
参数网格扫描工具（Phase 2）。

纯函数，模式参考 validation.py 的 cost_sensitivity_table：逐项调回调取 statistics，
汇总关键指标，对缺键容错返回 None。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def param_sweep_table(
    run_with_params: Callable[[dict], dict[str, Any]],
    grid: list[dict],
) -> list[dict[str, Any]]:
    """对 grid 中每组参数跑一次回测，汇总关键指标。

    Args:
        run_with_params: 给定参数 override dict（含 strategy_params/signal_params），
                         返回该组参数下的完整回测结果 dict（result，含 statistics 子 dict）。
        grid: 参数覆盖列表，每项含 strategy_params 和/或 signal_params 的 override。

    Returns:
        每组参数一行：params + total_return / sharpe_ratio / max_ddpercent /
        total_net_pnl / trade_count（缺键容错返回 None）。
    """
    def _safe_float(stats: dict[str, Any], key: str) -> float | None:
        """从 statistics dict 安全读取浮点指标，缺键或转换失败返回 None。"""
        val = stats.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    rows: list[dict[str, Any]] = []
    for override in grid:
        result = run_with_params(override) or {}
        stats: dict[str, Any] = result.get("statistics") or {}
        rows.append({
            "params": override,
            "total_return": _safe_float(stats, "total_return"),
            "sharpe_ratio": _safe_float(stats, "sharpe_ratio"),
            "max_ddpercent": _safe_float(stats, "max_ddpercent"),
            "total_net_pnl": _safe_float(stats, "total_net_pnl"),
            "trade_count": result.get("trade_count"),
        })
    return rows
