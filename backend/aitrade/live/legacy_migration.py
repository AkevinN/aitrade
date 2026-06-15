"""
一次性遗留数据迁移（Decision Instant Unification）。

把「按天」时代的持久化 JSON 读为「按时刻」新模型：
- Decision：`trade_date` → `decision_bar_dt` / `as_of`（= 该日 Session_Close）/ `bar_freq="1d"`。
- Trading_Plan：`data_basis` / `decision_time` / `decision_times` → `bar_freq="1d"` / `trigger_times`。

设计红线（零残留）：本模块是**唯一**容忍旧字段的地方。`DecisionStore.get` /
`TradingPlanStore.get` 在读取时调用迁移并**回写**一次，之后磁盘即新结构；除此之外
全代码库不再出现 `trade_date` / `data_basis` / `decision_time(s)` 旧概念。

纯函数（dict→dict），无 I/O，便于确定性测试（Property DI-6）。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .decision_instant import session_close

_DEFAULT_TRIGGER_TIME = "15:05"


def migrate_decision(raw: dict[str, Any]) -> dict[str, Any]:
    """把旧 Decision JSON（含 `trade_date`）迁移为按时刻的新结构。

    旧字段 `trade_date` 映射为 `decision_bar_dt`/`as_of`（= 该日 Session_Close）/
    `bar_freq="1d"`。幂等：已是新结构（含 `decision_bar_dt`）则仅补全可能缺失的
    新字段默认值；同时兼容 Wave 2c 新增字段 trigger_source。纯函数、无 I/O。

    Args:
        raw: 反序列化后的 Decision 字典，可能是旧结构（含 `trade_date`）或新结构。

    Returns:
        浅拷贝后的新结构字典：含 `decision_bar_dt`/`as_of`/`bar_freq` 且补齐
        `trigger_source`（缺失时默认 ""）；不修改入参。
    """
    out = dict(raw)
    if "trade_date" in out and "decision_bar_dt" not in out:
        d = date.fromisoformat(out.pop("trade_date"))
        close_iso = session_close(d, "1d").isoformat()
        out["decision_bar_dt"] = close_iso
        out.setdefault("as_of", close_iso)
        out.setdefault("bar_freq", "1d")
    # Wave 2c：新字段向后兼容（旧 JSON 缺失时注入默认值）
    out.setdefault("trigger_source", "")
    return out


def migrate_plan(raw: dict[str, Any]) -> dict[str, Any]:
    """把旧 Trading_Plan JSON 迁移为按时刻的新结构。

    旧字段 `data_basis`/`decision_time`/`decision_times` 收敛为 `bar_freq="1d"` 与
    去重升序的 `trigger_times`（为空时回退到 _DEFAULT_TRIGGER_TIME "15:05"），并清除
    全部旧字段实现零残留；另补齐 Phase 3 M2 的 v2 字段默认值。幂等、纯函数、无 I/O。

    Args:
        raw: 反序列化后的 Trading_Plan 字典，可能是旧结构或新结构。

    Returns:
        浅拷贝后的新结构字典：含 `bar_freq`/`trigger_times`，旧字段已剔除，
        并补齐 strategy_type/signal_source/signal_params/trigger_schedule/portfolio_id
        的默认值（已有值不覆盖）；不修改入参。
    """
    out = dict(raw)
    if "bar_freq" not in out or "trigger_times" not in out:
        times = out.get("decision_times") or (
            [out["decision_time"]] if out.get("decision_time") else [_DEFAULT_TRIGGER_TIME]
        )
        out.setdefault("bar_freq", "1d")
        out["trigger_times"] = sorted({t for t in times if t}) or [_DEFAULT_TRIGGER_TIME]
    # 清除旧字段（零残留）。
    for key in ("data_basis", "decision_time", "decision_times"):
        out.pop(key, None)
    # Phase 3 M2：新增 v2 字段默认值（幂等 setdefault，已有值不覆盖）。
    out.setdefault("strategy_type", "cnn")
    out.setdefault("signal_source", "")
    out.setdefault("signal_params", {})
    out.setdefault("trigger_schedule", "daily")
    out.setdefault("portfolio_id", "")
    return out
