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
    """旧 Decision JSON（含 `trade_date`）→ 新结构（`decision_bar_dt`/`as_of`/`bar_freq`）。

    幂等：已是新结构（含 `decision_bar_dt`）则原样返回。
    """
    if "decision_bar_dt" in raw or "trade_date" not in raw:
        return raw
    out = dict(raw)
    d = date.fromisoformat(out.pop("trade_date"))
    close_iso = session_close(d, "1d").isoformat()
    out["decision_bar_dt"] = close_iso
    out.setdefault("as_of", close_iso)
    out.setdefault("bar_freq", "1d")
    return out


def migrate_plan(raw: dict[str, Any]) -> dict[str, Any]:
    """旧 Trading_Plan JSON（`data_basis`/`decision_time(s)`）→ 新（`bar_freq`/`trigger_times`）。

    幂等：已含 `bar_freq` 与 `trigger_times` 则仅清除可能残留的旧字段。
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
    return out
