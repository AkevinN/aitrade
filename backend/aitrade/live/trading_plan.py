"""
交易计划（Trading Plan Automation，新增 2）：把一次决策所需的完整配置保存为
可复用、可编辑、可启用/停用的计划，作为手动「按计划触发」与自动调度的输入。

持久化采用 JSON 文件落盘（每 plan_id 一文件），复用既有 `DecisionStore` 范式。

脱敏红线（需求 2.9 / 9.4）：`notify_channels` 仅存通道名（如 ["dingtalk", "wecom"]），
**绝不**存储任何 webhook/secret/token；凭证由 `notifier_channels.build_notifier`
在运行时从环境变量解析。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .legacy_migration import migrate_plan


@dataclass
class TradingPlan:
    plan_id: str
    name: str
    # —— 决策参数（与 LiveDecisionRequest 同义；触发时构造请求用）——
    model: str
    vt_symbol: str
    scheme: str
    buy_threshold: float = 0.6
    position_ratio: float = 0.95
    min_volume: int = 100
    model_version: str = ""
    data_source: str = "pull"  # "upload" | "pull"
    should_exit: bool = False
    halted: bool = False
    # 组合快照与风控（自动触发用计划保存时的快照；快照滞后见 spec「已知缺口」）
    portfolio: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    # —— 调度配置（时刻原生）——
    enabled: bool = False
    bar_freq: str = "1d"  # 决策 bar 频率；v1 仅 "1d"（日内属 Phase 2）
    # 每交易日的调度唤醒时刻（HH:MM）；每个时刻触发一次，决策 bar 由 as_of 截断决定
    trigger_times: list[str] = field(default_factory=lambda: ["15:05"])
    # —— 通知 ——（仅通道名，无凭证）
    notify_channels: list[str] = field(default_factory=list)
    # —— 元信息 ——
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]


def effective_trigger_times(plan: "TradingPlan") -> list[str]:
    """计划生效的唤醒时刻集合（去重升序；"HH:MM" 字典序即时间序）。"""
    return sorted({t for t in plan.trigger_times if t})


class TradingPlanStore:
    """交易计划 JSON 持久化（每 plan_id 一文件），复用 DecisionStore 范式。

    读取时经一次性迁移（旧 data_basis/decision_time(s) → bar_freq/trigger_times）并回写，
    使磁盘逐步收敛为时刻原生结构（零残留）。
    """

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, plan_id: str) -> Path:
        safe = plan_id.replace("/", "_").replace(":", "_")
        return self.base_path / f"{safe}.json"

    def _load(self, path: Path, *, rewrite: bool) -> TradingPlan:
        raw = json.loads(path.read_text(encoding="utf-8"))
        migrated = migrate_plan(raw)
        if rewrite and migrated != raw:
            path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2), encoding="utf-8")
        return TradingPlan(**migrated)

    def get(self, plan_id: str) -> Optional[TradingPlan]:
        path = self._path(plan_id)
        if not path.exists():
            return None
        return self._load(path, rewrite=True)

    def save(self, plan: TradingPlan) -> Path:
        path = self._path(plan.plan_id)
        path.write_text(
            json.dumps(asdict(plan), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def delete(self, plan_id: str) -> bool:
        path = self._path(plan_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_all(self) -> list[TradingPlan]:
        return [self._load(p, rewrite=False) for p in sorted(self.base_path.glob("*.json"))]
