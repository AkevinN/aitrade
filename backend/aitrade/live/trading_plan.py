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
    """交易计划：把一次决策所需的完整配置持久化为可复用、可编辑的单元。

    同时用于手动「按计划触发」与自动调度的输入（enable=True 时由 PlanScheduler 自动触发）。

    脱敏红线（需求 2.9 / 9.4）：notify_channels 仅存通道名，凭证由 build_notifier 在运行时解析。

    Attributes:
        plan_id:          唯一标识（12 位 hex），由 new_id() 生成。
        name:             计划展示名称（如 "etf_momentum_daily"）。
        model:            CNN 模型名（cnn 计划必填）。
        vt_symbol:        目标标的（cnn 计划必填）。
        scheme:           方案名，参与 signal_id 命名空间。
        bar_freq:         决策 bar 频率，"1d" 日频，分钟频为盘中监控模式。
        trigger_times:    日频计划的调度唤醒时刻列表（HH:MM）。
        notify_channels:  通知通道名列表（不含凭证）。
        strategy_type:    "cnn"（默认）| "rule"（规则信号组合调仓）。
        signal_source:    rule 计划的注册表信号源名（如 "etf_momentum"）。
        trigger_schedule: 调度粒度："daily" | "weekly_first" | "monthly_first"。
    """

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
    # —— Phase 3：策略类型与调度粒度 ——
    strategy_type: str = "cnn"             # "cnn" | "rule"
    signal_source: str = ""                # rule 计划的注册表信号源名（如 "etf_momentum"）
    signal_params: dict = field(default_factory=dict)
    trigger_schedule: str = "daily"        # "daily" | "weekly_first" | "monthly_first"
    portfolio_id: str = ""                 # rule 计划关联的持仓账本 id（Phase 3 后续任务消费）
    # —— 元信息 ——
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @staticmethod
    def new_id() -> str:
        """生成新的 plan_id（12 位 hex UUID 片段）。"""
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
        """从文件加载并迁移 TradingPlan，可选地回写迁移结果。

        Args:
            path:    JSON 文件路径（已存在）。
            rewrite: 若迁移后内容有变，是否回写磁盘（逐步收敛新结构）。

        Returns:
            迁移后的 TradingPlan 对象。
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        migrated = migrate_plan(raw)
        if rewrite and migrated != raw:
            path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2), encoding="utf-8")
        return TradingPlan(**migrated)

    def get(self, plan_id: str) -> Optional[TradingPlan]:
        """读取计划，不存在返回 None；读取时自动执行一次性迁移并回写。

        Args:
            plan_id: 计划唯一标识。

        Returns:
            TradingPlan 对象；文件不存在时返回 None。
        """
        path = self._path(plan_id)
        if not path.exists():
            return None
        return self._load(path, rewrite=True)

    def save(self, plan: TradingPlan) -> Path:
        """将计划序列化为 JSON 并落盘，返回写入路径。

        Args:
            plan: 待持久化的 TradingPlan 对象。

        Returns:
            写入的 .json 文件路径。
        """
        path = self._path(plan.plan_id)
        path.write_text(
            json.dumps(asdict(plan), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def delete(self, plan_id: str) -> bool:
        """删除计划文件。

        Args:
            plan_id: 计划唯一标识。

        Returns:
            True 表示文件存在且已删除，False 表示文件不存在。
        """
        path = self._path(plan_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_all(self) -> list[TradingPlan]:
        """返回目录下全部计划列表（按文件名升序，即 plan_id 字典序）。

        Returns:
            TradingPlan 列表；目录为空时返回空列表。不回写迁移结果（list 路径只读）。
        """
        return [self._load(p, rewrite=False) for p in sorted(self.base_path.glob("*.json"))]
