"""
调仓决策实体与持久化（RebalanceDecision / RebalanceStore）。

Phase 3 M2 规则策略调仓：多标的组合调仓决策落盘，保证可回溯 + 幂等。

- `RebalanceDecision`：一次规则策略产出的调仓指令，含 items（增减持列表）、
  目标持仓快照、风控摘要；`signal_id` 为幂等键。
- `RebalanceStore`：JSON 文件持久化（每 signal_id 一文件），范式对齐 DecisionStore：
  - `save_if_absent`：幂等占位语义（get 命中即不写），返回 (saved, existing)。
  - `get`：dict→dataclass，items 嵌套还原。
  - `list_ids` / `list_all`：按文件名升序。
  - `delete`（归档式）：文件移入 archive/ 子目录，解除幂等占位，保留审计痕迹。
  - `update_status`：确认时读改写，tmp+os.replace 原子替换（比 DecisionStore 更稳）。

存储目录：`LIVE_REBALANCE_PATH`（AITRADE_HOME/live/rebalances），由 config.py 统一声明。
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RebalanceItem:
    """单只标的调仓指令。"""

    vt_symbol: str
    action: str             # "buy" | "sell"
    volume: int
    price: float | None = None    # 参考价（决策时刻行情）
    signal: float | None = None
    reason: str = ""


@dataclass
class RebalanceDecision:
    """一次规则策略产出的调仓决策（幂等键 signal_id，由编排器负责生成）。"""

    signal_id: str          # 幂等键（make_signal_id 产出，scheme 命名空间 "rule:..." 由编排器负责）
    decision_bar_dt: str    # 决策 bar 时刻 ISO
    as_of: str              # 决策时刻 ISO
    bar_freq: str
    scheme: str
    portfolio_id: str
    items: list[RebalanceItem]
    target_portfolio: dict[str, int]  # 调仓后目标持仓（stock → 股数）
    risk_summary: list[dict] = field(default_factory=list)   # RiskInspector records [{check, passed, detail}]
    status: str = "proposed"          # "proposed" | "confirmed"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    confirmed_at: str = ""            # 确认时间，空串=未确认
    trigger_source: str = ""          # "scheduler" | "manual" | ""（旧数据默认空串）
    elapsed_ms: int | None = None     # 编排器入口到落盘完成的毫秒数
    notify_ok: bool | None = None     # Notifier.send 实测返回值；未尝试发送（幂等命中/hold）时为 None


def _decision_from_dict(raw: dict) -> RebalanceDecision:
    """dict → RebalanceDecision（嵌套还原 items 列表）。

    Wave 2c migrate 钩子：setdefault 新增字段（旧 JSON 缺失时注入默认值，不抛错不回写）。
    """
    data = dict(raw)
    # items 嵌套还原：list[dict] → list[RebalanceItem]
    data["items"] = [
        RebalanceItem(**item) if isinstance(item, dict) else item
        for item in data.get("items", [])
    ]
    # Wave 2c：新字段向后兼容（旧 JSON 缺失时注入默认值）
    data.setdefault("trigger_source", "")
    data.setdefault("elapsed_ms", None)
    data.setdefault("notify_ok", None)
    return RebalanceDecision(**data)


class RebalanceStore:
    """调仓决策 JSON 持久化（每 signal_id 一文件），支持幂等查询。

    存储目录由调用方传入（通常为 config.LIVE_REBALANCE_PATH）；首次实例化自动 mkdir。
    """

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, signal_id: str) -> Path:
        safe = signal_id.replace("/", "_").replace(":", "_")
        return self.base_path / f"{safe}.json"

    def exists(self, signal_id: str) -> bool:
        return self._path(signal_id).exists()

    def get(self, signal_id: str) -> RebalanceDecision | None:
        """读取单条决策；不存在返回 None。"""
        path = self._path(signal_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _decision_from_dict(raw)

    def save(self, decision: RebalanceDecision) -> Path:
        """无条件写入（覆盖已有）。常规持久化场景使用此方法。"""
        path = self._path(decision.signal_id)
        path.write_text(json.dumps(asdict(decision), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def save_if_absent(self, decision: RebalanceDecision) -> tuple[bool, RebalanceDecision]:
        """幂等占位语义：若 signal_id 已存在则不写，返回 (False, existing)；
        否则写入返回 (True, decision)。幂等性与 DecisionStore 用途一致。
        """
        path = self._path(decision.signal_id)
        if path.exists():
            existing = self.get(decision.signal_id)
            return False, existing  # type: ignore[return-value]
        path.write_text(json.dumps(asdict(decision), ensure_ascii=False, indent=2), encoding="utf-8")
        return True, decision

    def list_ids(self) -> list[str]:
        """按文件名升序返回所有活跃决策的 signal_id（不含 archive/ 子目录）。"""
        return sorted(p.stem for p in self.base_path.glob("*.json"))

    def list_all(self) -> list[RebalanceDecision]:
        """按 signal_id 升序返回所有活跃决策对象。"""
        return [
            _decision_from_dict(json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(self.base_path.glob("*.json"))
        ]

    def delete(self, signal_id: str) -> Path | None:
        """归档式删除：决策文件移入 archive/ 子目录（文件名追加时间戳）。

        解除该 signal_id 的幂等占位——之后同一 signal_id 可重新产出决策；
        归档文件保留审计痕迹，不被 get/list_ids 纳入。不存在则返回 None。
        """
        path = self._path(signal_id)
        if not path.exists():
            return None
        archive_dir = self.base_path / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        target = archive_dir / f"{path.stem}.{stamp}{path.suffix}"
        path.rename(target)
        return target

    def update_status(
        self,
        signal_id: str,
        status: str,
        confirmed_at: str = "",
    ) -> RebalanceDecision | None:
        """原子读改写：更新 status（以及可选的 confirmed_at）。

        使用 tmp+os.replace 原子替换，比 DecisionStore.save 直接覆盖更稳（崩溃安全）。
        signal_id 不存在返回 None；成功返回更新后的 RebalanceDecision。
        非法 status 值直接抛 ValueError。
        """
        _VALID_STATUSES = {"proposed", "confirmed"}
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"非法 status 值 '{status}'，仅允许：{sorted(_VALID_STATUSES)}"
            )
        path = self._path(signal_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["status"] = status
        if confirmed_at:
            raw["confirmed_at"] = confirmed_at
        # tmp+os.replace 原子替换
        tmp_path = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp.json")
        try:
            tmp_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return _decision_from_dict(raw)
