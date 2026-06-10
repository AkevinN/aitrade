"""
决策记录与持久化（决策时刻统一）：一次信号决策落盘，保证可回溯 + 幂等。

Decision 以「时刻」为单元：`decision_bar_dt`（决策 bar 的时刻）+ `as_of`（决策时刻）+
`bar_freq`（决策 bar 频率，`1d` 即日频），取代旧 `trade_date`。`signal_id` 由
`decision_instant.make_signal_id(decision_bar_dt, bar_freq, scheme, model_version)` 生成，
同 signal_id 不重复处理/重复提醒。旧 JSON（含 `trade_date`）在读取时经一次性迁移转入新结构。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .legacy_migration import migrate_decision


@dataclass
class Decision:
    signal_id: str            # 幂等键，如 "2026-06-08:eod_buy_v1:model@v3"
    decision_bar_dt: str      # 决策 bar 时刻 ISO（取代 trade_date）
    as_of: str                # 决策时刻 ISO
    bar_freq: str             # "1d" | ...（决策 bar 频率）
    scheme: str
    action: str               # buy / sell / hold
    vt_symbol: Optional[str] = None
    volume: int = 0
    price: Optional[float] = None
    signal: Optional[float] = None
    reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class DecisionStore:
    """决策的 JSON 持久化（每 signal_id 一文件），支持幂等查询。"""

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, signal_id: str) -> Path:
        safe = signal_id.replace("/", "_").replace(":", "_")
        return self.base_path / f"{safe}.json"

    def exists(self, signal_id: str) -> bool:
        return self._path(signal_id).exists()

    def get(self, signal_id: str) -> Optional[Decision]:
        path = self._path(signal_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        migrated = migrate_decision(raw)  # 旧 trade_date → 时刻结构（一次性，唯一兼容处）
        if migrated != raw:
            path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2), encoding="utf-8")
        return Decision(**migrated)

    def save(self, decision: Decision) -> Path:
        path = self._path(decision.signal_id)
        path.write_text(json.dumps(asdict(decision), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def list_ids(self) -> list[str]:
        # 仅纳入决策文件 {signal_id}.json，排除 sibling 的 {signal_id}.trace.json，
        # 避免 trace 文件的 stem（"{signal_id}.trace"）被误当成独立决策 id（需求 8.3）。
        return sorted(
            p.stem
            for p in self.base_path.glob("*.json")
            if not p.name.endswith(".trace.json")
        )
