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
    """一次决策的不可变记录（持久化单元）。

    以 signal_id 为幂等键：同 signal_id 不重复处理/提醒/落盘。
    所有时刻字段均为 ISO 字符串，便于 JSON 序列化与文件持久化。

    Attributes:
        signal_id:       幂等键，如 "2026-06-08:eod_buy_v1:model@v3"。
        decision_bar_dt: 决策 bar 的收盘时刻（ISO），取代旧 trade_date。
        as_of:           决策产出时刻（ISO），即编排器被触发的时刻。
        bar_freq:        决策 bar 频率，"1d" 为日频，分钟频如 "5m"。
        scheme:          方案名，参与 signal_id 与提醒标题。
        action:          决策动作："buy" / "sell" / "hold"。
        vt_symbol:       目标标的，如 "000001.SZSE"。
        volume:          建议手数（股数），0 表示持有观望。
        price:           建议价位（决策 bar 收盘价）。
        signal:          模型输出信号值（概率或得分）。
        reason:          决策理由文本，面向人工审核。
        created_at:      记录创建时刻（ISO，秒精度）。
        trigger_source:  触发来源："scheduler" | "manual" | ""（旧数据兼容）。
    """

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
    trigger_source: str = ""  # "scheduler" | "manual" | ""（旧数据默认空串）


class DecisionStore:
    """决策的 JSON 持久化（每 signal_id 一文件），支持幂等查询。"""

    def __init__(self, base_path: Path | str) -> None:
        """初始化 DecisionStore。

        Args:
            base_path: 决策 JSON 文件存放目录；不存在时自动创建。
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, signal_id: str) -> Path:
        """将 signal_id 安全化后返回对应 JSON 文件路径。

        Args:
            signal_id: 幂等键，"/" 与 ":" 替换为 "_" 以兼容文件系统。

        Returns:
            该 signal_id 对应的 .json 文件路径（不保证文件存在）。
        """
        safe = signal_id.replace("/", "_").replace(":", "_")
        return self.base_path / f"{safe}.json"

    def exists(self, signal_id: str) -> bool:
        """判断该 signal_id 是否已有落盘决策（幂等判定入口）。

        Args:
            signal_id: 幂等键。

        Returns:
            True 表示文件存在，该 signal_id 已处理过。
        """
        return self._path(signal_id).exists()

    def get(self, signal_id: str) -> Optional[Decision]:
        """读取指定 signal_id 的决策；不存在返回 None。

        读取时自动执行一次性迁移（旧 trade_date → decision_bar_dt/as_of/bar_freq），
        迁移后若内容有变则回写磁盘，使磁盘逐步收敛为新结构。

        Args:
            signal_id: 幂等键。

        Returns:
            Decision 对象；文件不存在或 signal_id 无效时返回 None。
        """
        path = self._path(signal_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        migrated = migrate_decision(raw)  # 旧 trade_date → 时刻结构（一次性，唯一兼容处）
        if migrated != raw:
            path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2), encoding="utf-8")
        return Decision(**migrated)

    def save(self, decision: Decision) -> Path:
        """将决策序列化为 JSON 并落盘，返回写入路径。

        同 signal_id 重复调用会覆盖（不幂等——调用方应在 exists() 后才调用 save）。

        Args:
            decision: 待持久化的 Decision 对象。

        Returns:
            写入的 .json 文件路径。
        """
        path = self._path(decision.signal_id)
        path.write_text(json.dumps(asdict(decision), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def list_ids(self) -> list[str]:
        """返回所有活跃决策的 signal_id 列表（升序）。

        仅纳入决策文件 {signal_id}.json，排除 sibling 的 .trace.json，
        避免 trace 文件 stem 被误当成独立决策 id（需求 8.3）。
        glob 不递归，archive/ 子目录下的归档文件天然不在列表/幂等判定范围内。

        Returns:
            signal_id 字符串列表，按字典序升序排列。
        """
        return sorted(
            p.stem
            for p in self.base_path.glob("*.json")
            if not p.name.endswith(".trace.json")
        )

    def archive(self, signal_id: str) -> Optional[Path]:
        """归档式删除：决策文件移入 archive/ 子目录（文件名追加时间戳）。

        解除该 signal_id 的幂等占位——之后同一 Decision_Bar 可重新产出决策与提醒；
        归档文件保留审计痕迹，不被 get/list_ids 纳入。

        Args:
            signal_id: 待归档决策的幂等键。

        Returns:
            归档后的文件路径（archive/{stem}.{时间戳}.json）；该 signal_id
            无落盘文件时返回 None。
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
