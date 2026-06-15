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
    """单只标的的一条调仓指令（增持或减持），是 RebalanceDecision.items 的元素。

    描述"对某标的买/卖多少股"，并附带决策时刻的参考价与信号值，供回溯与下单使用。

    Attributes:
        vt_symbol: 合约代码，如 ``"000001.SZSE"``。
        action: 调仓方向，取值 ``"buy"``（增持）或 ``"sell"``（减持）。
        volume: 本次调仓股数，正整数。
        price: 决策时刻的参考价（当时行情），None 表示决策时无可用报价。
        signal: 触发该调仓的策略信号值，None 表示未携带信号。
        reason: 人类可读的调仓理由，空串表示未填写。
    """

    vt_symbol: str
    action: str             # "buy" | "sell"
    volume: int
    price: float | None = None    # 参考价（决策时刻行情）
    signal: float | None = None
    reason: str = ""


@dataclass
class RebalanceDecision:
    """一次规则策略产出的完整调仓决策，是 RebalanceStore 的持久化单元。

    汇集本轮所有标的的增减持指令（items）、调仓后目标持仓快照、风控摘要与状态，
    以 signal_id 作为幂等键保证同一信号只落盘一次。编排器产出、用户确认后流转 status。

    Attributes:
        signal_id: 幂等键，由 make_signal_id 产出，scheme 命名空间（如 ``"rule:..."``）由编排器负责。
        decision_bar_dt: 触发决策的那根 bar 的时刻，ISO 字符串。
        as_of: 做出决策的时刻，ISO 字符串。
        bar_freq: bar 周期，如 ``"d"``/``"30m"``。
        scheme: 产出该决策的策略方案标识。
        portfolio_id: 所属组合标识。
        items: 本轮所有标的的调仓指令列表，可能为空（无需调仓）。
        target_portfolio: 调仓后的目标持仓快照，``{合约代码: 股数}``。
        risk_summary: RiskInspector 风控检查记录，元素形如 ``{check, passed, detail}``；默认空列表。
        status: 决策状态，``"proposed"``（待确认）或 ``"confirmed"``（已确认）。
        created_at: 创建时间 ISO 字符串，默认取当前时刻（秒精度）。
        confirmed_at: 确认时间 ISO 字符串，空串表示尚未确认。
        trigger_source: 触发来源，``"scheduler"``/``"manual"``/``""``（旧数据默认空串）。
        elapsed_ms: 从编排器入口到落盘完成的耗时毫秒数，None 表示未记录。
        notify_ok: Notifier.send 的实测返回值；未尝试发送（幂等命中/hold）时为 None。
    """

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
        """绑定存储目录并确保其存在。

        Args:
            base_path: 决策文件根目录，通常为 config.LIVE_REBALANCE_PATH；
                不存在时会连同父目录一并创建。
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, signal_id: str) -> Path:
        """把 signal_id 映射为对应的 JSON 文件路径。

        将 signal_id 中的 ``/`` 与 ``:`` 替换为 ``_`` 以得到合法文件名。

        Args:
            signal_id: 决策幂等键。

        Returns:
            base_path 下形如 ``<safe_id>.json`` 的文件路径（不保证文件已存在）。
        """
        safe = signal_id.replace("/", "_").replace(":", "_")
        return self.base_path / f"{safe}.json"

    def exists(self, signal_id: str) -> bool:
        """判断指定 signal_id 的决策文件是否已落盘。

        Args:
            signal_id: 决策幂等键。

        Returns:
            文件存在返回 True，否则 False（不读取/解析文件内容）。
        """
        return self._path(signal_id).exists()

    def get(self, signal_id: str) -> RebalanceDecision | None:
        """按幂等键读取单条调仓决策。

        命中后从 JSON 反序列化并经 _decision_from_dict 还原（含 items 嵌套
        list[dict]→list[RebalanceItem]、旧字段默认值兜底）。

        Args:
            signal_id: 决策幂等键，定位对应 JSON 文件（路径映射见 _path）。

        Returns:
            还原后的 RebalanceDecision（items 已还原为 RebalanceItem 列表）；
            文件不存在时返回 None。
        """
        path = self._path(signal_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _decision_from_dict(raw)

    def save(self, decision: RebalanceDecision) -> Path:
        """无条件写入决策（覆盖同 signal_id 的已有文件）。

        常规持久化场景使用此方法；需要幂等占位（已存在则不写）时改用
        save_if_absent。文件名由 decision.signal_id 经 _path 映射得到。

        Args:
            decision: 待写入的调仓决策对象；其 signal_id 决定目标文件名，
                整体经 asdict 序列化为 JSON（含 items 等嵌套字段）。

        Returns:
            写入后的 JSON 文件路径（base_path 下的 ``<safe_id>.json``）。
        """
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
