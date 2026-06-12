"""
持仓账本（PositionBook）：跟踪多标的组合的当前持仓，记录最近确认的调仓 signal_id
以防重复确认，并提供原子写保证。

核心语义（apply_rebalance）：
  - 防重复：state.last_signal_id == decision.signal_id → ValueError（调用方转 409）。
  - 先全量校验后应用（原子性）：任一 sell 超卖 → 整笔拒绝，账本不变。
  - 归零 symbol 从 dict 移除，减少账本噪声。
  - 成功后 last_signal_id=signal_id、updated_at=now iso、save。
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .rebalance_decision import RebalanceDecision


@dataclass
class PortfolioState:
    """组合持仓快照（v1 不追踪现金）。"""

    portfolio_id: str
    positions: dict[str, int] = field(default_factory=dict)  # vt_symbol → 股数
    cash: float | None = None      # v1 可不追踪现金；None=未启用
    last_signal_id: str = ""          # 最近一次已确认调仓的幂等键（防重复确认）
    updated_at: str = ""


class PositionBook:
    """持仓账本：每 portfolio_id 一个 JSON 文件，tmp+os.replace 原子写。

    文件路径：`{base_path}/{portfolio_id}.json`（portfolio_id 中 / 与 : 替换为 _）。
    """

    def __init__(self, base_path: Path | str) -> None:
        """
        Args:
            base_path: 账本文件根目录，不存在时自动创建。
                       每个 portfolio_id 对应一个 ``<safe_portfolio_id>.json`` 文件。
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, portfolio_id: str) -> Path:
        safe = portfolio_id.replace("/", "_").replace(":", "_")
        return self.base_path / f"{safe}.json"

    def load(self, portfolio_id: str) -> PortfolioState:
        """加载持仓状态；文件缺失则返回空账本（positions={}）。"""
        path = self._path(portfolio_id)
        if not path.exists():
            return PortfolioState(portfolio_id=portfolio_id)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return PortfolioState(**raw)

    def save(self, state: PortfolioState) -> None:
        """原子写：tmp+os.replace，崩溃安全。"""
        path = self._path(state.portfolio_id)
        tmp_path = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp.json")
        try:
            tmp_path.write_text(
                json.dumps(asdict(state), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def apply_rebalance(
        self,
        portfolio_id: str,
        decision: RebalanceDecision,
    ) -> PortfolioState:
        """将已确认的调仓决策应用到持仓账本，返回更新后的 PortfolioState。

        语义（不可部分应用，保证原子性）：
        1. 防重复：last_signal_id == decision.signal_id → ValueError（调用方转 409）。
        2. 先全量校验：任一 sell 超过当前持仓 → 整笔拒绝，账本不变。
        3. 全量通过后才逐 item 应用：buy+=、sell-=；归零的 symbol 移除。
        4. 成功后更新 last_signal_id、updated_at，原子写文件。
        """
        state = self.load(portfolio_id)

        # 防重复确认
        if state.last_signal_id == decision.signal_id:
            raise ValueError(f"该调仓已确认过: {decision.signal_id}")

        # 第一遍：全量校验（不修改 state）
        for item in decision.items:
            if item.action == "sell":
                current = state.positions.get(item.vt_symbol, 0)
                if current < item.volume:
                    raise ValueError(
                        f"卖出 {item.vt_symbol} 超过当前持仓: "
                        f"请求卖出 {item.volume} 股，当前持仓 {current} 股"
                    )

        # 第二遍：应用（全量校验通过后才执行）
        new_positions: dict[str, int] = dict(state.positions)
        for item in decision.items:
            sym = item.vt_symbol
            if item.action == "buy":
                new_positions[sym] = new_positions.get(sym, 0) + item.volume
            elif item.action == "sell":
                new_positions[sym] = new_positions.get(sym, 0) - item.volume

        # 归零的 symbol 从 dict 移除
        new_positions = {k: v for k, v in new_positions.items() if v > 0}

        state.positions = new_positions
        state.last_signal_id = decision.signal_id
        state.updated_at = datetime.now().isoformat(timespec="seconds")
        self.save(state)
        return state
