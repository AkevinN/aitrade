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
    """组合持仓快照（v1 不追踪现金），PositionBook 的持久化单元。

    Attributes:
        portfolio_id: 组合标识，对应一个账本 JSON 文件。
        positions: 当前持仓，vt_symbol → 股数；零持仓标的不在内（应用后会被移除）。
        cash: 可用现金（元）；v1 可不追踪，None 表示未启用现金跟踪。
        last_signal_id: 最近一次已确认调仓的幂等键，用于防重复确认；初始为空串。
        updated_at: 最近一次写盘时间（ISO8601，秒精度）；从未应用过调仓时为空串。
    """

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
        """初始化持仓账本：以 base_path 为账本文件根目录（不存在则创建）。

        Args:
            base_path: 账本文件根目录，不存在时自动创建。
                       每个 portfolio_id 对应一个 ``<safe_portfolio_id>.json`` 文件。
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, portfolio_id: str) -> Path:
        """将 portfolio_id 映射为账本文件路径（/ 与 : 替换为 _ 以保证文件名合法）。

        Args:
            portfolio_id: 组合标识，可含 / 或 :（会被规整为 _）。

        Returns:
            ``{base_path}/{safe_portfolio_id}.json`` 路径对象。
        """
        safe = portfolio_id.replace("/", "_").replace(":", "_")
        return self.base_path / f"{safe}.json"

    def load(self, portfolio_id: str) -> PortfolioState:
        """加载组合的持仓状态。

        Args:
            portfolio_id: 组合标识，对应一个账本 JSON 文件。

        Returns:
            反序列化得到的 PortfolioState；文件缺失时返回该 portfolio_id 的空账本
            （positions={}、last_signal_id=""、updated_at=""）。
        """
        path = self._path(portfolio_id)
        if not path.exists():
            return PortfolioState(portfolio_id=portfolio_id)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return PortfolioState(**raw)

    def save(self, state: PortfolioState) -> None:
        """将持仓状态原子写入账本文件（先写同目录 tmp 再 os.replace，崩溃安全）。

        写入路径由 state.portfolio_id 决定；同名文件存在时被整体替换。
        无论成功与否都会清理临时文件。

        Args:
            state: 待持久化的组合持仓状态。

        Returns:
            None。
        """
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

        Args:
            portfolio_id: 目标组合标识，先据此 load 当前账本。
            decision: 已确认的调仓决策（含幂等键 signal_id 与逐标的 buy/sell items）。

        Returns:
            应用调仓后的 PortfolioState（已写盘）；positions 已剔除归零标的，
            last_signal_id 更新为 decision.signal_id，updated_at 为当前时刻。

        Raises:
            ValueError: 该 signal_id 已确认过（重复确认），或任一 sell 超过当前持仓
                （此时账本保持不变，不做任何部分应用）。
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
