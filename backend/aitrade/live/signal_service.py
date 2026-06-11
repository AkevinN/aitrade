"""
半自动信号服务（迭代 6）：决策时点产出今日决策并提醒（不下单）。

复用回测同口径的信号与阈值；经前置风控过滤；幂等（同 signal_id 不重复处理/提醒）；
决策落盘可回溯。出场（卖出）由调用方依据持有期/出场规则给出 should_exit 标志，
与回测的 fixed_hold/auto 出场口径保持一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .decision import Decision, DecisionStore
from .decision_instant import DecisionInstant, make_signal_id
from .notifier import Notifier
from .risk import RiskManager


@dataclass
class PortfolioSnapshot:
    """决策所需的组合快照。"""
    portfolio_value: float          # 组合总市值（现金+持仓）
    total_position_value: float = 0.0   # 当前总持仓市值
    current_position: int = 0       # 目标标的当前持仓股数
    current_symbol_value: float = 0.0   # 目标标的当前持仓市值


class SignalService:
    """把信号 + 风控 + 提醒 + 持久化 编排为一次"今日决策"。"""

    def __init__(
        self,
        scheme_name: str,
        buy_threshold: float,
        risk: RiskManager,
        store: DecisionStore,
        notifier: Notifier,
        position_ratio: float = 0.95,
        min_volume: int = 100,
        model_version: str = "",
    ) -> None:
        self.scheme_name = scheme_name
        self.buy_threshold = buy_threshold
        self.risk = risk
        self.store = store
        self.notifier = notifier
        self.position_ratio = position_ratio
        self.min_volume = min_volume
        self.model_version = model_version

    def _sized_volume(self, target_value: float, price: float) -> int:
        if price <= 0:
            return 0
        return int(math.floor(target_value / price / self.min_volume)) * self.min_volume

    def run_for_instant(
        self,
        instant: DecisionInstant,
        *,
        decision_bar_dt: datetime,
        signal: float,
        price: float,
        portfolio: PortfolioSnapshot,
        vt_symbol: str,
        should_exit: bool = False,
        halted: bool = False,
    ) -> Decision:
        """产出该 Decision_Instant 的决策（幂等）。已存在则直接返回，不重复提醒。

        幂等键 signal_id 由 Decision_Bar（`decision_bar_dt` + `bar_freq`）决定：同一 bar
        的多次触发（如同日多个收盘后唤醒时刻）收敛为一次决策、一次提醒。
        """
        signal_id = make_signal_id(
            decision_bar_dt, instant.bar_freq, self.scheme_name, self.model_version
        )
        existing = self.store.get(signal_id)
        if existing is not None:
            return existing

        decision = self._decide(
            signal_id, instant, decision_bar_dt, signal, price, portfolio, vt_symbol, should_exit, halted
        )
        self.store.save(decision)
        if decision.action in ("buy", "sell"):
            self.notifier.send(
                title=f"[{self.scheme_name}] {('买入' if decision.action == 'buy' else '卖出')}信号",
                message=(
                    f"{decision.decision_bar_dt} {decision.vt_symbol} {decision.action} "
                    f"{decision.volume} 股 @≈{decision.price} | 概率={decision.signal} | {decision.reason}"
                ),
            )
        return decision

    def _decide(
        self,
        signal_id: str,
        instant: DecisionInstant,
        decision_bar_dt: datetime,
        signal: float,
        price: float,
        portfolio: PortfolioSnapshot,
        vt_symbol: str,
        should_exit: bool,
        halted: bool,
    ) -> Decision:
        base = dict(signal_id=signal_id, decision_bar_dt=decision_bar_dt.isoformat(),
                    as_of=instant.as_of.isoformat(), bar_freq=instant.bar_freq,
                    scheme=self.scheme_name, vt_symbol=vt_symbol, signal=signal, price=price)

        # 出场优先：持仓且到出场条件
        if should_exit and portfolio.current_position > 0:
            return Decision(action="sell", volume=portfolio.current_position,
                            reason="到出场条件（与回测出场口径一致）", **base)

        # 入场：空仓且概率达标
        if portfolio.current_position == 0 and signal > self.buy_threshold:
            target_value = portfolio.portfolio_value * self.position_ratio
            capacity, capacity_reason = self.risk.buy_capacity(
                vt_symbol=vt_symbol,
                portfolio_value=portfolio.portfolio_value,
                current_total_position_value=portfolio.total_position_value,
                current_symbol_value=portfolio.current_symbol_value,
                halted=halted,
            )
            if capacity <= 0:
                self.risk.check_buy(
                    vt_symbol=vt_symbol,
                    intended_value=0.0,
                    portfolio_value=portfolio.portfolio_value,
                    current_total_position_value=portfolio.total_position_value,
                    current_symbol_value=portfolio.current_symbol_value,
                    halted=halted,
                )
                return Decision(action="hold", volume=0,
                                reason=f"风控拦截：{capacity_reason or '无可用买入额度'}", **base)

            # 目标仓位先由策略给出，再按总仓位/单票仓位剩余额度裁剪，避免“超上限则一分不买”。
            clipped_value = min(target_value, capacity)
            volume = self._sized_volume(clipped_value, price)
            if volume < self.min_volume:
                return Decision(action="hold", volume=0,
                                reason="可用买入额度不足以买入最小手数", **base)
            intended_value = volume * price
            ok, reason = self.risk.check_buy(
                vt_symbol=vt_symbol,
                intended_value=intended_value,
                portfolio_value=portfolio.portfolio_value,
                current_total_position_value=portfolio.total_position_value,
                current_symbol_value=portfolio.current_symbol_value,
                halted=halted,
            )
            if not ok:
                return Decision(action="hold", volume=0, reason=f"风控拦截：{reason}", **base)
            return Decision(action="buy", volume=volume, reason="概率达标且通过风控", **base)

        # 其它情况：观望
        why = "概率未达买入阈值" if portfolio.current_position == 0 else "持有中，未到出场条件"
        return Decision(action="hold", volume=0, reason=why, **base)
