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
from ..cnn.thresholds import threshold_scale_check
from .decision import Decision, DecisionStore
from .decision_instant import DecisionInstant, make_signal_id
from .notifier import Notifier
from .risk import RiskManager


@dataclass
class PortfolioSnapshot:
    """决策所需的组合快照。

    Attributes:
        portfolio_value:       组合总市值（现金 + 持仓，元）。
        total_position_value:  当前全部持仓市值（元），用于总仓位上限校验。
        current_position:      目标标的当前持仓股数（0 表示当前空仓）。
        current_symbol_value:  目标标的当前持仓市值（元），用于单票上限校验。
    """

    portfolio_value: float          # 组合总市值（现金+持仓）
    total_position_value: float = 0.0   # 当前总持仓市值
    current_position: int = 0       # 目标标的当前持仓股数
    current_symbol_value: float = 0.0   # 目标标的当前持仓市值


class SignalService:
    """把信号 + 风控 + 提醒 + 持久化 编排为一次"今日决策"。

    复用回测同口径的信号与阈值；经前置风控过滤；幂等（同 signal_id 不重复处理/提醒）；
    决策落盘可回溯。与回测的出场口径保持一致（should_exit 标志由调用方给出）。
    """

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
        """初始化 SignalService。

        Args:
            scheme_name:    方案名（参与 signal_id 与提醒标题）。
            buy_threshold:  买入信号阈值（概率 / 得分），超过即触发买入检查。
            risk:           风控对象（RiskManager 或鸭子类型兼容的 RiskInspector）。
            store:          DecisionStore，用于幂等查询与落盘。
            notifier:       通知器，买入/卖出决策时发送提醒。
            position_ratio: 目标仓位占组合市值的比例（0~1），默认 0.95。
            min_volume:     最小交易手数（股数），不足一手不买入，默认 100。
            model_version:  模型版本标签，参与 signal_id 生成（空串则不含版本）。
        """
        self.scheme_name = scheme_name
        self.buy_threshold = buy_threshold
        self.risk = risk
        self.store = store
        self.notifier = notifier
        self.position_ratio = position_ratio
        self.min_volume = min_volume
        self.model_version = model_version

    def _sized_volume(self, target_value: float, price: float) -> int:
        """将目标市值换算为向下取整到最小手数的股数。

        Args:
            target_value: 目标买入市值（元）。
            price:        当前价格（元/股）。

        Returns:
            按 min_volume 向下取整的股数；price <= 0 时返回 0。
        """
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
        trigger_source: str = "",
        objective: str | None = None,
    ) -> Decision:
        """产出该 Decision_Instant 的决策（幂等）。已存在则直接返回，不重复提醒。

        幂等键 signal_id 由 Decision_Bar（`decision_bar_dt` + `bar_freq`）决定：同一 bar
        的多次触发（如同日多个收盘后唤醒时刻）收敛为一次决策、一次提醒。

        Wave 2c：
        - trigger_source 写入 Decision 字段（落盘时一并持久化，无额外 save 调用）。
        - 实测 notifier.send 返回值存入 self.last_notify_ok（None=未发送/幂等命中）。
          调用方（orchestrator）可读取该属性以落盘通知实录（R5.1）。

        Args:
            instant: 决策时刻（含 as_of 与 bar_freq）；bar_freq 参与 signal_id 生成。
            decision_bar_dt: Decision_Bar 的收盘时刻，与 bar_freq 一起决定幂等键 signal_id。
            signal: 模型信号值（概率/得分），与 buy_threshold 比较决定是否触发买入。
            price: Decision_Bar 收盘价（元/股），用于仓位规模换算与提醒展示。
            portfolio: 组合快照（总市值/持仓股数/持仓市值），供风控与仓位规模计算。
            vt_symbol: 目标标的合约代码，如 "000001.SZSE"。
            should_exit: 是否到出场条件（由调用方依持有期/出场规则给出），默认 False；
                为 True 且当前有持仓时优先产出卖出决策。
            halted: 标的是否停牌/封死，默认 False；传入风控做买入放行判定。
            trigger_source: 触发来源标签（如 "manual"/"schedule"），落盘时写入 Decision；
                默认空串。
            objective: 信号帧自描述的模型输出口径（``"classification"`` /
                ``"regression"`` / ``"path_class"`` / None）。非 None 时按其口径用
                :func:`threshold_scale_check` 自检 buy_threshold；尺度不匹配（如回归
                模型套了 0.6 概率阈值）则该次决策标记拒绝（不产生买入），与回测
                端点共用同一规则（回测实盘一致红线）。None（legacy 信号帧）跳过校验。

        Returns:
            Decision 对象（action ∈ buy/sell/hold，含 volume/price/signal/reason）。
            幂等命中（同 signal_id 已落盘）时直接返回既有 Decision，不重新走风控、不提醒、
            不重复落盘（此时 self.last_notify_ok 保持 None）。
        """
        self.last_notify_ok: bool | None = None  # Wave 2c：实测通知结果

        signal_id = make_signal_id(
            decision_bar_dt, instant.bar_freq, self.scheme_name, self.model_version
        )
        existing = self.store.get(signal_id)
        if existing is not None:
            return existing

        decision = self._decide(
            signal_id, instant, decision_bar_dt, signal, price, portfolio,
            vt_symbol, should_exit, halted, objective,
        )
        decision.trigger_source = trigger_source  # Wave 2c：落盘前写入触发来源（一次 save）
        self.store.save(decision)
        if decision.action in ("buy", "sell"):
            self.last_notify_ok = self.notifier.send(
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
        objective: str | None = None,
    ) -> Decision:
        """内部决策逻辑（不含幂等/持久化/通知，由 run_for_instant 包装）。

        优先级：出场（should_exit + 有持仓）> 阈值尺度自检拒绝 > 入场（空仓 + 信号达标 +
        风控放行）> 持有观望。

        Args:
            signal_id:        本次决策的幂等键。
            instant:          决策时刻（含 bar_freq）。
            decision_bar_dt:  Decision_Bar 的收盘时刻。
            signal:           模型信号值（概率/得分）。
            price:            Decision_Bar 收盘价（元/股）。
            portfolio:        组合快照。
            vt_symbol:        目标标的。
            should_exit:      是否强制出场。
            halted:           标的是否停牌/封死。
            objective:        信号帧自描述的模型输出口径（None 跳过阈值尺度校验）。

        Returns:
            Decision 对象（action ∈ buy/sell/hold）。阈值尺度与 objective 不匹配时
            返回 hold（标记拒绝、不产生买入）。
        """
        base = dict(signal_id=signal_id, decision_bar_dt=decision_bar_dt.isoformat(),
                    as_of=instant.as_of.isoformat(), bar_freq=instant.bar_freq,
                    scheme=self.scheme_name, vt_symbol=vt_symbol, signal=signal, price=price)

        # 出场优先：持仓且到出场条件
        if should_exit and portfolio.current_position > 0:
            return Decision(action="sell", volume=portfolio.current_position,
                            reason="到出场条件（与回测出场口径一致）", **base)

        # 阈值尺度自检：信号帧自带 objective 时，按其口径校验 buy_threshold（与回测端点同规则）。
        # 不匹配（如回归模型套概率阈值）→ 该次决策标记拒绝，不产生买入决策（不下单）。
        threshold_reasons = threshold_scale_check(objective, self.buy_threshold)
        if threshold_reasons and portfolio.current_position == 0:
            return Decision(action="hold", volume=0,
                            reason="阈值与模型 objective 不匹配，拒绝买入：" + "；".join(threshold_reasons),
                            **base)

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
