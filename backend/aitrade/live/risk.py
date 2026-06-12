"""
风控（迭代 6 起，迭代 9 加强）：任何下单/建议前必过风控。

提供前置检查：黑名单、单票上限、总仓位上限、停牌/涨跌停封死过滤、
单日亏损熔断、人工 kill-switch。返回 (是否放行, 原因)，原因用于提醒与审计。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskConfig:
    """风控参数配置。

    Attributes:
        blacklist:                禁止买入的标的集合（vt_symbol，经 normalize 后存储）。
        max_total_position_ratio: 总持仓市值 / 组合总市值 上限（0~1），默认 0.95。
        max_single_position_ratio: 单票市值 / 组合总市值 上限（0~1），默认 0.30。
        daily_loss_limit:         单日亏损熔断阈值（相对组合市值比例），0 表示不启用熔断。
        allow_when_halted:        停牌/涨跌停封死标的是否允许买入，默认 False。
    """

    blacklist: set[str] = field(default_factory=set)
    max_total_position_ratio: float = 0.95   # 总持仓市值 / 组合市值 上限
    max_single_position_ratio: float = 0.30  # 单票市值 / 组合市值 上限
    daily_loss_limit: float = 0.05           # 单日亏损达组合该比例触发熔断（0=不启用）
    allow_when_halted: bool = False          # 停牌/涨跌停封死时是否允许交易


class RiskManager:
    """前置风控管理器。kill_switch 与 circuit_broken 为运行时状态（不持久化）。

    任何下单/建议前必须经此风控：`can_trade()` 检全局闸门，`check_buy()` 检买入细节。
    `RiskGuardedGateway` 在 send_order 前调用 `can_trade()`；`SignalService` 在买入决策前
    调用 `check_buy()` / `buy_capacity()`。

    Example:
        >>> mgr = RiskManager(RiskConfig(max_single_position_ratio=0.2))
        >>> mgr.can_trade()
        (True, '')
        >>> mgr.trip_kill_switch()
        >>> mgr.can_trade()
        (False, '人工 kill-switch 已触发，暂停所有交易')
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        """初始化 RiskManager。

        Args:
            config: 风控配置，None 时使用全默认参数。
        """
        self.config = config or RiskConfig()
        self.kill_switch: bool = False
        self.circuit_broken: bool = False

    def trip_kill_switch(self) -> None:
        """人工触发 kill-switch，立即暂停所有交易（全局闸门）。"""
        self.kill_switch = True

    def reset(self) -> None:
        """重置 kill-switch 与熔断标志（人工恢复用）。"""
        self.kill_switch = False
        self.circuit_broken = False

    def update_daily_pnl(self, today_pnl: float, capital: float) -> bool:
        """更新当日盈亏，触发熔断返回 True。"""
        if self.config.daily_loss_limit > 0 and capital > 0:
            if today_pnl <= -abs(self.config.daily_loss_limit) * capital:
                self.circuit_broken = True
        return self.circuit_broken

    def can_trade(self) -> tuple[bool, str]:
        """全局闸门：检查 kill-switch 与单日亏损熔断。

        Returns:
            (True, "") 表示可交易；(False, 原因) 表示被阻断。
        """
        if self.kill_switch:
            return False, "人工 kill-switch 已触发，暂停所有交易"
        if self.circuit_broken:
            return False, "单日亏损熔断已触发，暂停交易"
        return True, ""

    def check_buy(
        self,
        vt_symbol: str,
        intended_value: float,
        portfolio_value: float,
        current_total_position_value: float,
        current_symbol_value: float = 0.0,
        halted: bool = False,
    ) -> tuple[bool, str]:
        """买入前置检查（权威判定）。

        按序检查：全局闸门 → 黑名单 → 停牌/封死 → 组合市值非正 → 总仓位上限 → 单票上限。

        Args:
            vt_symbol:                    目标标的。
            intended_value:               拟买入市值（元），用于仓位上限校验。
            portfolio_value:              组合总市值（元）。
            current_total_position_value: 当前总持仓市值（元），不含本次买入。
            current_symbol_value:         该标的当前持仓市值（元），默认 0。
            halted:                       标的当日是否停牌/封死，默认 False。

        Returns:
            (True, "") 放行；(False, 原因文本) 拦截，原因供审计/提醒。
        """
        ok, reason = self.can_trade()
        if not ok:
            return False, reason
        if vt_symbol in self.config.blacklist:
            return False, f"{vt_symbol} 在黑名单中"
        if halted and not self.config.allow_when_halted:
            return False, f"{vt_symbol} 停牌/涨跌停封死，禁止买入"
        if portfolio_value <= 0:
            return False, "组合市值非正，禁止买入"
        # 总仓位上限
        new_total = current_total_position_value + intended_value
        if new_total > self.config.max_total_position_ratio * portfolio_value + 1e-6:
            return False, (
                f"超总仓位上限：拟新增后 {new_total:.0f} > "
                f"{self.config.max_total_position_ratio:.0%}×{portfolio_value:.0f}"
            )
        # 单票上限
        new_symbol = current_symbol_value + intended_value
        if new_symbol > self.config.max_single_position_ratio * portfolio_value + 1e-6:
            return False, (
                f"超单票上限：{vt_symbol} 拟新增后 {new_symbol:.0f} > "
                f"{self.config.max_single_position_ratio:.0%}×{portfolio_value:.0f}"
            )
        return True, ""

    def buy_capacity(
        self,
        *,
        vt_symbol: str,
        portfolio_value: float,
        current_total_position_value: float,
        current_symbol_value: float = 0.0,
        halted: bool = False,
    ) -> tuple[float, str]:
        """返回当前可新增买入金额上限。

        黑名单、停牌、kill-switch/熔断等硬风控返回 0；总仓位/单票仓位上限返回
        二者剩余额度的较小值。调用方仍需按手数向下取整，并最终再走 `check_buy`
        做权威校验。
        """
        ok, reason = self.can_trade()
        if not ok:
            return 0.0, reason
        if vt_symbol in self.config.blacklist:
            return 0.0, f"{vt_symbol} 在黑名单中"
        if halted and not self.config.allow_when_halted:
            return 0.0, f"{vt_symbol} 停牌/涨跌停封死，禁止买入"
        if portfolio_value <= 0:
            return 0.0, "组合市值非正，禁止买入"

        total_capacity = (
            self.config.max_total_position_ratio * portfolio_value
            - current_total_position_value
        )
        single_capacity = (
            self.config.max_single_position_ratio * portfolio_value
            - current_symbol_value
        )
        capacity = max(0.0, min(total_capacity, single_capacity))
        return capacity, ""
