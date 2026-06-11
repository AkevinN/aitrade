"""
风控明细薄包装（交易操作台粘合代码 1）。

`RiskManager.check_buy` 仅返回 `(ok, reason)`，缺逐项明细。`RiskInspector`
包住它：内部按 `RiskManager` 同序「重放」各检查项，逐项记录为
`{check, passed, detail}`，再委托真实 `RiskManager.check_buy` 得到权威
`(ok, reason)` 作为最终判定。

判定权威始终是 `RiskManager`，明细仅用于展示 / 审计，避免双实现漂移。
通过鸭子类型暴露与 `RiskManager` 一致的 `check_buy(...)` 与 `can_trade()`
签名，可直接注入 `SignalService`，无需改动 `RiskManager` 或 `SignalService`。
"""

from __future__ import annotations

from .risk import RiskManager


class RiskInspector:
    """包住 RiskManager，逐项记录风控检查明细。判定结果仍以 RiskManager 为准。"""

    def __init__(self, risk: RiskManager) -> None:
        self._risk = risk
        self.records: list[dict] = []  # [{check, passed, detail}]

    def can_trade(self) -> tuple[bool, str]:
        """透传，保持 SignalService 调用面一致。"""
        return self._risk.can_trade()

    def buy_capacity(
        self,
        *,
        vt_symbol: str,
        portfolio_value: float,
        current_total_position_value: float,
        current_symbol_value: float = 0.0,
        halted: bool = False,
    ) -> tuple[float, str]:
        """透传可买额度计算；明细仍由最终 `check_buy` 记录。"""
        return self._risk.buy_capacity(
            vt_symbol=vt_symbol,
            portfolio_value=portfolio_value,
            current_total_position_value=current_total_position_value,
            current_symbol_value=current_symbol_value,
            halted=halted,
        )

    def check_buy(
        self,
        *,
        vt_symbol: str,
        intended_value: float,
        portfolio_value: float,
        current_total_position_value: float,
        current_symbol_value: float = 0.0,
        halted: bool = False,
    ) -> tuple[bool, str]:
        """按 RiskManager 同序重放各检查项并记录明细，最终判定委托 RiskManager。"""
        cfg = self._risk.config
        rec = self.records.append  # 局部别名

        # 1) 全局闸门（kill-switch / 熔断）
        gate_ok, gate_reason = self._risk.can_trade()
        rec({
            "check": "kill_switch_or_circuit",
            "passed": gate_ok,
            "detail": gate_reason or "通过",
        })

        # 2) 黑名单
        bl_ok = vt_symbol not in cfg.blacklist
        rec({
            "check": "blacklist",
            "passed": bl_ok,
            "detail": "通过" if bl_ok else f"{vt_symbol} 在黑名单中",
        })

        # 3) 停牌 / 涨跌停封死
        halt_ok = (not halted) or cfg.allow_when_halted
        rec({
            "check": "halted",
            "passed": halt_ok,
            "detail": "通过" if halt_ok else f"{vt_symbol} 停牌/涨跌停封死，禁止买入",
        })

        # 4) 总仓位上限
        new_total = current_total_position_value + intended_value
        total_ok = new_total <= cfg.max_total_position_ratio * portfolio_value + 1e-6
        rec({
            "check": "max_total_position",
            "passed": total_ok,
            "detail": (
                f"拟新增后 {new_total:.0f} vs 上限 "
                f"{cfg.max_total_position_ratio:.0%}×{portfolio_value:.0f}"
            ),
        })

        # 5) 单票上限
        new_symbol = current_symbol_value + intended_value
        single_ok = new_symbol <= cfg.max_single_position_ratio * portfolio_value + 1e-6
        rec({
            "check": "max_single_position",
            "passed": single_ok,
            "detail": (
                f"{vt_symbol} 拟新增后 {new_symbol:.0f} vs 上限 "
                f"{cfg.max_single_position_ratio:.0%}×{portfolio_value:.0f}"
            ),
        })

        # 权威判定：委托真实 RiskManager（明细仅展示，避免双实现漂移）
        return self._risk.check_buy(
            vt_symbol=vt_symbol,
            intended_value=intended_value,
            portfolio_value=portfolio_value,
            current_total_position_value=current_total_position_value,
            current_symbol_value=current_symbol_value,
            halted=halted,
        )
