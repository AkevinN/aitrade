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
    """包住 RiskManager，逐项记录风控检查明细。判定结果仍以 RiskManager 为准。

    通过鸭子类型暴露与 RiskManager 一致的 can_trade / buy_capacity / check_buy 签名，
    可直接注入 SignalService。每次 check_buy 都会把各检查项明细累积进 records。

    Attributes:
        records: 检查明细列表，每项为 {"check": 检查项名, "passed": bool, "detail": 说明}。
            随 check_buy 调用持续追加，调用方需要时自行清空。
    """

    def __init__(self, risk: RiskManager) -> None:
        """初始化 RiskInspector。

        Args:
            risk: 被包装的 RiskManager 实例，承载权威判定逻辑与配置。
                  RiskInspector 不修改其状态，仅读取 `config` 属性并透传方法调用。
        """
        self._risk = risk
        self.records: list[dict] = []  # [{check, passed, detail}]

    def can_trade(self) -> tuple[bool, str]:
        """透传 RiskManager 的全局交易闸门判定，保持 SignalService 调用面一致。

        用于查询 kill-switch / 熔断等全局门是否放行，不针对单个标的。

        Returns:
            (ok, reason) 二元组：ok 为 True 表示当前允许交易，reason 为空串；
            ok 为 False 时 reason 为禁止原因（如熔断触发）。
        """
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
        """透传 RiskManager 的可买额度计算；本方法不记录明细（明细仅由 check_buy 产生）。

        在风控约束下，估算该标的当前还能买入多少市值。

        Args:
            vt_symbol: 合约代码，如 "510300.SSE"。
            portfolio_value: 组合总市值（现金 + 持仓），用作各比例上限的基数。
            current_total_position_value: 当前全部持仓市值，用于校验总仓位上限。
            current_symbol_value: 该标的当前持仓市值，用于校验单票上限，默认 0.0（空仓）。
            halted: 该标的是否停牌 / 涨跌停封死，默认 False。

        Returns:
            (capacity, reason) 二元组：capacity 为还可买入的市值（>= 0，受总仓位、
            单票上限等约束取最小值）；reason 为额度受何约束限制的说明文本。
        """
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
        """按 RiskManager 同序重放各风控检查项、逐项追加到 records，最终判定委托 RiskManager。

        本方法有副作用：每次调用都会向 self.records 追加 5 条 {check, passed, detail}
        明细（闸门、黑名单、停牌、总仓位上限、单票上限），供前端展示 / 审计。明细仅作展示，
        权威放行与否始终以末尾的 RiskManager.check_buy 返回值为准，避免双实现漂移。

        Args:
            vt_symbol: 拟买入的合约代码，如 "510300.SSE"。
            intended_value: 本次拟买入的市值（元，正数）。
            portfolio_value: 组合总市值（现金 + 持仓），用作各比例上限的基数。
            current_total_position_value: 当前全部持仓市值，用于校验总仓位上限。
            current_symbol_value: 该标的当前持仓市值，用于校验单票上限，默认 0.0（空仓）。
            halted: 该标的是否停牌 / 涨跌停封死，默认 False。

        Returns:
            (ok, reason) 二元组（来自 RiskManager.check_buy）：ok 为 True 表示放行、
            reason 为空串；ok 为 False 时 reason 为首个未通过项的拒绝原因。
        """
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
