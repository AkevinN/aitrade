"""
组合级风控管理器（Portfolio Risk Manager）。

负责两项组合层面的熔断/闸门控制：

1. **回撤熔断**（circuit breaker）：追踪组合净值峰值，一旦回撤超过 max_drawdown
   阈值，立即置 broken=True，后续所有 evaluate 调用的 allow_buy=False / buy_factor=0，
   直到人工调用 reset() 解除（reset 同时清零 peak，以新起点重新衡量）。
2. **趋势闸门**（trend gate）：读取基准指数（默认沪深300 ETF 日线）的最近 N 日
   均线，若当日收盘跌破均线则将 buy_factor 压缩为 below_ma_buy_factor（默认 0.5），
   趋势强则 buy_factor=1.0。

**状态持久化说明**：
- RuntimeStateStore 键 ``"portfolio_risk"`` → ``{portfolio_id: {...}}``
- 只在状态**真正变化**时（peak 更新或熔断触发）调用 set()，减少写放大。
- 已知债：RuntimeStateStore 整文件读改写无锁，并发写不同键会互相覆盖——
  本模块的写点将由编排器（任务 3.5）收敛为单点串行调用。

**卖出永远不受本风控阻止**：evaluate 的 allow_buy/buy_factor 只控制买入；
调用方若需放行卖出，直接忽略 verdict 即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .runtime_state import RuntimeStateStore

# RuntimeStateStore 的顶级键，值形态：{portfolio_id: {peak_value, broken, broken_date, reason}}
_PORTFOLIO_RISK_KEY = "portfolio_risk"


@dataclass
class PortfolioRiskConfig:
    """组合风控配置（可复用于多个组合，各组合各有独立运行时状态）。"""

    max_drawdown: float = 0.15
    """组合回撤熔断阈值（0.15 = 15%）。"""

    trend_ma_window: int = 60
    """趋势闸门均线窗口（交易日数）。"""

    benchmark_symbol: str = "510300.SSE"
    """趋势基准标的（默认沪深 300 ETF 日线）。"""

    below_ma_buy_factor: float = 0.5
    """基准跌破均线时买入额度系数（0.0 = 禁买）。"""


@dataclass
class PortfolioRiskVerdict:
    """evaluate 的输出：风控裁决。"""

    allow_buy: bool
    """熔断时为 False；趋势弱/正常时为 True（趋势仅影响 buy_factor，不禁止买入）。"""

    buy_factor: float
    """买入额度系数：1.0 = 正常；below_ma_buy_factor = 趋势弱；0.0 = 熔断。"""

    broken: bool
    """当前是否处于熔断状态。"""

    records: list[dict] = field(default_factory=list)
    """风控检查明细列表，每项 {check: str, passed: bool, detail: str}。"""


class PortfolioRiskManager:
    """组合级风控。

    状态经 RuntimeStateStore 持久化——每次 evaluate 都从 store 读取最新状态、
    状态变迁时写回，绝不依赖实例属性跨请求存活（live 模块已知陷阱：RiskManager
    每次决策重建；本类即使长生命周期复用，也能看到其他实例触发的熔断）。

    evaluate() 检查序列（每项产出一条 record）：
      1. circuit  — 读取已持久化的熔断状态
      2. drawdown — 更新 peak、计算当前回撤，触发熔断时写回 store
      3. trend    — 基准跌破均线时压缩 buy_factor；数据不足则 fail-open

    reset(portfolio_id) 清除 broken/broken_date/reason **且清零 peak**：
    熔断后人工处置完毕，以新起点衡量，避免残留旧高点导致新周期立即再熔断。
    """

    def __init__(
        self,
        state_store: RuntimeStateStore,
        config: PortfolioRiskConfig | None = None,
        *,
        lab: Any = None,
    ) -> None:
        """构造组合风控管理器。

        Args:
            state_store: 运行时状态存储，承载各组合的 peak/熔断状态并跨请求持久化。
            config: 组合风控配置；为 None 时使用 PortfolioRiskConfig 默认值
                （回撤阈值 15%、MA60 趋势闸门、沪深 300 ETF 基准）。
            lab: 行情数据访问对象（须提供 load_bar_frame）；为 None 时趋势闸门
                fail-open（始终判定趋势正常，不压缩 buy_factor）。
        """
        self._store = state_store
        self.config = config or PortfolioRiskConfig()
        self._lab = lab

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _load_state(self, portfolio_id: str) -> dict:
        """从 store 加载指定 portfolio 的风控状态；不存在时返回空 dict。"""
        all_states: dict = self._store.get(_PORTFOLIO_RISK_KEY, {}) or {}
        return dict(all_states.get(portfolio_id) or {})

    def _save_state(self, portfolio_id: str, pstate: dict) -> None:
        """将 portfolio 风控状态写回 store（整体覆盖同键）。"""
        all_states: dict = self._store.get(_PORTFOLIO_RISK_KEY, {}) or {}
        all_states = dict(all_states)
        all_states[portfolio_id] = pstate
        self._store.set(_PORTFOLIO_RISK_KEY, all_states)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        portfolio_id: str,
        *,
        portfolio_value: float,
        as_of: date,
    ) -> PortfolioRiskVerdict:
        """执行组合风控评估，按 circuit→drawdown→trend 顺序检查并返回裁决与明细。

        每次调用都从 store 读取该组合最新状态，仅在 peak 更新或熔断触发时写回。
        已熔断或本次触发熔断时短路返回（allow_buy=False、buy_factor=0.0）；
        正常时趋势仅影响 buy_factor 不影响 allow_buy。

        Args:
            portfolio_id: 组合 ID，对应 RuntimeStateStore 中 "portfolio_risk" 下的子键。
            portfolio_value: 当前组合净值，用于更新峰值与计算回撤。
            as_of: 评估日期，写入熔断记录并作为趋势查询的截止日。

        Returns:
            PortfolioRiskVerdict：含 allow_buy、buy_factor（1.0 正常 /
            below_ma_buy_factor 趋势弱 / 0.0 熔断）、broken 及 records 明细列表
            （每个检查产出一条 {check, passed, detail}）。
        """
        records: list[dict] = []
        pstate = self._load_state(portfolio_id)

        # ----------------------------------------------------------
        # 检查 1：熔断闸（circuit）
        # ----------------------------------------------------------
        already_broken = bool(pstate.get("broken", False))
        if already_broken:
            broken_date = pstate.get("broken_date", "未知日期")
            broken_reason = pstate.get("reason", "未知原因")
            records.append({
                "check": "circuit",
                "passed": False,
                "detail": f"组合已熔断（{broken_date}）：{broken_reason}，请人工复位后方可买入",
            })
            # 熔断时直接短路：不更新 peak、不查趋势，直接返回
            return PortfolioRiskVerdict(
                allow_buy=False,
                buy_factor=0.0,
                broken=True,
                records=records,
            )

        # 熔断闸通过
        records.append({
            "check": "circuit",
            "passed": True,
            "detail": "未熔断",
        })

        # ----------------------------------------------------------
        # 检查 2：回撤更新（drawdown）
        # ----------------------------------------------------------
        # peak 单调不减：首次取当前值
        old_peak = pstate.get("peak_value")
        if old_peak is None:
            new_peak = portfolio_value
        else:
            new_peak = max(float(old_peak), portfolio_value)

        # 计算当前回撤
        if new_peak > 0:
            dd = (new_peak - portfolio_value) / new_peak
        else:
            dd = 0.0

        state_changed = False

        if new_peak != old_peak:
            pstate["peak_value"] = new_peak
            state_changed = True

        now_broken = dd > self.config.max_drawdown
        if now_broken:
            reason = (
                f"当前回撤 {dd:.2%} 超过阈值 {self.config.max_drawdown:.2%}，"
                f"峰值 {new_peak:.2f}，当前值 {portfolio_value:.2f}"
            )
            pstate["broken"] = True
            pstate["broken_date"] = as_of.isoformat()
            pstate["reason"] = reason
            state_changed = True

            records.append({
                "check": "drawdown",
                "passed": False,
                "detail": reason,
            })

            if state_changed:
                self._save_state(portfolio_id, pstate)

            return PortfolioRiskVerdict(
                allow_buy=False,
                buy_factor=0.0,
                broken=True,
                records=records,
            )

        # 回撤在阈值以内
        records.append({
            "check": "drawdown",
            "passed": True,
            "detail": f"当前回撤 {dd:.2%}，阈值 {self.config.max_drawdown:.2%}，峰值 {new_peak:.2f}",
        })

        if state_changed:
            self._save_state(portfolio_id, pstate)

        # ----------------------------------------------------------
        # 检查 3：趋势闸门（trend）
        # ----------------------------------------------------------
        buy_factor = 1.0
        trend_passed, trend_detail = self._check_trend(as_of)
        if not trend_passed:
            buy_factor = self.config.below_ma_buy_factor

        records.append({
            "check": "trend",
            "passed": trend_passed,
            "detail": trend_detail,
        })

        return PortfolioRiskVerdict(
            allow_buy=True,
            buy_factor=buy_factor,
            broken=False,
            records=records,
        )

    def reset(self, portfolio_id: str) -> None:
        """人工复位：清除 broken/broken_date/reason，并清零 peak（以新起点衡量）。

        清零 peak 的理由：熔断后人工处置完毕，残留的旧高点会使新周期立即触发
        重新熔断（peak 依然是历史高位），因此与 broken 一并清除，让组合从当前
        净值重新起算。

        Args:
            portfolio_id: 待复位的组合 ID；不存在时静默无操作（幂等）。

        Returns:
            None。状态变更直接写回 RuntimeStateStore。
        """
        all_states: dict = self._store.get(_PORTFOLIO_RISK_KEY, {}) or {}
        all_states = dict(all_states)
        # 彻底清除该组合所有状态（peak 也清零）
        all_states.pop(portfolio_id, None)
        self._store.set(_PORTFOLIO_RISK_KEY, all_states)

    # ------------------------------------------------------------------
    # 趋势检查（内部）
    # ------------------------------------------------------------------

    def _check_trend(self, as_of: date) -> tuple[bool, str]:
        """检查基准指数趋势：当日收盘是否站上 MA{trend_ma_window}。

        从 lab 加载基准标的截至 as_of 的日线（向前多取 window*3 自然日以覆盖
        非交易日），取最近 window 根 close 求均线。任一前置条件不满足
        （lab 未注入 / 加载异常 / 数据不足）均 fail-open 判为趋势正常。

        Args:
            as_of: 趋势查询的截止日期（含）。

        Returns:
            (passed, detail) 二元组：passed=True 表示趋势强（close >= MA）或
            fail-open；passed=False 表示趋势弱（close < MA）。detail 为人类可读
            说明，写入 evaluate 的 records。
        """
        if self._lab is None:
            return True, "lab 未注入，趋势闸门跳过（fail-open）"

        cfg = self.config
        window = cfg.trend_ma_window
        symbol = cfg.benchmark_symbol

        try:
            # 取截至 as_of 的最近 window 根 close（从足够早的日期开始加载）
            from datetime import timedelta

            start = as_of - timedelta(days=window * 3)  # 留余量覆盖非交易日
            df = self._lab.load_bar_frame(symbol, "d", start, as_of)
        except Exception as exc:  # noqa: BLE001
            return True, f"基准数据加载异常（{exc!s}），趋势闸门跳过（fail-open）"

        if df is None or df.is_empty():
            return True, "基准数据不足，趋势闸门跳过（fail-open）"

        # 取最近 window 根 close
        closes = df["close"].to_list()
        if len(closes) < window:
            return (
                True,
                f"基准数据不足（{len(closes)} 根 < 窗口 {window}），趋势闸门跳过（fail-open）",
            )

        recent = closes[-window:]
        ma = sum(recent) / window
        latest_close = recent[-1]

        if latest_close < ma:
            return (
                False,
                f"基准 {symbol} 收盘 {latest_close:.4f} 跌破 MA{window} {ma:.4f}，趋势弱",
            )

        return (
            True,
            f"基准 {symbol} 收盘 {latest_close:.4f} >= MA{window} {ma:.4f}，趋势正常",
        )
