"""
CNN 专用回测策略 —— 单标的、由 CNN 概率信号驱动。

支持两种出场模式（出场口径必须与训练 label 的 exit 定义保持一致，
否则回测与实盘/label 会背离，见 docs/07）：

- ``threshold``（默认，向后兼容）：概率 > buy_threshold 开仓，概率 < sell_threshold 平仓。
- ``fixed_hold``：概率 > buy_threshold 开仓后，**固定持有 hold_days 个交易日后平仓**，
  与「持有到次日收盘」类固定持有期 label 对齐。出场不依赖信号，确定性强、便于核对。

说明：共享回测引擎采用「下一根 bar 成交」（T 下单、T+1 成交）。因此 fixed_hold 的
实际成交是「按下一根 bar 撮合」，与 label 的收盘价口径存在一根 bar 的执行近似；
入场/出场价精确口径（close vs next_open）由 LabelSpec.price_ref 在迭代 1 收敛。

path_class 七列信号（含 prob_sl）下可启用入场否决：``veto_threshold`` 控制否决阈值，
prob_sl >= veto_threshold 时拒绝新建仓，仅影响开仓判断，出场逻辑不受任何影响。
classification/regression 三列信号（无 prob_sl）时否决逻辑自动关闭，向后兼容。
"""

from datetime import date
from typing import Optional

import polars as pl

from ..backtest.strategy import BaseStrategy
from ..backtest.types import Direction, TradeData, BarData
from .thresholds import threshold_scale_check


class CNNSignalStrategy(BaseStrategy):
    """由 CNN 概率信号驱动的单标的策略。

    三种出场模式（threshold / fixed_hold / oco）均支持「入场否决」机制：
    path_class 信号含 prob_sl 列时，若 prob_sl >= veto_threshold 则拒绝新建仓（
    否决计数记入 _veto_count），仅影响入场判断，出场逻辑始终执行。
    classification/regression 信号无 prob_sl 列时否决恒为 False，完全向后兼容。
    """

    buy_threshold: float = 0.6     # 概率 > 该值买入
    sell_threshold: float = 0.4    # 概率 < 该值卖出（仅 threshold 模式使用）
    position_ratio: float = 0.95   # 最大投入组合资金比例
    min_volume: int = 100          # 最小交易单位（A股 100 股）
    price_add: float = 0.002       # 限价单价格缓冲（市价化挂单，默认 20bp）
    exit_mode: str = "threshold"   # 出场模式：threshold | fixed_hold | oco
    hold_days: int = 1             # fixed_hold/oco：固定/最大持有交易日数（oco 下为时间回退，0=不启用）
    take_profit: float = 0.0       # oco：止盈幅度（如 0.02=+2%），0=不启用
    stop_loss: float = 0.0         # oco：止损幅度（如 0.03=-3%），0=不启用
    veto_threshold: float = 1.0    # path_class：信号行 prob_sl >= 该值则否决买入；1.0=等效关闭（向后兼容）

    def on_init(self) -> None:
        """策略初始化回调 —— 重置内部持仓状态、按信号帧 objective 自检阈值、记录初始化日志。

        从 ``self.strategy_engine.signal_df`` 读首行 ``objective`` 列（缺列→None），
        调用 :func:`threshold_scale_check` 自检 buy/sell 阈值尺度。检出违规时设置
        ``self._threshold_invalid=True`` 并 write_log 告警；后续三个入场分支在买入前
        短路拒绝（出场逻辑零影响），构成 API 400 主拦截之外的防御纵深。
        缺列（legacy/规则策略信号帧）或阈值合法时 ``_threshold_invalid`` 恒为 False，零影响。
        """
        # 固定持有/OCO 出场的内部状态
        self._entry_fill_dt: Optional[date] = None   # 当前持仓的建仓成交日
        self._entry_price: Optional[float] = None    # 当前持仓的建仓成交价（OCO 止盈止损基准）
        self._hold_count: int = 0                    # 自建仓起已持有的交易日数
        self._last_count_dt: Optional[date] = None   # 上次计数的交易日（防同日重复计数）
        self._veto_count: int = 0                    # path_class 否决买入的累计次数（供外部任务读取）

        # 阈值尺度自检（防御纵深）：信号帧含 objective 列时按其口径校验 buy/sell 阈值。
        self._threshold_invalid: bool = False
        sig = getattr(self.strategy_engine, "signal_df", None)
        objective: Optional[str] = None
        if sig is not None and not sig.is_empty() and "objective" in sig.columns:
            objective = str(sig["objective"][0])
        reasons = threshold_scale_check(objective, self.buy_threshold, self.sell_threshold)
        if reasons:
            self._threshold_invalid = True
            self.write_log("阈值与模型 objective 不匹配，将拒绝开仓：" + "；".join(reasons))

        self.write_log(f"CNN 信号策略已初始化，出场模式={self.exit_mode}")

    def _entry_vetoed(self, signal: pl.DataFrame) -> bool:
        """入场否决检查：信号含 prob_sl 列且首行 prob_sl >= veto_threshold 时返回 True。

        **仅在「概率达标、即将下买单」之前调用**——三种出场模式（threshold/fixed_hold/oco）
        语义完全对齐：_veto_count 统计的是「本要买入却被否决的次数」，空仓期间概率不达标
        时不调用此方法，不产生无意义的否决计数与日志。

        classification/regression 信号无 prob_sl 列，恒返回 False（向后兼容）。
        触发时 write_log 记录否决事件（含引擎时刻与 prob_sl 值）并累计 _veto_count。
        仅在「空仓考虑入场」时由各入场分支调用（持仓时不调用、不计数）。

        Args:
            signal: get_signal() 返回的当前时刻信号 DataFrame（非空）。
                    调用方应已确认 prob > buy_threshold 后再调用本方法。

        Returns:
            True 表示触发否决（不应开新仓）；False 表示允许正常入场判断。

        Example:
            >>> prob = float(signal["signal"][0])
            >>> if prob > self.buy_threshold and not self._entry_vetoed(signal):
            ...     # 下买单
        """
        if "prob_sl" not in signal.columns:
            return False
        prob_sl = float(signal["prob_sl"][0])
        if prob_sl >= self.veto_threshold:
            self._veto_count += 1
            engine_dt = self.strategy_engine.datetime
            dt_str = f" {engine_dt}" if engine_dt is not None else ""
            self.write_log(
                f"否决买入:{dt_str} prob_sl={prob_sl:.3f}"
                f" >= veto_threshold={self.veto_threshold}"
            )
            return True
        return False

    def on_trade(self, trade: TradeData) -> None:
        """成交回报回调 —— 维护固定持有计数与建仓价。

        建仓成交时记录成交日期与价格并重置持有计数；
        平仓成交时清空全部状态，允许后续重新入场。

        Args:
            trade: 成交回报对象，含 direction/datetime/price 等字段。
        """
        trade_dt = trade.datetime.date() if trade.datetime else None
        if trade.direction == Direction.LONG:
            # 建仓成交：记录成交日/成交价并重置持有计数
            self._entry_fill_dt = trade_dt
            self._entry_price = float(trade.price)
            self._hold_count = 0
            self._last_count_dt = None
        else:
            # 平仓成交：清空持仓状态，允许后续重新入场
            self._entry_fill_dt = None
            self._entry_price = None
            self._hold_count = 0
            self._last_count_dt = None

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K 线切片回调 —— 按 exit_mode 分派到对应的交易逻辑。

        Args:
            bars: 当前时刻各证券的 BarData 字典（vt_symbol → BarData）。
        """
        if self.exit_mode == "fixed_hold":
            self._on_bars_fixed_hold(bars)
        elif self.exit_mode == "oco":
            self._on_bars_oco(bars)
        else:
            self._on_bars_threshold(bars)

    # ------------------------------------------------------------------
    # 阈值出场（默认，向后兼容）
    # ------------------------------------------------------------------
    def _on_bars_threshold(self, bars: dict[str, BarData]) -> None:
        """阈值出场模式（默认，向后兼容）的 K 线处理逻辑。

        概率 > buy_threshold 且空仓时买入；概率 < sell_threshold 且持仓时平仓。
        出场完全依赖 CNN 概率信号，realized 持有期不固定。

        Args:
            bars: 当前时刻各证券的 BarData 字典。
        """
        signal = self.get_signal()
        if signal.is_empty():
            return

        prob = float(signal["signal"][0])
        vt_symbol = str(signal["vt_symbol"][0])

        bar = bars.get(vt_symbol)
        if not bar or not bar.close_price:
            return

        current_pos = self.get_pos(vt_symbol)
        portfolio_value = self.get_portfolio_value()

        if (
            prob > self.buy_threshold
            and current_pos == 0
            and not self._threshold_invalid
            and not self._entry_vetoed(signal)
        ):
            volume = self._target_volume(portfolio_value, bar.close_price)
            if volume >= self.min_volume:
                self.set_target(vt_symbol, volume)
        elif prob < self.sell_threshold and current_pos > 0:
            self.set_target(vt_symbol, 0)

        self.execute_trading(bars, price_add=self.price_add)

    # ------------------------------------------------------------------
    # 固定持有出场（与固定持有期 label 对齐）
    # ------------------------------------------------------------------
    def _on_bars_fixed_hold(self, bars: dict[str, BarData]) -> None:
        """固定持有出场模式的 K 线处理逻辑，与固定持有期 label 对齐。

        出场优先：持有满 hold_days 个交易日后强制平仓（不依赖信号）。
        入场：空仓且无在途建仓时，概率达标才买入。
        每个交易日仅计数一次（防同日重复计数），确保 hold_days 为实际交易日数。

        Args:
            bars: 当前时刻各证券的 BarData 字典。
        """
        if not self.vt_symbols:
            return
        vt_symbol = self.vt_symbols[0]
        bar = bars.get(vt_symbol)
        if not bar or not bar.close_price:
            return

        engine_dt = self.strategy_engine.datetime
        today = engine_dt.date() if engine_dt else None
        current_pos = self.get_pos(vt_symbol)

        # 1) 出场优先：持有满 hold_days 个交易日则强制平仓，出场不看信号
        if current_pos > 0 and self._entry_fill_dt is not None and today is not None:
            if self._last_count_dt != today:
                self._hold_count += 1
                self._last_count_dt = today
            if self._hold_count >= self.hold_days:
                self.set_target(vt_symbol, 0)
                self.execute_trading(bars, price_add=self.price_add)
                return

        # 2) 入场：空仓且无在途建仓，概率达标、阈值合法且未被否决才买入
        if current_pos == 0 and self._entry_fill_dt is None and not self._threshold_invalid:
            signal = self.get_signal()
            if not signal.is_empty():
                prob = float(signal["signal"][0])
                if prob > self.buy_threshold and not self._entry_vetoed(signal):
                    volume = self._target_volume(self.get_portfolio_value(), bar.close_price)
                    if volume >= self.min_volume:
                        self.set_target(vt_symbol, volume)

        self.execute_trading(bars, price_add=self.price_add)

    # ------------------------------------------------------------------
    # OCO 出场（止盈 + 止损，路径依赖；保守假设止损先到）
    # ------------------------------------------------------------------
    def _on_bars_oco(self, bars: dict[str, BarData]) -> None:
        """OCO 出场模式（止盈 + 止损，路径依赖）的 K 线处理逻辑。

        持仓时，同一根 bar 内 high/low 先后未知时保守假设止损先触发：
        1. 止损触发（low <= sl_price）→ 按 sl_price 当根 bar 内盘中出场；
        2. 止盈触发（high >= tp_price）→ 按 tp_price 当根 bar 内盘中出场；
        3. 最大持有期回退（hold_count >= hold_days）→ 下一根 bar 成交出场。
        T+1 限制下，买入当日不触发止盈止损。

        空仓时：概率达标则建仓。

        Args:
            bars: 当前时刻各证券的 BarData 字典。
        """
        if not self.vt_symbols:
            return
        vt_symbol = self.vt_symbols[0]
        bar = bars.get(vt_symbol)
        if not bar or not bar.close_price:
            return

        engine = self.strategy_engine
        engine_dt = engine.datetime
        today = engine_dt.date() if engine_dt else None
        current_pos = self.get_pos(vt_symbol)

        # 1) 持仓中：止损/止盈（当根触发价成交）优先，其次最大持有期回退
        if current_pos > 0 and self._entry_price:
            if self._last_count_dt != today:
                self._hold_count += 1
                self._last_count_dt = today

            # T+1：当日买入不可当日卖出
            bought_today = self._entry_fill_dt == today
            can_sell = not (getattr(engine, "t_plus1", False) and bought_today)

            tp_price = self._entry_price * (1 + self.take_profit)
            sl_price = self._entry_price * (1 - self.stop_loss)

            if can_sell:
                # 同一根 bar 内 high/low 先后未知 → 保守假设止损先触发
                if self.stop_loss > 0 and bar.low_price <= sl_price:
                    engine.sell_to_close_intrabar(vt_symbol, sl_price, current_pos)
                    return
                if self.take_profit > 0 and bar.high_price >= tp_price:
                    engine.sell_to_close_intrabar(vt_symbol, tp_price, current_pos)
                    return
                # 最大持有期回退（与 fixed_hold 同口径，下一根成交）
                if self.hold_days > 0 and self._hold_count >= self.hold_days:
                    self.set_target(vt_symbol, 0)
                    self.execute_trading(bars, price_add=self.price_add)
            return

        # 2) 空仓：概率达标、阈值合法且未被否决才建仓
        if current_pos == 0 and self._entry_fill_dt is None and not self._threshold_invalid:
            signal = self.get_signal()
            if not signal.is_empty():
                prob = float(signal["signal"][0])
                if prob > self.buy_threshold and not self._entry_vetoed(signal):
                    volume = self._target_volume(self.get_portfolio_value(), bar.close_price)
                    if volume >= self.min_volume:
                        self.set_target(vt_symbol, volume)

        self.execute_trading(bars, price_add=self.price_add)

    # ------------------------------------------------------------------
    def _target_volume(self, portfolio_value: float, price: float) -> int:
        """按组合市值与 position_ratio 估算可买入股数（向下取整到 min_volume 整数倍）。

        买入金额 = portfolio_value * position_ratio，再按 price 折算股数并向下
        取整到 min_volume（A股 100 股）的整数倍，确保下单量合法。

        Args:
            portfolio_value: 当前组合总市值，单位元；非正值会得到 0 股。
            price: 标的当前价格，单位元/股。

        Returns:
            可买入股数（min_volume 的整数倍）；price <= 0 时返回 0（无法下单）。
        """
        if price <= 0:
            return 0
        buy_value = portfolio_value * self.position_ratio
        return int(buy_value / price / self.min_volume) * self.min_volume


# 注册到共享策略注册表，供 Scheme 配置层按名取用（迭代 3）
from ..backtest.registry import register_strategy  # noqa: E402

register_strategy("cnn_signal", CNNSignalStrategy)
