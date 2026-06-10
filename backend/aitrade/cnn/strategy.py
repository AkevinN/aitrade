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
"""

from datetime import date
from typing import Optional

from ..backtest.strategy import BaseStrategy
from ..backtest.types import Direction, TradeData, BarData


class CNNSignalStrategy(BaseStrategy):
    """由 CNN 概率信号驱动的单标的策略。"""

    buy_threshold: float = 0.6     # 概率 > 该值买入
    sell_threshold: float = 0.4    # 概率 < 该值卖出（仅 threshold 模式使用）
    position_ratio: float = 0.95   # 最大投入组合资金比例
    min_volume: int = 100          # 最小交易单位（A股 100 股）
    price_add: float = 0.002       # 限价单价格缓冲（市价化挂单，默认 20bp）
    exit_mode: str = "threshold"   # 出场模式：threshold | fixed_hold | oco
    hold_days: int = 1             # fixed_hold/oco：固定/最大持有交易日数（oco 下为时间回退，0=不启用）
    take_profit: float = 0.0       # oco：止盈幅度（如 0.02=+2%），0=不启用
    stop_loss: float = 0.0         # oco：止损幅度（如 0.03=-3%），0=不启用

    def on_init(self) -> None:
        """策略初始化回调"""
        # 固定持有/OCO 出场的内部状态
        self._entry_fill_dt: Optional[date] = None   # 当前持仓的建仓成交日
        self._entry_price: Optional[float] = None    # 当前持仓的建仓成交价（OCO 止盈止损基准）
        self._hold_count: int = 0                    # 自建仓起已持有的交易日数
        self._last_count_dt: Optional[date] = None   # 上次计数的交易日（防同日重复计数）
        self.write_log(f"CNN 信号策略已初始化，出场模式={self.exit_mode}")

    def on_trade(self, trade: TradeData) -> None:
        """成交回报回调 —— 维护固定持有计数与建仓价"""
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
        """K 线切片回调 —— 核心交易逻辑（按出场模式分派）"""
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

        if prob > self.buy_threshold and current_pos == 0:
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

        # 2) 入场：空仓且无在途建仓，概率达标才买入
        if current_pos == 0 and self._entry_fill_dt is None:
            signal = self.get_signal()
            if not signal.is_empty():
                prob = float(signal["signal"][0])
                if prob > self.buy_threshold:
                    volume = self._target_volume(self.get_portfolio_value(), bar.close_price)
                    if volume >= self.min_volume:
                        self.set_target(vt_symbol, volume)

        self.execute_trading(bars, price_add=self.price_add)

    # ------------------------------------------------------------------
    # OCO 出场（止盈 + 止损，路径依赖；保守假设止损先到）
    # ------------------------------------------------------------------
    def _on_bars_oco(self, bars: dict[str, BarData]) -> None:
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

        # 2) 空仓：概率达标建仓
        if current_pos == 0 and self._entry_fill_dt is None:
            signal = self.get_signal()
            if not signal.is_empty():
                prob = float(signal["signal"][0])
                if prob > self.buy_threshold:
                    volume = self._target_volume(self.get_portfolio_value(), bar.close_price)
                    if volume >= self.min_volume:
                        self.set_target(vt_symbol, volume)

        self.execute_trading(bars, price_add=self.price_add)

    # ------------------------------------------------------------------
    def _target_volume(self, portfolio_value: float, price: float) -> int:
        """按组合市值与 position_ratio 估算可买入手数（向下取整到 min_volume）"""
        if price <= 0:
            return 0
        buy_value = portfolio_value * self.position_ratio
        return int(buy_value / price / self.min_volume) * self.min_volume


# 注册到共享策略注册表，供 Scheme 配置层按名取用（迭代 3）
from ..backtest.registry import register_strategy  # noqa: E402

register_strategy("cnn_signal", CNNSignalStrategy)
