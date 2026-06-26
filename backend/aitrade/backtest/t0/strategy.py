"""半仓做 T 策略（HalfPositionT0Strategy）：引擎@1m 上的日内做 T。

每个交易日：开盘按 TickPolicy 挂 卖`O+sell_tick` / 买`O−buy_tick` 限价单，日内由引擎撮合，
收盘把仓位拉回半仓（intrabar 以收盘价结算）。撮合/成本/T+1 全部复用引擎，本类只决定
「何时挂什么单 / 何时回半仓」。

仓位口径：``base_weight``（默认 0.5）为半仓锚；``swing_frac``（默认 1.0）为做 T 一手占
半仓的比例（1.0=全半仓摆动）。开盘先把昨日遗留仓位拉回半仓（卖出的是昨仓，满足 T+1），
再挂当日 ±档单。

已知限制（受引擎符号级 T+1）：当日「买腿」成交后，当日收盘的「下调卖出」会被 T+1 拦截
（引擎按符号级锁定，不区分昨仓/今仓），该下调顺延到次日开盘回半仓时完成。详见 design 已知缺口。
"""

from __future__ import annotations

from datetime import date, time

from ..engine import round_to
from ..strategy import BaseStrategy
from ..types import BarData
from .tick_policy import DailyBar, DailyHistory, FixedTick, TickPolicy

_LOT = 100.0  # A 股最小交易单位（股）


def _round_lot(shares: float) -> float:
    """把股数四舍五入到 100 股整。"""
    return max(0.0, round(shares / _LOT) * _LOT)


class HalfPositionT0Strategy(BaseStrategy):
    """半仓做 T 策略。配置项经 setting 注入（见模块/类属性）。"""

    # —— 可配置项（setting 注入）——
    vt_symbol: str = ""
    tick_policy: TickPolicy = None        # type: ignore[assignment]
    swing_frac: float = 1.0
    base_weight: float = 0.5
    close_time: time = time(14, 57)       # 该时刻及之后的首根 bar 视为收盘，触发回半仓

    def on_init(self) -> None:
        """初始化日界与日线累积状态。"""
        if self.tick_policy is None:
            self.tick_policy = FixedTick()
        self._hist: DailyHistory = DailyHistory()
        self._cur_date: date | None = None
        self._o = self._h = self._l = self._c = 0.0
        self._did_close: bool = False

    def _vt(self) -> str:
        """返回本策略标的（缺省取 vt_symbols[0]）。"""
        return self.vt_symbol or self.vt_symbols[0]

    def _half_shares(self, price: float) -> float:
        """按当前总资产与给定价推算半仓股数（含已持仓市值，现金口径一致）。"""
        vt = self._vt()
        size = self.strategy_engine.sizes.get(vt, 1)
        pos = self.get_pos(vt)
        pv = self.get_cash_available() + pos * price * size
        return _round_lot(self.base_weight * pv / (price * size))

    def _rebalance_to_half(self, price: float) -> None:
        """以给定价把仓位 intrabar 拉回半仓：缺则买、余则卖（卖出受引擎 T+1 约束）。"""
        vt = self._vt()
        target = self._half_shares(price)
        diff = target - self.get_pos(vt)
        if diff >= _LOT:
            self.strategy_engine.buy_to_open_intrabar(vt, price, diff)
        elif diff <= -_LOT:
            self.strategy_engine.sell_to_close_intrabar(vt, price, -diff)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """日界检测 → 开盘回半仓+挂±档单；日内累积 OHLC；收盘回半仓。"""
        vt = self._vt()
        bar = bars.get(vt)
        if bar is None:
            return

        d = bar.datetime.date()
        if d != self._cur_date:
            # 收尾昨日：把累积的昨日 OHLC 落入历史（供 TickPolicy 无前视取用）
            if self._cur_date is not None:
                self._hist.append(DailyBar(self._cur_date, self._o, self._h, self._l, self._c))
            # 开新日
            self._cur_date = d
            self._o = bar.open_price
            self._h = bar.high_price
            self._l = bar.low_price
            self._c = bar.close_price
            self._did_close = False

            # 开盘：先清挂单 + 把昨日遗留仓位拉回半仓（卖的是昨仓，T+1 允许），再挂当日 ±档单
            self.cancel_all()
            self._rebalance_to_half(bar.open_price)
            self._place_band_orders(bar.open_price)
        else:
            # 日内：累积 OHLC
            self._h = max(self._h, bar.high_price)
            self._l = min(self._l, bar.low_price)
            self._c = bar.close_price
            # 收盘：该日首个 >= close_time 的 bar，撤未成交挂单 + 回半仓（intrabar 以收盘价）
            if not self._did_close and bar.datetime.time() >= self.close_time:
                self.cancel_all()
                self._rebalance_to_half(bar.close_price)
                self._did_close = True

    def _place_band_orders(self, open_price: float) -> None:
        """按 TickPolicy 在开盘价上下挂卖/买限价单，数量=swing_frac×半仓股数。"""
        vt = self._vt()
        pricetick = self.strategy_engine.priceticks[vt]
        sell_tick, buy_tick = self.tick_policy.ticks_for(self._cur_date, self._hist)
        vol = _round_lot(self.swing_frac * self._half_shares(open_price))
        if vol < _LOT:
            return  # 半仓股数过小（资金过少/价过高），当日不做 T，仅维持半仓
        sell_price = round_to(open_price + sell_tick, pricetick)
        buy_price = round_to(open_price - buy_tick, pricetick)
        # 卖腿数量不超过当前可卖持仓（避免裸卖）；买腿固定 vol
        sell_vol = min(vol, _round_lot(self.get_pos(vt)))
        if sell_vol >= _LOT:
            self.sell(vt, sell_price, sell_vol)
        self.buy(vt, buy_price, vol)

    def on_trade(self, trade) -> None:
        """成交回报（pos_data 由基类维护），此处无需额外处理。"""
        pass
