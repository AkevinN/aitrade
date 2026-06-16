"""回测引擎停牌无量不成交测试 + 涨跌停分级测试。

覆盖以下场景：
1. 正例：有量数据标的，在停牌日（fill_bar，volume=0）不应成交；
   订单应存活到下一根真实 bar 再成交。
2. 反例（豁免逻辑）：全程 volume=0 的数据源（如 parquet 缺 volume 列），
   不受门槛影响，订单照常成交。
3. OCO 正例：有量数据标的，在停牌日 OCO 腿不应被撮合。
4. infer_limit_ratio 纯函数表驱动用例。
5. 引擎层涨跌停分级行为用例（创业板 +15% 成交、转债 ratio=None 无限制、
   contract.json 配置 limit_ratio=0.05 的标的 +8% 封板拒单、OCO 跌停路径）。
"""

from __future__ import annotations

import logging
import pytest
from datetime import datetime

from aitrade.backtest.engine import BacktestingEngine
from aitrade.backtest.strategy import BaseStrategy
from aitrade.backtest.types import BarData, TradeData


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------

VT_SYMBOL = "TEST.SSE"
SYMBOL = "TEST"
EXCHANGE = "SSE"


class _MemoryLoader:
    """最小 BarDataLoader：内存行情，合约配置由外部直接写引擎属性。"""

    def __init__(self, bars: list[BarData]) -> None:
        self._bars = bars

    def load_bar_data(self, vt_symbol, interval, start, end) -> list[BarData]:
        return list(self._bars)

    def load_contract_settings(self) -> dict:
        return {}


def _setup_engine(bars: list[BarData]) -> BacktestingEngine:
    """构造引擎，直接设置合约参数（绕过 load_contract_settings）。"""
    loader = _MemoryLoader(bars)
    engine = BacktestingEngine(loader)
    start = bars[0].datetime
    end = bars[-1].datetime
    engine.set_parameters([VT_SYMBOL], "d", start, end, capital=1_000_000)
    engine.sizes[VT_SYMBOL] = 1
    engine.priceticks[VT_SYMBOL] = 0.01
    engine.long_rates[VT_SYMBOL] = 0.0
    engine.short_rates[VT_SYMBOL] = 0.0
    engine.stamp_duties[VT_SYMBOL] = 0.0
    engine.slippages[VT_SYMBOL] = 0.0
    return engine


def _make_bar(dt: datetime, price: float, volume: float) -> BarData:
    return BarData(
        symbol=SYMBOL,
        exchange=EXCHANGE,
        datetime=dt,
        interval="d",
        open_price=price,
        high_price=price + 1,
        low_price=price - 1,
        close_price=price,
        volume=volume,
    )


# ---------------------------------------------------------------------------
# 策略桩：D1 on_bars 时挂买单，之后不再操作
# ---------------------------------------------------------------------------

class _BuyOnFirstBarStrategy(BaseStrategy):
    """D1 收到第一根 bar 时按收盘价挂一手买单，之后不操作。"""

    def on_init(self) -> None:
        self._bought = False

    def on_bars(self, bars: dict[str, BarData]) -> None:
        if not self._bought and VT_SYMBOL in bars:
            bar = bars[VT_SYMBOL]
            self.buy(VT_SYMBOL, bar.close_price + 5, 100)
            self._bought = True

    def on_trade(self, trade: TradeData) -> None:
        pass


class _BuyOnFirstBarOCOStrategy(BaseStrategy):
    """D1 买入后在 D2 挂 OCO 止盈止损卖单。"""

    def on_init(self) -> None:
        self._state = "idle"

    def on_bars(self, bars: dict[str, BarData]) -> None:
        if self._state == "idle" and VT_SYMBOL in bars:
            bar = bars[VT_SYMBOL]
            self.buy(VT_SYMBOL, bar.close_price + 5, 100)
            self._state = "bought"
        elif self._state == "holding" and VT_SYMBOL in bars:
            bar = bars[VT_SYMBOL]
            # 止盈高于当前价，止损低于当前价；停牌日 open=high=low=close=前收，均不触发
            self.send_oco(VT_SYMBOL, bar.close_price + 10, bar.close_price - 10, 100)
            self._state = "oco_placed"

    def on_trade(self, trade: TradeData) -> None:
        if trade.direction == "long" and self._state == "bought":
            self._state = "holding"


# ---------------------------------------------------------------------------
# 正例：有量标的停牌日不成交
# ---------------------------------------------------------------------------

def test_suspension_bar_no_fill() -> None:
    """停牌日（fill_bar，volume=0）不应成交；订单须等到 D3 真实 bar 才成交。

    行情设计：
        D1  price=100, volume=10000  (正常 bar，策略挂买单)
        D2  该标的缺 bar → 引擎合成 fill_bar (volume=0, four_price=100)
        D3  price=101, volume=10000  (真实 bar，预期成交)
    """
    d1 = datetime(2024, 1, 1)
    d2 = datetime(2024, 1, 2)
    d3 = datetime(2024, 1, 3)

    # 夹具设计：dummy symbol 的 D2 bar 使 D2 进入全局时间轴；
    # 主 symbol D2 缺行 → 引擎合成 volume=0 的 fill_bar，触发停牌不成交门槛。
    dummy_symbol = "DUMMY.SSE"
    main_bars = [
        _make_bar(d1, 100.0, 10000),
        _make_bar(d3, 101.0, 10000),
    ]
    dummy_bars = [
        _make_bar(d1, 50.0, 5000),
        _make_bar(d2, 50.0, 5000),  # 此 bar 让 d2 进入 dts
        _make_bar(d3, 50.0, 5000),
    ]

    class _DualLoader:
        def load_bar_data(self, vt_symbol, interval, start, end) -> list[BarData]:
            if vt_symbol == VT_SYMBOL:
                return list(main_bars)
            return list(dummy_bars)

        def load_contract_settings(self) -> dict:
            return {}

    engine = BacktestingEngine(_DualLoader())
    engine.vt_symbols = [VT_SYMBOL, dummy_symbol]
    engine.interval = "d"
    engine.start = d1
    engine.end = d3
    engine.capital = 1_000_000
    engine.cash = 1_000_000
    engine.risk_free = 0
    engine.annual_days = 240

    for sym in [VT_SYMBOL, dummy_symbol]:
        engine.sizes[sym] = 1
        engine.priceticks[sym] = 0.01
        engine.long_rates[sym] = 0.0
        engine.short_rates[sym] = 0.0
        engine.stamp_duties[sym] = 0.0
        engine.slippages[sym] = 0.0

    import polars as pl
    engine.add_strategy(
        _BuyOnFirstBarStrategy,
        {},
        pl.DataFrame(),
    )
    engine.load_data()
    engine.run_backtesting()

    trades = engine.get_all_trades()
    # 预期：只有一笔成交，发生在 D3（主 symbol 恢复正常 bar）
    assert len(trades) == 1, f"预期 1 笔成交（D3），实际 {len(trades)} 笔"
    trade = trades[0]
    assert trade.datetime is not None
    assert trade.datetime.date() == d3.date(), (
        f"预期成交日 {d3.date()}，实际 {trade.datetime.date()}"
    )


# ---------------------------------------------------------------------------
# 反例：全程 volume=0 的数据源应豁免，订单照常成交
# ---------------------------------------------------------------------------

def test_no_volume_data_exempt_from_threshold() -> None:
    """全程 volume=0 的数据源（如缺 volume 列的 parquet）：
    不受停牌门槛影响，订单照常成交。
    """
    d1 = datetime(2024, 2, 1)
    d2 = datetime(2024, 2, 2)
    d3 = datetime(2024, 2, 3)

    # 三根 bar 全部 volume=0（模拟 parquet 缺 volume 列兜底 0.0）
    bars = [
        _make_bar(d1, 100.0, 0.0),
        _make_bar(d2, 101.0, 0.0),
        _make_bar(d3, 102.0, 0.0),
    ]
    engine = _setup_engine(bars)
    import polars as pl
    engine.add_strategy(_BuyOnFirstBarStrategy, {}, pl.DataFrame())
    engine.load_data()
    engine.run_backtesting()

    trades = engine.get_all_trades()
    # 豁免逻辑：该 symbol 从无 volume>0 bar，不应被拦截 → 至少 1 笔成交
    assert len(trades) >= 1, "全程 volume=0 的数据源应豁免门槛，订单应成交"


# ---------------------------------------------------------------------------
# OCO 正例：有量标的停牌日 OCO 腿不应被撮合
# ---------------------------------------------------------------------------

def test_suspension_bar_oco_no_fill() -> None:
    """停牌日 OCO 腿不应成交：主 symbol 停牌日缺 bar，引擎合成 fill_bar（volume=0）。

    行情设计：
        D1  price=100, volume=10000  (策略挂买单)
        D2  主 symbol 缺 bar（fill_bar，volume=0）  — OCO 腿已挂，不应触发
        D3  price=100（与 D1 等价，OCO 止盈/止损均不触发）
        D4  price=115, volume=10000  (真实 bar，止盈价=110 → OCO 止盈腿触发)
    """
    dummy_symbol = "DUMMY2.SSE"
    d1 = datetime(2024, 3, 1)
    d2 = datetime(2024, 3, 2)
    d3 = datetime(2024, 3, 3)
    d4 = datetime(2024, 3, 4)

    main_bars = [
        _make_bar(d1, 100.0, 10000),
        # D2 缺失 → 引擎合成 fill_bar
        _make_bar(d3, 100.0, 10000),
        _make_bar(d4, 115.0, 10000),
    ]
    dummy_bars = [
        _make_bar(d1, 50.0, 5000),
        _make_bar(d2, 50.0, 5000),
        _make_bar(d3, 50.0, 5000),
        _make_bar(d4, 50.0, 5000),
    ]

    class _DualLoader2:
        def load_bar_data(self, vt_symbol, interval, start, end) -> list[BarData]:
            if vt_symbol == VT_SYMBOL:
                return list(main_bars)
            return list(dummy_bars)

        def load_contract_settings(self) -> dict:
            return {}

    engine = BacktestingEngine(_DualLoader2())
    engine.vt_symbols = [VT_SYMBOL, dummy_symbol]
    engine.interval = "d"
    engine.start = d1
    engine.end = d4
    engine.capital = 1_000_000
    engine.cash = 1_000_000
    engine.risk_free = 0
    engine.annual_days = 240

    for sym in [VT_SYMBOL, dummy_symbol]:
        engine.sizes[sym] = 1
        engine.priceticks[sym] = 0.01
        engine.long_rates[sym] = 0.0
        engine.short_rates[sym] = 0.0
        engine.stamp_duties[sym] = 0.0
        engine.slippages[sym] = 0.0

    import polars as pl
    engine.add_strategy(_BuyOnFirstBarOCOStrategy, {}, pl.DataFrame())
    engine.load_data()
    engine.run_backtesting()

    trades = engine.get_all_trades()
    # D1 挂买单，D2 fill_bar 停牌不成交，D3 真实 bar 买入成交并挂 OCO，D4 止盈触发。
    # 总计 2 笔：D3 买入 + D4 OCO 止盈
    assert len(trades) == 2, f"预期 2 笔成交（D3 买入 + D4 OCO 止盈），实际 {len(trades)} 笔"
    trade_dates = sorted(t.datetime.date() for t in trades)
    assert trade_dates[0] == d3.date(), f"第一笔应在 D3，实际 {trade_dates[0]}"
    assert trade_dates[1] == d4.date(), f"第二笔应在 D4，实际 {trade_dates[1]}"


# ---------------------------------------------------------------------------
# 真实 volume=0 bar（非合成 fill_bar）：停牌日不成交
# ---------------------------------------------------------------------------

def test_real_volume_zero_bar_no_fill() -> None:
    """数据源提供真实 volume=0 的 bar（A 股部分停牌日形态）：中间 bar 不应成交，
    订单应存活到第三根正常 bar 才成交。

    行情序列：[vol=10000, vol=0, vol=10000]
    """
    d1 = datetime(2024, 4, 1)
    d2 = datetime(2024, 4, 2)
    d3 = datetime(2024, 4, 3)

    bars = [
        _make_bar(d1, 100.0, 10000),  # D1 正常，策略挂买单
        _make_bar(d2, 100.0, 0),       # D2 停牌（真实 volume=0 bar）
        _make_bar(d3, 101.0, 10000),  # D3 恢复正常
    ]
    engine = _setup_engine(bars)
    import polars as pl
    engine.add_strategy(_BuyOnFirstBarStrategy, {}, pl.DataFrame())
    engine.load_data()
    engine.run_backtesting()

    trades = engine.get_all_trades()
    # D2 是真实 volume=0 bar，属于 volume_supported 标的的停牌日，不应成交
    # D3 恢复有量，买单在 D3 成交
    assert len(trades) == 1, f"预期 1 笔成交（D3），实际 {len(trades)} 笔"
    assert trades[0].datetime.date() == d3.date(), (
        f"预期成交日 {d3.date()}，实际 {trades[0].datetime.date()}"
    )


# ---------------------------------------------------------------------------
# 豁免告警：全程 volume=0 的标的应触发 warning 日志
# ---------------------------------------------------------------------------

def test_no_volume_data_emits_warning() -> None:
    """全程 volume=0 的数据源加载后，load_data 应输出 WARNING 级别日志，
    提示该标的已被豁免停牌不成交门槛。
    """
    d1 = datetime(2024, 5, 1)
    d2 = datetime(2024, 5, 2)

    bars = [
        _make_bar(d1, 100.0, 0.0),
        _make_bar(d2, 101.0, 0.0),
    ]
    engine = _setup_engine(bars)

    with _CapturingHandler("aitrade.backtest.engine") as handler:
        engine.load_data()

    warnings = [r for r in handler.records if r.levelno == logging.WARNING]
    assert warnings, "全程无量标的应触发 WARNING 日志提示豁免"
    assert VT_SYMBOL in warnings[0].getMessage(), (
        f"告警应包含标的名 {VT_SYMBOL}，实际：{warnings[0].getMessage()}"
    )


class _CapturingHandler(logging.Handler):
    """上下文管理器：临时捕获指定 logger 的日志记录。"""

    def __init__(self, logger_name: str) -> None:
        super().__init__()
        self._logger_name = logger_name
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def __enter__(self) -> _CapturingHandler:
        logging.getLogger(self._logger_name).addHandler(self)
        return self

    def __exit__(self, *_) -> None:
        logging.getLogger(self._logger_name).removeHandler(self)


# =============================================================================
# Task 0.2: infer_limit_ratio 纯函数表驱动测试
# =============================================================================

@pytest.mark.parametrize("vt_symbol,expected", [
    # 创业板：300/301 开头 → 0.2
    ("300001.SZSE", 0.2),
    ("300999.SZSE", 0.2),
    ("301001.SZSE", 0.2),
    # 科创板：688/689 开头 → 0.2
    ("688001.SSE",  0.2),
    ("689001.SSE",  0.2),
    # 北交所：.BSE 结尾 → 0.3
    ("430001.BSE",  0.3),
    ("899999.BSE",  0.3),
    # 沪市可转债：110/111/113/118 开头 .SSE → None
    ("110001.SSE",  None),
    ("111001.SSE",  None),
    ("113001.SSE",  None),
    ("118001.SSE",  None),
    # 深市可转债：123/127/128 开头 .SZSE → None
    ("123001.SZSE", None),
    ("127001.SZSE", None),
    ("128001.SZSE", None),
    # 默认主板/ETF：0.1
    ("600000.SSE",  0.1),
    ("000001.SZSE", 0.1),
    ("510300.SSE",  0.1),  # ETF 走默认
])
def test_infer_limit_ratio(vt_symbol: str, expected: float | None) -> None:
    """infer_limit_ratio 纯函数：按品种前缀/交易所推断涨跌停比例。"""
    from aitrade.backtest.instrument import infer_limit_ratio
    result = infer_limit_ratio(vt_symbol)
    assert result == expected, (
        f"infer_limit_ratio({vt_symbol!r}) = {result}，预期 {expected}"
    )


# =============================================================================
# Task 0.2: 引擎层涨跌停分级行为测试
# =============================================================================

def _setup_engine_with_ratio(
    bars: list[BarData],
    vt_symbol: str,
    limit_ratio: float | None = None,
) -> BacktestingEngine:
    """构造引擎，可选设置 limit_ratios（None 意味着依靠 infer 推断）。"""
    symbol, exchange = vt_symbol.rsplit(".", 1)

    class _SingleLoader:
        def load_bar_data(self, sym, interval, start, end) -> list[BarData]:
            return list(bars)
        def load_contract_settings(self) -> dict:
            return {}

    engine = BacktestingEngine(_SingleLoader())
    start = bars[0].datetime
    end = bars[-1].datetime
    engine.set_parameters([vt_symbol], "d", start, end, capital=10_000_000)
    engine.sizes[vt_symbol] = 1
    engine.priceticks[vt_symbol] = 0.01
    engine.long_rates[vt_symbol] = 0.0
    engine.short_rates[vt_symbol] = 0.0
    engine.stamp_duties[vt_symbol] = 0.0
    engine.slippages[vt_symbol] = 0.0
    if limit_ratio is not None:
        engine.limit_ratios[vt_symbol] = limit_ratio
    return engine


def _make_bar_ex(
    vt_symbol: str,
    dt: datetime,
    pre_close: float,
    ratio: float,          # 本根相对前收的涨幅（如 0.15 = +15%）
    volume: float = 10000,
) -> BarData:
    """构造一根 bar：四价围绕 pre_close * (1 + ratio)。"""
    symbol, exchange = vt_symbol.rsplit(".", 1)
    price = round(pre_close * (1 + ratio), 2)
    return BarData(
        symbol=symbol,
        exchange=exchange,
        datetime=dt,
        interval="d",
        open_price=price,
        high_price=price + 0.5,
        low_price=price - 0.5,
        close_price=price,
        volume=volume,
    )


class _BuyOnFirstBarStrategy2(BaseStrategy):
    """D1 挂一个远高于当前价的买单（模拟市价单），D2 撮合时测试封板逻辑。"""

    def on_init(self) -> None:
        self._bought = False

    def on_bars(self, bars: dict[str, BarData]) -> None:
        sym = next(iter(bars))
        if not self._bought:
            bar = bars[sym]
            # 远高于当前价的限价单：D2 的任何 low_price 都低于该价，限价条件满足
            # 只有封板（low_price >= limit_up）才会拒单
            self.buy(sym, bar.close_price * 2, 100)
            self._bought = True

    def on_trade(self, trade: TradeData) -> None:
        pass


class _SellOnSecondBarStrategy(BaseStrategy):
    """D1 买入后 D2 挂高价卖单（测试 OCO 跌停拒单时用普通卖单替代）。"""

    def on_init(self) -> None:
        self._state = 0

    def on_bars(self, bars: dict[str, BarData]) -> None:
        sym = next(iter(bars))
        bar = bars[sym]
        if self._state == 0:
            # D1 按收盘价买入
            self.buy(sym, bar.close_price + 5, 100)
            self._state = 1

    def on_trade(self, trade: TradeData) -> None:
        if self._state == 1:
            self._state = 2


def test_gem_plus15_should_cross() -> None:
    """创业板（300001.SZSE）±20% 限制下，D2(+15%) 的 bar 应可成交（未封板）。

    旧逻辑按 ±10% → D2 的 low_price=114.5 超过旧 limit_up=110 → 封板拒单。
    新逻辑 limit_up=120，+15% 在限制内，D1 挂的买单应在 D2 成交。
    """
    vt_symbol = "300001.SZSE"
    d1 = datetime(2024, 6, 1)
    d2 = datetime(2024, 6, 2)

    bars = [
        BarData(
            symbol="300001", exchange="SZSE", datetime=d1, interval="d",
            open_price=100.0, high_price=101.0, low_price=99.0, close_price=100.0,
            volume=10000,
        ),
        # D2: +15%，low_price=114.5，应低于 limit_up=120（±20%），可成交
        BarData(
            symbol="300001", exchange="SZSE", datetime=d2, interval="d",
            open_price=115.0, high_price=115.5, low_price=114.5, close_price=115.0,
            volume=10000,
        ),
    ]

    engine = _setup_engine_with_ratio(bars, vt_symbol)

    import polars as pl
    engine.add_strategy(_BuyOnFirstBarStrategy2, {}, pl.DataFrame())
    engine.load_data()
    engine.run_backtesting()

    trades = engine.get_all_trades()
    assert len(trades) >= 1, (
        f"创业板 +15% 的 bar 应成交（±20% 限制），实际成交 {len(trades)} 笔"
    )


def test_convertible_bond_plus25_should_cross() -> None:
    """沪市可转债（110001.SSE）无涨跌停（ratio=None），D2(+25%) 的 bar 应可成交。

    旧逻辑 limit_up=110，+25% 的 bar.low_price=124.5 > 110 → 封板拒单。
    新逻辑 ratio=None → limit_up=inf → 永远允许，D1 挂的买单应在 D2 成交。
    """
    vt_symbol = "110001.SSE"
    d1 = datetime(2024, 6, 1)
    d2 = datetime(2024, 6, 2)

    bars = [
        BarData(
            symbol="110001", exchange="SSE", datetime=d1, interval="d",
            open_price=100.0, high_price=101.0, low_price=99.0, close_price=100.0,
            volume=10000,
        ),
        # D2: +25%，ratio=None → 无封板，应成交
        BarData(
            symbol="110001", exchange="SSE", datetime=d2, interval="d",
            open_price=125.0, high_price=125.5, low_price=124.5, close_price=125.0,
            volume=10000,
        ),
    ]

    engine = _setup_engine_with_ratio(bars, vt_symbol)

    import polars as pl
    engine.add_strategy(_BuyOnFirstBarStrategy2, {}, pl.DataFrame())
    engine.load_data()
    engine.run_backtesting()

    trades = engine.get_all_trades()
    assert len(trades) >= 1, (
        f"可转债（ratio=None）+25% 的 bar 应成交，实际成交 {len(trades)} 笔"
    )


def test_custom_limit_ratio_blocks_order() -> None:
    """contract.json 配 limit_ratio=0.05 的标的：D2(+8%) 应被封板拒单（不成交）。

    D1 挂超高价买单，D2: +8%，low_price=107.5 > limit_up(=pre_close*1.05=105)
    → long_cross 不满足，封板拒单；D3 价格仍超过 5% 故同样被拒单。
    全程无成交，断言 D2 无成交。
    """
    vt_symbol = "600000.SSE"
    d1 = datetime(2024, 6, 1)
    d2 = datetime(2024, 6, 2)
    d3 = datetime(2024, 6, 3)

    bars = [
        BarData(
            symbol="600000", exchange="SSE", datetime=d1, interval="d",
            open_price=100.0, high_price=101.0, low_price=99.0, close_price=100.0,
            volume=10000,
        ),
        # D2: +8%，bar.low_price=107.5 > limit_up=105（5%）→ 封板，拒单不成交
        BarData(
            symbol="600000", exchange="SSE", datetime=d2, interval="d",
            open_price=108.0, high_price=108.5, low_price=107.5, close_price=108.0,
            volume=10000,
        ),
        # D3: +6%，同样超过 5%（pre_close 已更新为 108，limit_up=108*1.05≈113.4），
        # low_price=105.5 < 113.4 → D3 会成交（此处验证关键：D2 无成交即可）
        BarData(
            symbol="600000", exchange="SSE", datetime=d3, interval="d",
            open_price=106.0, high_price=106.5, low_price=105.5, close_price=106.0,
            volume=10000,
        ),
    ]

    # 注入 limit_ratio=0.05（模拟 contract.json 配置覆盖）
    engine = _setup_engine_with_ratio(bars, vt_symbol, limit_ratio=0.05)

    import polars as pl
    engine.add_strategy(_BuyOnFirstBarStrategy2, {}, pl.DataFrame())
    engine.load_data()
    engine.run_backtesting()

    trades = engine.get_all_trades()
    # 关键断言：D2 无成交（被封板拒单）
    trade_dates = [t.datetime.date() for t in trades if t.datetime is not None]
    assert d2.date() not in trade_dates, (
        f"limit_ratio=0.05 标的：D2(+8%) 不应成交，实际成交日期：{trade_dates}"
    )


def test_oco_down_limit_uses_inferred_ratio() -> None:
    """OCO 跌停路径使用品种化 ratio：创业板标的（300001.SZSE）跌停价 = pre_close * 0.8。

    行情设计：
        D1  price=100，策略买入
        D2  price=100，买单成交；策略挂 OCO（止盈 200，止损 83）
        D3  -18%（82.0），high=82，low=81
            旧逻辑 limit_down = 0.9*100=90，high_price(82) ≤ 90 → OCO 被跌停封板拒掉；
            新逻辑 limit_down = 0.8*100=80，high_price(82) > 80 → OCO 止损触发。
    注：D3 pre_close 取 D2 收盘价（100）→ 跌停阈值 = 100*0.8=80。
    """
    vt_symbol = "300001.SZSE"
    d1 = datetime(2024, 7, 1)
    d2 = datetime(2024, 7, 2)
    d3 = datetime(2024, 7, 3)
    tp_price = 200.0   # 止盈：远高于市价，不会触发
    sl_price = 83.0    # 止损：在 D3 的 low_price(81) 以上，会触发

    bars = [
        # D1: price=100，策略 on_bars 买入
        BarData(
            symbol="300001", exchange="SZSE", datetime=d1, interval="d",
            open_price=100.0, high_price=101.0, low_price=99.0, close_price=100.0,
            volume=10000,
        ),
        # D2: price=100，买单在 cross_order 成交，on_trade→holding，on_bars 挂 OCO
        BarData(
            symbol="300001", exchange="SZSE", datetime=d2, interval="d",
            open_price=100.0, high_price=101.0, low_price=99.0, close_price=100.0,
            volume=10000,
        ),
        # D3: -18%（82.0），high=82，low=81
        # 新 limit_down = 100 * (1-0.2) = 80，high(82) > 80 → OCO 止损触发
        BarData(
            symbol="300001", exchange="SZSE", datetime=d3, interval="d",
            open_price=82.0, high_price=82.0, low_price=81.0, close_price=82.0,
            volume=10000,
        ),
    ]

    class _BuyThenOCOStrategy(BaseStrategy):
        """D1 买入，D2 on_bars 挂 OCO 止损单，D3 触发。"""
        def on_init(self) -> None:
            self._state = "idle"

        def on_bars(self, bars: dict[str, BarData]) -> None:
            sym = next(iter(bars))
            bar = bars[sym]
            if self._state == "idle":
                self.buy(sym, bar.close_price + 5, 100)
                self._state = "buying"
            elif self._state == "holding":
                self.send_oco(sym, tp_price, sl_price, 100)
                self._state = "oco_placed"

        def on_trade(self, trade: TradeData) -> None:
            if self._state == "buying" and trade.direction == "long":
                self._state = "holding"

    engine = _setup_engine_with_ratio(bars, vt_symbol)

    import polars as pl
    engine.add_strategy(_BuyThenOCOStrategy, {}, pl.DataFrame())
    engine.load_data()
    engine.run_backtesting()

    trades = engine.get_all_trades()
    # 应有 2 笔：D2 买入成交 + D3 OCO 止损触发
    assert len(trades) == 2, (
        f"创业板 OCO 跌停路径：预期 2 笔（D2 买入 + D3 止损），实际 {len(trades)} 笔"
    )
    trade_dates = sorted(t.datetime.date() for t in trades)
    assert trade_dates[1] == d3.date(), (
        f"OCO 止损应在 D3 触发，实际 {trade_dates[1]}"
    )


# =============================================================================
# Task 0.3: T+1 豁免标的 + sell_to_close_intrabar 绕过修复
# =============================================================================

EXEMPT_SYMBOL = "113001.SSE"   # 沪市可转债：t_plus1=false 豁免标的
NORMAL_SYMBOL = "600001.SSE"   # 普通 A 股：受 T+1 约束


class _NoOpStrategy(BaseStrategy):
    """什么都不做的策略桩，供直接测试引擎内部方法用。"""

    def on_init(self) -> None:
        pass

    def on_bars(self, bars: dict[str, BarData]) -> None:
        pass

    def on_trade(self, trade: TradeData) -> None:
        pass


def _setup_t1_engine(
    bars: list[BarData],
    vt_symbol: str,
    *,
    t_plus1: bool = True,
    exempt: bool = False,
) -> BacktestingEngine:
    """构造开启 T+1 的引擎，可选将 vt_symbol 加入豁免集合。

    挂载 _NoOpStrategy，使 cross_order / sell_to_close_intrabar 可以调用策略回调。
    """
    symbol, exchange = vt_symbol.rsplit(".", 1)

    class _Loader:
        def load_bar_data(self, sym, interval, start, end) -> list[BarData]:
            return list(bars)
        def load_contract_settings(self) -> dict:
            return {}

    import polars as pl
    engine = BacktestingEngine(_Loader())
    start = bars[0].datetime
    end = bars[-1].datetime
    engine.set_parameters([vt_symbol], "d", start, end, capital=10_000_000)
    engine.sizes[vt_symbol] = 1
    engine.priceticks[vt_symbol] = 0.01
    engine.long_rates[vt_symbol] = 0.0
    engine.short_rates[vt_symbol] = 0.0
    engine.stamp_duties[vt_symbol] = 0.0
    engine.slippages[vt_symbol] = 0.0
    engine.t_plus1 = t_plus1
    if exempt:
        engine.t_plus1_exempt.add(vt_symbol)
    # 挂载无操作策略，保证引擎内部回调不报 AttributeError
    engine.add_strategy(_NoOpStrategy, {}, pl.DataFrame())
    return engine


def _make_bar_sym(vt_symbol: str, dt: datetime, price: float, volume: float = 10000) -> BarData:
    symbol, exchange = vt_symbol.rsplit(".", 1)
    return BarData(
        symbol=symbol,
        exchange=exchange,
        datetime=dt,
        interval="d",
        open_price=price,
        high_price=price + 1,
        low_price=price - 1,
        close_price=price,
        volume=volume,
    )


# ---------------------------------------------------------------------------
# Task 0.3-1: 豁免标的（t_plus1_exempt）当日买当日卖应成交
# ---------------------------------------------------------------------------

def test_t1_exempt_symbol_same_day_buy_sell_crosses() -> None:
    """t_plus1=True 且标的在 t_plus1_exempt 集合中：当日买入 → 当日卖出应成交。

    设计：手动注入"D2 已买入"记录，然后挂卖单并调用 cross_order，
    验证豁免标的可以在同日撮合卖单。
    """
    from datetime import datetime as _dt
    d1 = _dt(2025, 1, 2)
    d2 = _dt(2025, 1, 3)

    bars = [
        _make_bar_sym(EXEMPT_SYMBOL, d1, 100.0),
        _make_bar_sym(EXEMPT_SYMBOL, d2, 100.0),
    ]
    engine = _setup_t1_engine(bars, EXEMPT_SYMBOL, t_plus1=True, exempt=True)

    # 手动推进到 D2，注入"D2 买入"记录，然后测试卖单能否通过 cross_order
    engine.datetime = d2
    engine.bars[EXEMPT_SYMBOL] = bars[1]
    engine.pre_closes[EXEMPT_SYMBOL] = 100.0
    engine.buy_dates[EXEMPT_SYMBOL] = d2.date()   # 模拟 D2 买入

    from aitrade.backtest.types import Direction, Offset, OrderData
    sell_order = OrderData(
        symbol="113001", exchange="SSE", orderid="sell1",
        direction=Direction.SHORT, offset=Offset.CLOSE,
        price=99.0, volume=100, status="nottraded",
        datetime=d2, gateway_name=engine.gateway_name,
    )
    engine.active_limit_orders[sell_order.vt_orderid] = sell_order

    prev_trade_count = engine.trade_count
    engine.cross_order()

    assert engine.trade_count > prev_trade_count, (
        "豁免标的 T+1 下当日买当日卖应成交，但 cross_order 未撮合"
    )


# ---------------------------------------------------------------------------
# Task 0.3-2: 非豁免标的当日卖单不成交、次日成交（helper 重构回归锚）
# ---------------------------------------------------------------------------

def test_t1_non_exempt_same_day_blocks_next_day_crosses() -> None:
    """t_plus1=True 且标的不在豁免集合：当日卖单被拦截，次日成交。

    此用例是 _t_plus1_locked 重构的回归锚：旧逻辑覆盖，新 helper 行为不变。
    """
    from datetime import datetime as _dt
    d1 = _dt(2025, 2, 3)
    d2 = _dt(2025, 2, 4)
    d3 = _dt(2025, 2, 5)

    bars = [
        _make_bar_sym(NORMAL_SYMBOL, d1, 100.0),
        _make_bar_sym(NORMAL_SYMBOL, d2, 100.0),
        _make_bar_sym(NORMAL_SYMBOL, d3, 100.0),
    ]
    engine = _setup_t1_engine(bars, NORMAL_SYMBOL, t_plus1=True, exempt=False)

    engine.datetime = d2
    engine.bars[NORMAL_SYMBOL] = bars[1]
    engine.pre_closes[NORMAL_SYMBOL] = 100.0
    engine.buy_dates[NORMAL_SYMBOL] = d2.date()   # 模拟 D2 买入

    from aitrade.backtest.types import Direction, Offset, OrderData
    sell_order = OrderData(
        symbol="600001", exchange="SSE", orderid="sell2",
        direction=Direction.SHORT, offset=Offset.CLOSE,
        price=99.0, volume=100, status="nottraded",
        datetime=d2, gateway_name=engine.gateway_name,
    )
    engine.active_limit_orders[sell_order.vt_orderid] = sell_order

    engine.cross_order()
    assert engine.trade_count == 0, "T+1 非豁免标的：D2 买入当日卖单应被拦截"
    assert sell_order.vt_orderid in engine.active_limit_orders, "被拦截卖单应保留待次日"

    # 次日：buy_dates 仍是 D2，今天是 D3，不再被锁 → 应成交
    engine.datetime = d3
    engine.bars[NORMAL_SYMBOL] = bars[2]
    engine.cross_order()
    assert engine.trade_count == 1, "T+1 非豁免标的：次日应允许卖出"


# ---------------------------------------------------------------------------
# Task 0.3-3: sell_to_close_intrabar 非豁免标的当日拒绝
# ---------------------------------------------------------------------------

def test_t1_sell_to_close_intrabar_non_exempt_blocked() -> None:
    """t_plus1=True + 非豁免 + 当日有买入：sell_to_close_intrabar 应拒绝（不成交）。

    这是本任务堵的绕过点：之前 sell_to_close_intrabar 完全不检查 T+1。
    """
    from datetime import datetime as _dt
    d1 = _dt(2025, 3, 3)
    d2 = _dt(2025, 3, 4)

    bars = [
        _make_bar_sym(NORMAL_SYMBOL, d1, 100.0),
        _make_bar_sym(NORMAL_SYMBOL, d2, 100.0),
    ]
    engine = _setup_t1_engine(bars, NORMAL_SYMBOL, t_plus1=True, exempt=False)

    # 模拟 D2 已买入
    engine.datetime = d2
    engine.buy_dates[NORMAL_SYMBOL] = d2.date()

    prev_count = engine.trade_count
    engine.sell_to_close_intrabar(NORMAL_SYMBOL, 99.0, 100)

    assert engine.trade_count == prev_count, (
        "T+1 非豁免标的：sell_to_close_intrabar 当日买入后应拒绝成交，但成交了"
    )


# ---------------------------------------------------------------------------
# Task 0.3-4: sell_to_close_intrabar 豁免标的当日成交
# ---------------------------------------------------------------------------

def test_t1_sell_to_close_intrabar_exempt_crosses() -> None:
    """t_plus1=True + 豁免标的 + 当日有买入：sell_to_close_intrabar 应正常成交。"""
    from datetime import datetime as _dt
    d1 = _dt(2025, 4, 1)
    d2 = _dt(2025, 4, 2)

    bars = [
        _make_bar_sym(EXEMPT_SYMBOL, d1, 100.0),
        _make_bar_sym(EXEMPT_SYMBOL, d2, 100.0),
    ]
    engine = _setup_t1_engine(bars, EXEMPT_SYMBOL, t_plus1=True, exempt=True)

    # 模拟 D2 已买入
    engine.datetime = d2
    engine.buy_dates[EXEMPT_SYMBOL] = d2.date()

    prev_count = engine.trade_count
    engine.sell_to_close_intrabar(EXEMPT_SYMBOL, 99.0, 100)

    assert engine.trade_count > prev_count, (
        "豁免标的 T+1 下：sell_to_close_intrabar 当日买入后应允许成交"
    )


# =============================================================================
# Task 4.4 Part A: infer_t_plus1 纯函数 + 引擎自动推断豁免
# =============================================================================


@pytest.mark.parametrize("vt_symbol,expected", [
    # 沪市可转债：110/111/113/118 → False（T+0）
    ("110001.SSE",  False),
    ("111001.SSE",  False),
    ("113001.SSE",  False),
    ("118001.SSE",  False),
    # 深市可转债：123/127/128 → False（T+0）
    ("123001.SZSE", False),
    ("127001.SZSE", False),
    ("128001.SZSE", False),
    # 主板股票：True（T+1）
    ("600000.SSE",  True),
    ("000001.SZSE", True),
    # 创业板：True
    ("300001.SZSE", True),
    # 科创板：True
    ("688001.SSE",  True),
    # 北交所：True
    ("430001.BSE",  True),
    # ETF：True
    ("510300.SSE",  True),
    # 无交易所后缀：默认 True
    ("999999",       True),
])
def test_infer_t_plus1(vt_symbol: str, expected: bool) -> None:
    """infer_t_plus1 纯函数：转债前缀 → False（T+0），其余 → True（T+1）。"""
    from aitrade.backtest.instrument import infer_t_plus1
    result = infer_t_plus1(vt_symbol)
    assert result == expected, (
        f"infer_t_plus1({vt_symbol!r}) = {result}，预期 {expected}"
    )


def _setup_auto_infer_engine(
    bars: list[BarData],
    vt_symbol: str,
) -> BacktestingEngine:
    """构造引擎，无合约配置，依赖 set_parameters 自动推断 t_plus1_exempt。"""
    symbol, exchange = vt_symbol.rsplit(".", 1)

    class _Loader:
        def load_bar_data(self, sym, interval, start, end) -> list[BarData]:
            return list(bars)
        def load_contract_settings(self) -> dict:
            return {}

    import polars as pl
    engine = BacktestingEngine(_Loader())
    start = bars[0].datetime
    end = bars[-1].datetime
    engine.set_parameters([vt_symbol], "d", start, end, capital=10_000_000)
    engine.sizes[vt_symbol] = 1
    engine.priceticks[vt_symbol] = 0.01
    engine.long_rates[vt_symbol] = 0.0
    engine.short_rates[vt_symbol] = 0.0
    engine.stamp_duties[vt_symbol] = 0.0
    engine.slippages[vt_symbol] = 0.0
    engine.t_plus1 = True  # 全局 T+1 开
    engine.add_strategy(_NoOpStrategy, {}, pl.DataFrame())
    return engine


def test_cb_auto_infer_no_contract_setting_exempt() -> None:
    """转债无合约配置时，set_parameters 应自动将其加入 t_plus1_exempt（T+0 推断路径）。

    验证：113xxx.SSE 无合约配置 → t_plus1_exempt 包含该标的。
    """
    vt_symbol = "113050.SSE"
    d1 = datetime(2025, 5, 1)
    d2 = datetime(2025, 5, 2)

    bars = [
        _make_bar_sym(vt_symbol, d1, 100.0),
        _make_bar_sym(vt_symbol, d2, 100.0),
    ]
    engine = _setup_auto_infer_engine(bars, vt_symbol)

    assert vt_symbol in engine.t_plus1_exempt, (
        f"转债 {vt_symbol} 无合约配置时应自动加入 t_plus1_exempt，"
        f"实际 t_plus1_exempt={engine.t_plus1_exempt}"
    )


def test_stock_auto_infer_no_contract_setting_not_exempt() -> None:
    """普通股票无合约配置时，set_parameters 不应将其加入 t_plus1_exempt（T+1 保持约束）。

    验证：600xxx.SSE 无合约配置 → t_plus1_exempt 不包含该标的。
    """
    vt_symbol = "600000.SSE"
    d1 = datetime(2025, 5, 1)
    d2 = datetime(2025, 5, 2)

    bars = [
        _make_bar_sym(vt_symbol, d1, 100.0),
        _make_bar_sym(vt_symbol, d2, 100.0),
    ]
    engine = _setup_auto_infer_engine(bars, vt_symbol)

    assert vt_symbol not in engine.t_plus1_exempt, (
        f"股票 {vt_symbol} 无合约配置时不应加入 t_plus1_exempt，"
        f"实际 t_plus1_exempt={engine.t_plus1_exempt}"
    )


def test_cb_with_contract_setting_t_plus1_true_overrides_infer() -> None:
    """转债有合约配置且 t_plus1=True（显式锁定），应覆盖推断结果，不进入豁免集合。

    此用例验证：即便 infer_t_plus1 返回 False，contract.json 配置优先。
    """
    vt_symbol = "113050.SSE"
    d1 = datetime(2025, 6, 1)
    d2 = datetime(2025, 6, 2)
    bars = [
        _make_bar_sym(vt_symbol, d1, 100.0),
        _make_bar_sym(vt_symbol, d2, 100.0),
    ]

    class _LoaderWithSetting:
        def load_bar_data(self, sym, interval, start, end) -> list[BarData]:
            return list(bars)
        def load_contract_settings(self) -> dict:
            # 显式 t_plus1=True（覆盖推断的 T+0）
            return {vt_symbol: {
                "long_rate": 0.0,
                "short_rate": 0.0,
                "size": 1,
                "pricetick": 0.01,
                "t_plus1": True,  # 显式覆盖：锁定 T+1
            }}

    import polars as pl
    engine = BacktestingEngine(_LoaderWithSetting())
    engine.set_parameters([vt_symbol], "d", d1, d2, capital=1_000_000)
    engine.add_strategy(_NoOpStrategy, {}, pl.DataFrame())

    assert vt_symbol not in engine.t_plus1_exempt, (
        f"合约配置 t_plus1=True 应覆盖推断，{vt_symbol} 不应进入豁免集合"
    )
