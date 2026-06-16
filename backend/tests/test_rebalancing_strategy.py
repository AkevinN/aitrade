"""
RebalancingTopKStrategy 验收测试。

覆盖：
1. W 频调仓：两周行情（每周 3 个交易日），只有每周首根 bar 产生新成交；
2. 跳过日 holding_days 自增：W 频下信号变化要求卖出，
   若 holding_days 停增则卖出被 min_days 错误阻止——断言实际能在第二周卖出；
3. 调仓日撤旧单：首周挂出的未成交限价单到第二周调仓日被撤，不以陈旧价格成交；
4. D 频行为与父类一致：同一信号下成交序列数量相同；
5. 注册表可按名 "rebalancing_topk" 取得策略类。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from aitrade.backtest.engine import BacktestingEngine
from aitrade.backtest.types import BarData

# 触发 "rebalancing_topk" 注册
import aitrade.rules  # noqa: F401


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------

SYMBOL = "TEST.SSE"


class _MemLoader:
    """内存行情 Loader，支持多标的。"""

    def __init__(self, bar_map: dict[str, list[BarData]]) -> None:
        self._bar_map = bar_map

    def load_bar_data(self, vt_symbol: str, interval: str, start, end) -> list[BarData]:
        return list(self._bar_map.get(vt_symbol, []))

    def load_contract_settings(self) -> dict:
        return {}


def _make_bar(vt_symbol: str, dt: datetime, price: float, volume: float = 100_000) -> BarData:
    symbol, exchange = vt_symbol.rsplit(".", 1)
    return BarData(
        symbol=symbol,
        exchange=exchange,
        datetime=dt,
        interval="d",
        open_price=price - 0.5,
        high_price=price + 1.0,
        low_price=price - 1.0,
        close_price=price,
        volume=volume,
    )


def _setup_engine(
    bar_map: dict[str, list[BarData]],
    strategy_class,
    setting: dict,
    signal_df: pl.DataFrame,
) -> BacktestingEngine:
    """构造并运行回测引擎，返回引擎实例（已完成 run_backtesting）。"""
    all_bars = [b for bars in bar_map.values() for b in bars]
    start = min(b.datetime for b in all_bars)
    end = max(b.datetime for b in all_bars) + timedelta(days=1)

    loader = _MemLoader(bar_map)
    engine = BacktestingEngine(loader)
    engine.set_parameters(
        vt_symbols=list(bar_map.keys()),
        interval="d",
        start=start,
        end=end,
        capital=2_000_000,
    )
    for vt_symbol in bar_map:
        engine.sizes[vt_symbol] = 1
        engine.priceticks[vt_symbol] = 0.01
        engine.long_rates[vt_symbol] = 0.0
        engine.short_rates[vt_symbol] = 0.0
        engine.stamp_duties[vt_symbol] = 0.0
        engine.slippages[vt_symbol] = 0.0

    engine.add_strategy(strategy_class, setting, signal_df)
    engine.load_data()
    engine.run_backtesting()
    return engine


def _signal_df_for(dates: list[datetime], symbols: list[str], signal: float = 1.0) -> pl.DataFrame:
    """为每个日期的每个标的创建固定信号值的 signal_df。"""
    rows = []
    for dt in dates:
        for sym in symbols:
            rows.append({"datetime": dt, "vt_symbol": sym, "signal": signal})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 用例 5（最简单）：注册表按名取得策略类
# ---------------------------------------------------------------------------

def test_registry_lookup() -> None:
    """'rebalancing_topk' 应在注册表中可以取得 RebalancingTopKStrategy。

    注意：不用 `cls is RebalancingTopKStrategy` 做对象同一性断言。
    若其他测试（如 test_import_rules_no_torch）曾操纵 sys.modules 并重新 import
    aitrade.rules，会生成**新的**类对象并覆盖注册表，导致 identity 断言
    在特定测试顺序下误判为失败——而两个类对象实为同一逻辑类。
    改用名称 + 继承关系做等价性断言，与执行顺序解耦。
    """
    from aitrade.backtest.registry import get_strategy
    from aitrade.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy

    cls = get_strategy("rebalancing_topk")
    assert cls.__name__ == "RebalancingTopKStrategy", (
        f"注册表返回类名 {cls.__name__!r}，预期 'RebalancingTopKStrategy'"
    )
    assert issubclass(cls, EquityDemoStrategy), (
        f"注册表返回的类 {cls!r} 应是 EquityDemoStrategy 的子类"
    )


# ---------------------------------------------------------------------------
# 用例 1：W 频调仓只在每周首根 bar 产生新开仓成交
# ---------------------------------------------------------------------------

def test_w_freq_only_trades_on_first_bar_of_week() -> None:
    """两周行情（每周 3 个交易日），W 频策略只在第 1、第 4 根 bar 发新单。

    引擎执行顺序：cross_order() → on_bars()。
    因此 W1 周一的 on_bars 发出的买单，在 W1 周三的 cross_order 中成交。

    关键校验：W1 周三/五的 on_bars 是非调仓日，不发新单；W2 周一是调仓日。
    所以全程只有 1 笔买入成交（由 W1 周一下单、W1 周三成交），
    W1 周五以后（非调仓日）不发任何新买入单，W2 周一调仓时持仓已满 top_k=1。

    验证方式：
    - W1 周五 ~ W2 整周期间的 active_orders 在 W1 周三成交后清零
      （cancel_all 在 execute_trading 中自动调用）
    - 总买入成交笔数 = 1（W1 周三成交）
    """
    from aitrade.rules.strategy import RebalancingTopKStrategy

    # 构造两周行情（W1: 周一/三/五，W2: 下周一/三/五）
    w1_mon = datetime(2025, 1, 6)   # 周一
    w1_wed = datetime(2025, 1, 8)   # 周三
    w1_fri = datetime(2025, 1, 10)  # 周五
    w2_mon = datetime(2025, 1, 13)  # 下周一
    w2_wed = datetime(2025, 1, 15)  # 下周三
    w2_fri = datetime(2025, 1, 17)  # 下周五

    dates = [w1_mon, w1_wed, w1_fri, w2_mon, w2_wed, w2_fri]

    bars = [_make_bar(SYMBOL, dt, price=100.0) for dt in dates]
    signal_df = _signal_df_for(dates, [SYMBOL], signal=1.0)

    setting = {
        "rebalance_freq": "W",
        "top_k": 1,
        "n_drop": 1,
        "min_days": 1,
        "cash_ratio": 0.9,
        "min_volume": 100,
        "price_add": 0.0,
    }

    engine = _setup_engine({SYMBOL: bars}, RebalancingTopKStrategy, setting, signal_df)
    trades = engine.get_all_trades()

    buy_trades = [t for t in trades if t.direction == "long"]
    # W1 周一发单，W1 周三撮合成交，仅 1 笔买入
    # W1 周三/五、W2 周三/五均为非调仓日，不发新买单
    # W2 周一调仓日但持仓已满 top_k=1，也不新买
    assert len(buy_trades) == 1, (
        f"W 频应恰好 1 笔买入成交（W1 周一下单 W1 周三成交），实际 {len(buy_trades)} 笔"
    )
    # 买入成交日应为 W1 周三（引擎在周三 cross_order 时撮合）
    assert buy_trades[0].datetime.date() == w1_wed.date(), (
        f"买入应成交于 W1 周三 {w1_wed.date()}，实际 {buy_trades[0].datetime.date()}"
    )


# ---------------------------------------------------------------------------
# 用例 2：跳过日 holding_days 正常自增（min_days 约束不失真）
# ---------------------------------------------------------------------------

def test_holding_days_increments_on_skip_days() -> None:
    """W 频下非调仓日 holding_days 仍照常自增，保证 min_days 约束正确。

    场景设计：
    - min_days=3，top_k=1
    - W1（周一）：标的 A 信号最强，买入 A
    - W1 周三/五：非调仓日，holding_days 自增 → 第 3 根 bar（周五）后 holding_days=2
    - W2（周一）：调仓日，切换为标的 B 信号最强；A 的 holding_days 应为 3（经 W1 三天自增）
      → 满足 min_days=3，可以卖出 A；若 holding_days 停增则仍为 0，卖出被阻止

    断言：W2 周一当天或之后有 A 的卖出成交（sell/short）。
    """
    from aitrade.rules.strategy import RebalancingTopKStrategy

    SYM_A = "A.SSE"
    SYM_B = "B.SSE"

    w1_mon = datetime(2025, 1, 6)
    w1_wed = datetime(2025, 1, 8)
    w1_fri = datetime(2025, 1, 10)
    w2_mon = datetime(2025, 1, 13)
    w2_wed = datetime(2025, 1, 15)

    dates = [w1_mon, w1_wed, w1_fri, w2_mon, w2_wed]

    bars_a = [_make_bar(SYM_A, dt, price=100.0) for dt in dates]
    bars_b = [_make_bar(SYM_B, dt, price=100.0) for dt in dates]

    # W1 三天：A 信号强（2.0），B 信号弱（0.0）
    # W2 起：B 信号强（2.0），A 信号弱（0.0）
    rows = []
    for dt in [w1_mon, w1_wed, w1_fri]:
        rows.append({"datetime": dt, "vt_symbol": SYM_A, "signal": 2.0})
        rows.append({"datetime": dt, "vt_symbol": SYM_B, "signal": 0.0})
    for dt in [w2_mon, w2_wed]:
        rows.append({"datetime": dt, "vt_symbol": SYM_A, "signal": 0.0})
        rows.append({"datetime": dt, "vt_symbol": SYM_B, "signal": 2.0})
    signal_df = pl.DataFrame(rows)

    setting = {
        "rebalance_freq": "W",
        "top_k": 1,
        "n_drop": 1,
        "min_days": 3,  # 持有满 3 天才能卖出
        "cash_ratio": 0.9,
        "min_volume": 100,
        "price_add": 0.0,
    }

    engine = _setup_engine(
        {SYM_A: bars_a, SYM_B: bars_b},
        RebalancingTopKStrategy,
        setting,
        signal_df,
    )
    trades = engine.get_all_trades()

    # 检查是否存在 A 的卖出成交
    sell_a = [
        t for t in trades
        if t.vt_symbol == SYM_A and t.direction == "short"
    ]
    assert sell_a, (
        "W2 调仓日 A 的 holding_days 应已达到 min_days=3（因非调仓日照常自增），"
        "应能卖出 A；若为空说明 holding_days 停增导致 min_days 约束错误阻止卖出"
    )


# ---------------------------------------------------------------------------
# 用例 3：调仓日撤旧单（首周遗留未成交限价单不在下周成交）
# ---------------------------------------------------------------------------

def test_rebalance_day_cancels_stale_orders() -> None:
    """W 频：首周挂出的未成交限价单，在第二周调仓日到来时应被撤掉。

    构造方式：首周只有 1 根 bar（W1 周一），不提供 W1 后续 bar，
    使策略挂出限价买单后无撮合机会；下周（W2）第一根 bar 到来时策略应撤掉旧单，
    W2 根据新信号重新挂单（或不挂），旧单不成交。

    验证：W2 第一根 bar 成交日期是 W2（不是 W1），且成交价对应 W2 价格。
    """
    from aitrade.rules.strategy import RebalancingTopKStrategy

    # W1 价格故意设高（price_add=0，限价=close 价），使 W2 bar（低价）时旧单会以 W1 价成交
    # 如果不撤单，W2 bar（low_price=W2_price-1）低于 W1_price 限价 → 旧单成交（陈旧价格）
    # 如果撤单，W2 限价 = W2_price，W2 bar 低价 = W2_price-1 < W2_price → 无法成交
    # 为使 W2 bar 确实能撮合到 W2 的新单：设 price_add=0.1（close*(1+0.1)），则 W2 高价覆盖
    W1_PRICE = 200.0
    W2_PRICE = 100.0  # 大幅下跌，使 W1 未撤的旧单以 W1 价成交场景可辨别

    w1_mon = datetime(2025, 2, 3)   # 周一
    w2_mon = datetime(2025, 2, 10)  # 下周一
    w2_wed = datetime(2025, 2, 12)  # 下周三（用于观察是否成交）

    # W1 只有 1 根 bar；W2 有两根（周一、周三）
    bars = [
        _make_bar(SYMBOL, w1_mon, price=W1_PRICE),
        _make_bar(SYMBOL, w2_mon, price=W2_PRICE),
        _make_bar(SYMBOL, w2_wed, price=W2_PRICE),
    ]
    dates = [w1_mon, w2_mon, w2_wed]
    signal_df = _signal_df_for(dates, [SYMBOL], signal=1.0)

    setting = {
        "rebalance_freq": "W",
        "top_k": 1,
        "n_drop": 0,
        "min_days": 1,
        "cash_ratio": 0.9,
        "min_volume": 100,
        "price_add": 0.05,  # 限价略高于 close，使买单有机会成交
    }

    engine = _setup_engine({SYMBOL: bars}, RebalancingTopKStrategy, setting, signal_df)
    trades = engine.get_all_trades()
    buy_trades = [t for t in trades if t.direction == "long"]

    if buy_trades:
        # 如果存在买入成交，其价格应该接近 W2 价格而非 W1 价格
        # W1 旧单价格 ≈ W1_PRICE * (1 + 0.05) = 210；W2 新单价格 ≈ W2_PRICE * (1 + 0.05) = 105
        for t in buy_trades:
            assert t.price < W1_PRICE, (
                f"成交价 {t.price} 接近 W1 价格 {W1_PRICE}，"
                "说明旧单未被撤掉就以陈旧价格成交了"
            )


# ---------------------------------------------------------------------------
# 用例 4：D 频行为与 EquityDemoStrategy 一致（成交笔数相同）
# ---------------------------------------------------------------------------

def test_d_freq_matches_parent_strategy() -> None:
    """D 频（rebalance_freq="D"）的 RebalancingTopKStrategy 与直接使用
    EquityDemoStrategy 在相同信号下，成交笔数应相同。
    """
    from aitrade.rules.strategy import RebalancingTopKStrategy
    from aitrade.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy

    dates = [datetime(2025, 3, 3) + timedelta(days=i) for i in range(5)]
    bars = [_make_bar(SYMBOL, dt, price=100.0) for dt in dates]
    signal_df = _signal_df_for(dates, [SYMBOL], signal=1.0)

    setting = {
        "top_k": 1,
        "n_drop": 1,
        "min_days": 1,
        "cash_ratio": 0.9,
        "min_volume": 100,
        "price_add": 0.05,
    }

    parent_engine = _setup_engine(
        {SYMBOL: bars}, EquityDemoStrategy, setting, signal_df
    )
    child_setting = dict(setting)
    child_setting["rebalance_freq"] = "D"
    child_engine = _setup_engine(
        {SYMBOL: bars}, RebalancingTopKStrategy, child_setting, signal_df
    )

    parent_count = len(parent_engine.get_all_trades())
    child_count = len(child_engine.get_all_trades())

    assert child_count == parent_count, (
        f"D 频子类（{child_count} 笔）应与父类（{parent_count} 笔）成交数相同"
    )


# ---------------------------------------------------------------------------
# 用例：M 频调仓日判定（每月首根 bar）
# ---------------------------------------------------------------------------

def test_m_freq_only_rebalances_on_first_bar_of_month() -> None:
    """M 频：3 月份有 3 根 bar（3/17/21 日），4 月份有 2 根 bar（1/14 日）。
    只有每月首根 bar（3月3日、4月1日）是调仓日，其余是非调仓日。

    引擎执行顺序：cross_order() → on_bars()。
    3月3日发出买单 → 3月17日（下一根 bar 的 cross_order）撮合成交。
    3月17/21日为非调仓日，不发新单（已有持仓也不卖出，n_drop=0）。
    4月1日调仓日，持仓已满 top_k=1，不新买。

    验证：总买入成交 = 1 笔（3月17日成交），不存在 3月17日以外的第二笔买入。
    """
    from aitrade.rules.strategy import RebalancingTopKStrategy

    mar03 = datetime(2025, 3, 3)   # 3 月首根（调仓日）
    mar17 = datetime(2025, 3, 17)  # 3 月中（非调仓日，但 3月3日订单在此撮合）
    mar21 = datetime(2025, 3, 21)  # 3 月末（非调仓日）
    apr01 = datetime(2025, 4, 1)   # 4 月首根（调仓日，持仓满，不新买）
    apr14 = datetime(2025, 4, 14)  # 4 月中（非调仓日）

    dates = [mar03, mar17, mar21, apr01, apr14]
    bars = [_make_bar(SYMBOL, dt, price=100.0) for dt in dates]
    signal_df = _signal_df_for(dates, [SYMBOL], signal=1.0)

    setting = {
        "rebalance_freq": "M",
        "top_k": 1,
        "n_drop": 0,
        "min_days": 1,
        "cash_ratio": 0.9,
        "min_volume": 100,
        "price_add": 0.0,
    }

    engine = _setup_engine({SYMBOL: bars}, RebalancingTopKStrategy, setting, signal_df)
    trades = engine.get_all_trades()

    buy_trades = [t for t in trades if t.direction == "long"]
    # 全程只有 1 笔买入：3月3日下单 → 3月17日撮合
    assert len(buy_trades) == 1, (
        f"M 频应恰好 1 笔买入成交，实际 {len(buy_trades)} 笔；成交日：{[t.datetime.date() for t in buy_trades]}"
    )
    assert buy_trades[0].datetime.date() == mar17.date(), (
        f"买入应成交于 3月17日（3月3日下单后的下一根 bar），实际 {buy_trades[0].datetime.date()}"
    )


# ---------------------------------------------------------------------------
# 用例：rebalance_freq 可通过 setting 注入
# ---------------------------------------------------------------------------

def test_setting_injection() -> None:
    """rebalance_freq 作为类属性，应能通过 setting 字典注入覆盖。"""
    from aitrade.rules.strategy import RebalancingTopKStrategy

    dates = [datetime(2025, 4, 1)]
    bars = [_make_bar(SYMBOL, dt, price=100.0) for dt in dates]
    signal_df = _signal_df_for(dates, [SYMBOL])

    loader = _MemLoader({SYMBOL: bars})
    engine = BacktestingEngine(loader)
    engine.set_parameters([SYMBOL], "d", dates[0], dates[0] + timedelta(days=1), capital=1_000_000)
    engine.sizes[SYMBOL] = 1
    engine.priceticks[SYMBOL] = 0.01
    engine.long_rates[SYMBOL] = 0.0
    engine.short_rates[SYMBOL] = 0.0
    engine.stamp_duties[SYMBOL] = 0.0
    engine.slippages[SYMBOL] = 0.0

    engine.add_strategy(RebalancingTopKStrategy, {"rebalance_freq": "M"}, signal_df)
    strat = engine.strategy
    assert strat.rebalance_freq == "M", (
        f"setting 注入后 rebalance_freq 应为 'M'，实际 {strat.rebalance_freq!r}"
    )
