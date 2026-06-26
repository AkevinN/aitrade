"""HalfPositionT0Strategy 的 Hypothesis 属性测试（Task 7）。

覆盖 design.md「Correctness Properties」中的 Property 1/3/4/5：
- Property 5：不加杠杆、不做空（持仓权重 ∈ [0,1]、现金 ≥ 0）。
- Property 4：T+1 不可卖今仓（无「当日买入又当日卖出」的限价做空腿）。
- Property 3：幂等回半仓（无触价的「安静日」收盘权重≈半仓；连续安静日净值不漂移）。
- Property 1：无前视（扰动未来 bar 不改变过去日的成交）。

设计立场与已知简化（见 design.md）：策略受引擎**符号级** T+1 约束，当日买腿成交后，
当日收盘的下调卖出会被引擎按符号级 T+1 拦截（顺延次日）。因此 Property 4 在限价做 T 腿上
天然成立，本测试断言「鲁棒的非负仓位 + 非负现金 + 无当日买后限价做空腿减仓到昨仓以下」组合
不变量，而非脆弱的逐手撮合细节（见各测试内注释）。

Feature: half-position-t0-backtest
"""

from __future__ import annotations

from datetime import datetime, time

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aitrade.backtest.engine import BacktestingEngine
from aitrade.backtest.t0.strategy import HalfPositionT0Strategy
from aitrade.backtest.t0.tick_policy import FixedTick
from aitrade.backtest.types import BarData, Direction

# —— 公共常量 ——
_SYMBOL = "AAA"
_EXCHANGE = "SSE"
_VT = "AAA.SSE"
_CAPITAL = 1_000_000
_SELL_TICK = 0.02
_BUY_TICK = 0.02
_LOT = 100  # A 股最小交易单位（股）


def _grid(x: float) -> float:
    """把价格对齐到 0.01 的价格网格（四舍五入到分）。"""
    return round(round(x / 0.01) * 0.01, 2)


def _make_day(
    day: int,
    open_px: float,
    intraday_high: float,
    intraday_low: float,
    close: float,
) -> list[BarData]:
    """造一个交易日的 3 根 1m bar：09:30 开盘、10:30 日内高低、15:00 收盘。

    与集成测试 ``tests/test_t0_strategy.py`` 同构：首根 bar 触发开盘挂单，
    中间 bar 提供日内高低供撮合，末根（15:00 ≥ close_time）触发回半仓。

    Args:
        day: 2024-01 内的日序号（1..28），用于生成确定性时间戳。
        open_px: 开盘价（元），已对齐价格网格。
        intraday_high: 日内最高价（元），须 ≥ max(open, close)。
        intraday_low: 日内最低价（元），须 ≤ min(open, close) 且 > 0。
        close: 收盘价（元），已对齐价格网格。

    Returns:
        该交易日的 3 根 BarData，时间戳依次为 09:30 / 10:30 / 15:00。
    """
    base = datetime(2024, 1, day)

    def mk(hh: int, mm: int, o: float, h: float, ll: float, c: float) -> BarData:
        return BarData(
            symbol=_SYMBOL,
            exchange=_EXCHANGE,
            datetime=base.replace(hour=hh, minute=mm),
            interval="1m",
            open_price=o,
            high_price=h,
            low_price=ll,
            close_price=c,
            volume=10000,
        )

    return [
        mk(9, 30, open_px, open_px, open_px, open_px),
        mk(10, 30, open_px, intraday_high, intraday_low, open_px),
        mk(15, 0, close, close, close, close),
    ]


def _run(days_bars: list[list[BarData]], *, t_plus1: bool = False) -> BacktestingEngine:
    """用与集成测试相同的 harness 在引擎@1m 上跑给定多日 bar 序列。

    复用 ``tests/test_t0_strategy.py`` 的设置：sizes=1、priceticks=0.01、
    所有 rates/slippages=0、limit_ratios=None、close_time=14:57、FixedTick(0.02/0.02)。

    Args:
        days_bars: 逐日的 bar 列表（每日 3 根，见 :func:`_make_day`）。
        t_plus1: 是否开启引擎符号级 T+1 约束；默认 False（与集成测试一致）。

    Returns:
        已跑完 ``run_backtesting`` 的引擎实例，供读取 get_pos / get_all_trades / 现金等。
    """
    bars = [b for day in days_bars for b in day]
    eng = BacktestingEngine(data_loader=_MemLoader(bars))
    eng.set_parameters(
        [_VT], "1m", bars[0].datetime, bars[-1].datetime, capital=_CAPITAL
    )
    eng.sizes[_VT] = 1
    eng.priceticks[_VT] = 0.01
    eng.long_rates[_VT] = 0.0
    eng.short_rates[_VT] = 0.0
    eng.stamp_duties[_VT] = 0.0
    eng.slippages[_VT] = 0.0
    eng.limit_ratios[_VT] = None
    eng.t_plus1 = t_plus1
    eng.add_strategy(
        HalfPositionT0Strategy,
        {
            "vt_symbol": _VT,
            "tick_policy": FixedTick(sell_tick=_SELL_TICK, buy_tick=_BUY_TICK),
            "swing_frac": 1.0,
            "base_weight": 0.5,
            "close_time": time(14, 57),
        },
        None,
    )
    eng.load_data()
    eng.run_backtesting()
    return eng


class _MemLoader:
    """内存 bar 加载器：直接吐回构造好的 bar 列表（无 IO，确定性）。"""

    def __init__(self, bars: list[BarData]) -> None:
        self._bars = bars

    def load_bar_data(self, *a) -> list[BarData]:
        """返回预置的 bar 列表副本。"""
        return list(self._bars)

    def load_contract_settings(self) -> dict:
        """无合约元数据：返回空 dict。"""
        return {}


# —— Hypothesis 策略：生成「合法」的单日 OHLC ——


@st.composite
def _valid_day(draw, day: int) -> list[BarData]:
    """生成一个合法的随机交易日：价位 8–12 元、high≥max(o,c)、low≤min(o,c)、low>0。

    所有价格对齐到 0.01 网格，日内高低围绕开盘价随机扩展，保证 OHLC 的偏序合法。

    Args:
        day: 该日的日序号（传给 :func:`_make_day`）。

    Returns:
        该交易日的 3 根 BarData。
    """
    open_px = _grid(draw(st.floats(min_value=8.0, max_value=12.0)))
    close = _grid(draw(st.floats(min_value=8.0, max_value=12.0)))
    up = draw(st.floats(min_value=0.0, max_value=0.5))
    down = draw(st.floats(min_value=0.0, max_value=0.5))
    high = _grid(max(open_px, close) + up)
    low = _grid(min(open_px, close) - down)
    if low <= 0:
        low = _grid(min(open_px, close))  # 兜底：不会发生（价位下界 8 元），保险起见
    return _make_day(day, open_px, high, low, close)


@st.composite
def _valid_run(draw) -> list[list[BarData]]:
    """生成 2–5 个合法交易日的序列（日序号 1,2,3,...，时间戳天然递增）。

    Returns:
        逐日 bar 列表，可直接喂给 :func:`_run`。
    """
    n_days = draw(st.integers(min_value=2, max_value=5))
    return [draw(_valid_day(d)) for d in range(1, n_days + 1)]


_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ============================================================================
# Property 5: 不加杠杆、不做空
# ============================================================================
@_SETTINGS
@given(days=_valid_run())
def test_property5_no_leverage_no_short(days: list[list[BarData]]) -> None:
    # Feature: half-position-t0-backtest, Property 5: 不加杠杆、不做空——
    # 任一执行路径下，运行结束时持仓权重 ∈ [0,1]、现金 ≥ 0。
    #
    # 实用判据（design 注明）：跑完后 get_pos >= 0 且 get_cash_available() >= -1e-6；
    # 并额外断言期末持仓权重 pos*price/portfolio_value <= 1.0 + 0.05（从不实质加杠杆）。
    eng = _run(days)

    pos = eng.strategy.get_pos(_VT)
    cash = eng.get_cash_available()
    assert pos >= -1e-9, f"期末持仓为负（做空）: {pos}"
    assert cash >= -1e-6, f"期末现金为负（加杠杆）: {cash}"

    # 期末权重不实质 > 1：用末日收盘价估市值
    last_close = days[-1][-1].close_price
    portfolio_value = cash + pos * last_close  # size = 1
    assert portfolio_value > 0
    weight = pos * last_close / portfolio_value
    assert weight <= 1.0 + 0.05, f"期末权重 {weight} 实质 > 1（加杠杆）"


# ============================================================================
# Property 4: T+1 不可卖今仓
# ============================================================================
@_SETTINGS
@given(days=_valid_run())
def test_property4_t_plus1_no_same_day_resell(days: list[list[BarData]]) -> None:
    # Feature: half-position-t0-backtest, Property 4: T+1 不可卖今仓——
    # 开启引擎符号级 T+1（eng.t_plus1=True）后，任一当日买入的股份当日不被卖出。
    #
    # 鲁棒判据（design 明确：逐手撮合细节脆弱时取较弱不变量）：本测试断言以下组合不变量，
    # 它们是引擎符号级 T+1 锁定的可观测后果，且不依赖区分昨仓/今仓的脆弱内部状态：
    #   (a) 全程末仓非负、现金非负（容许 design 注明的「粗粒度 T+1 carry」分级误差）；
    #   (b) 任一交易日内，SHORT 成交只卖「该日开盘已持有的仓位（昨仓）」——即当日累计
    #       净 SHORT 量不超过该日开盘前的持仓量。这等价于「当日买入的股份当日不被卖出」：
    #       今仓（当日 LONG 成交新增的量）始终不被同日 SHORT 触及。
    # 注：合法存在「同日 先 LONG 后 SHORT」——开盘回半仓的卖腿卖的是昨仓，故不能简单地
    #     禁止同日 LONG→SHORT 顺序；正确口径是「SHORT 累计量 ≤ 昨仓量」（见 design 已知缺口）。
    eng = _run(days, t_plus1=True)

    # (a) 末仓非负；现金允许极小负值（粗粒度 T+1 carry 的 lot-rounding 残差，远小于 1 手市值）。
    assert eng.strategy.get_pos(_VT) >= -1e-9
    last_open = days[-1][0].open_price
    assert eng.get_cash_available() >= -_LOT * last_open, (
        f"现金负值超出粗粒度 T+1 carry 容差: {eng.get_cash_available()}"
    )

    # (b) 按时间重放成交，逐日核对：当日 SHORT 累计量 ≤ 该日「开盘前持仓量（昨仓）」。
    trades = sorted(eng.get_all_trades(), key=lambda t: (t.datetime, t.tradeid))
    pos_before_today = 0.0       # 进入当前交易日前（昨日收盘）的持仓
    running_pos = 0.0            # 随成交滚动的持仓
    cur_date = None
    short_sold_today = 0.0       # 当日累计 SHORT 量
    for t in trades:
        d = t.datetime.date()
        if d != cur_date:
            pos_before_today = running_pos  # 昨仓 = 进入今日前的持仓
            short_sold_today = 0.0
            cur_date = d
        if t.direction == Direction.LONG:
            running_pos += t.volume
        else:  # SHORT
            short_sold_today += t.volume
            running_pos -= t.volume
            # 当日卖出量不得超过昨仓——超出即意味着卖了今仓，违反 T+1。
            assert short_sold_today <= pos_before_today + 1e-6, (
                f"{d}: 当日 SHORT 累计 {short_sold_today} 超过昨仓 {pos_before_today}，"
                f"卖了今仓，违反 T+1"
            )
            assert running_pos >= -1e-6, f"{d}: 仓位被卖成负数 {running_pos}"


# ============================================================================
# Property 3: 幂等回半仓（安静日不漂移）
# ============================================================================
@_SETTINGS
@given(
    open_px=st.sampled_from([9.0, 9.5, 10.0, 10.5, 11.0]),
    n_quiet=st.integers(min_value=1, max_value=3),
)
def test_property3_quiet_day_restores_half(open_px: float, n_quiet: int) -> None:
    # Feature: half-position-t0-backtest, Property 3: 幂等回半仓——
    # 在「安静日」（日内高低严格落在 (open-buy_tick, open+sell_tick) 内，两腿都不触价）
    # 收盘后持仓权重 ≈ base_weight(0.5)，容差 ±1 手(100 股)。
    #
    # 安静日刻意构造（不依赖随机）：日内 high/low 仅偏离开盘 0.005 元 < 0.02 的档位，
    # 故卖腿(open+0.02)/买腿(open-0.02)均不触价 → 当日只在开盘建/维持半仓。
    eps = 0.005  # < min(buy_tick, sell_tick) = 0.02
    quiet_high = _grid(open_px + eps)
    quiet_low = _grid(open_px - eps)

    # 第一天先建半仓（同样安静），随后 n_quiet 个安静日
    days = [
        _make_day(d, open_px, quiet_high, quiet_low, open_px)
        for d in range(1, n_quiet + 2)
    ]
    eng = _run(days)

    pos = eng.strategy.get_pos(_VT)
    target = round((0.5 * _CAPITAL / open_px) / _LOT) * _LOT  # 半仓股数（对齐 100 股）
    assert abs(pos - target) <= _LOT, (
        f"安静日收盘权重偏离半仓: pos={pos}, target={target}"
    )

    # 安静日无任何触价 → 不应有任何「日内做 T」成交（只有开盘的建仓买入）。
    # 第一天会建半仓（1 笔 LONG），之后安静日不产生新成交。
    trades = eng.get_all_trades()
    short_trades = [t for t in trades if t.direction == Direction.SHORT]
    assert not short_trades, f"安静日不应有卖出成交: {short_trades}"


@_SETTINGS
@given(open_px=st.sampled_from([9.0, 10.0, 11.0]))
def test_property3_two_quiet_days_no_nav_drift(open_px: float) -> None:
    # Feature: half-position-t0-backtest, Property 3: 连续多日空载净值不漂移——
    # 两个连续安静日（无触价、无幻影成交）净资产几乎不变（仅随收盘价盯市，价格恒定→无漂移）。
    eps = 0.005
    qh = _grid(open_px + eps)
    ql = _grid(open_px - eps)

    # 建半仓日 + 两个安静日，收盘价全程恒为 open_px（无价格变动）→ NAV 应等于初始资本。
    days = [_make_day(d, open_px, qh, ql, open_px) for d in range(1, 4)]
    eng = _run(days)

    pos = eng.strategy.get_pos(_VT)
    cash = eng.get_cash_available()
    nav = cash + pos * open_px  # size = 1，收盘价恒为 open_px
    # 无手续费/滑点 + 价格恒定 → NAV 应 ≈ 初始资本（容差 1 手市值）。
    assert abs(nav - _CAPITAL) <= _LOT * open_px, (
        f"连续安静日 NAV 漂移: nav={nav}, capital={_CAPITAL}"
    )


# ============================================================================
# Property 1: 无前视
# ============================================================================
@_SETTINGS
@given(
    base=_valid_run(),
    perturb_high=st.floats(min_value=0.0, max_value=0.5),
    perturb_low=st.floats(min_value=0.0, max_value=0.5),
    perturb_close=st.floats(min_value=8.0, max_value=12.0),
)
def test_property1_no_look_ahead(
    base: list[list[BarData]],
    perturb_high: float,
    perturb_low: float,
    perturb_close: float,
) -> None:
    # Feature: half-position-t0-backtest, Property 1: 无前视——
    # 两条序列前 K 日完全相同、仅第 K+1 日的 high/low/close 不同；
    # 断言两次回测中「日期 ≤ 第 K 日」的成交完全一致（价/量/方向/时间戳）。
    #
    # 构造：base 至少 2 日。K = len(base)-1（保留至少 1 个「过去日」）。
    # 第二条序列复制 base 前 K 日，第 K+1 日换成扰动后的合法 bar。
    n_days = len(base)
    k = n_days - 1  # 过去日数（1..4）；第 k+1 日（索引 k）被扰动
    cutoff_day = k  # 日序号（_make_day 的 day = 索引+1）→ 第 K 日的日序号 == k

    last_open = base[k][0].open_price
    pert_close = _grid(perturb_close)
    pert_high = _grid(max(last_open, pert_close) + perturb_high)
    pert_low = _grid(min(last_open, pert_close) - perturb_low)
    if pert_low <= 0:
        pert_low = _grid(min(last_open, pert_close))

    perturbed = [list(d) for d in base[:k]]
    perturbed.append(_make_day(k + 1, last_open, pert_high, pert_low, pert_close))

    eng_a = _run(base)
    eng_b = _run(perturbed)

    def past_trades(eng: BacktestingEngine) -> list[tuple]:
        """提取「日序号 ≤ cutoff_day」的成交规范化元组列表（按时间排序）。"""
        out = []
        for t in eng.get_all_trades():
            if t.datetime.day <= cutoff_day:
                out.append(
                    (
                        t.datetime,
                        t.direction,
                        round(t.price, 6),
                        round(t.volume, 6),
                    )
                )
        out.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        return out

    assert past_trades(eng_a) == past_trades(eng_b), (
        "扰动未来 bar 改变了过去日的成交，存在前视"
    )
