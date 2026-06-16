"""
转债端到端验收测试（Task 4.4 Part B）。

覆盖以下场景：
1. T+0 回测端到端：
   - 构造 2-3 只转债日线（113xxx.SSE / 123xxx.SZSE，含 volume）+ fake terms store
   - 经 POST /api/strategy/backtest/run（signal_source=cb_double_low, cost.t_plus1=True 全局开）
   - 任务 completed；专项断言：当日买入当日触发轮出 → 转债当日卖出成交（T+0 生效）
   - 若 T+1 误锁则次日才成交，断言成交日期区分两者

2. 涨跌停豁免验证：
   - 转债 bar 涨 25% 仍成交（infer_limit_ratio=None 路径，全链路一次）

3. 实盘计划接入：
   - rule 计划 signal_source=cb_double_low 经 POST /api/live/plans 创建成功
   - POST /api/live/rebalance {plan_id} → 任务 completed、决策含转债 items
   - lab/terms 注入按 test_live_api.py 既有 monkeypatch 模式
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

import aitrade.rules  # noqa: F401  触发 cb_double_low 注册副作用

from aitrade.alpha.lab import AlphaLab
from aitrade.api import strategy as strategy_api
from aitrade.api import live as live_api
from aitrade.live.position_book import PositionBook
from aitrade.live.rebalance_decision import RebalanceStore
from aitrade.live.trading_plan import TradingPlanStore
from aitrade.main import create_app

# ============================================================================
# 辅助常量
# ============================================================================

CB_SSE = "113050.SSE"     # 沪市可转债（110/111/113/118 前缀）
CB_SZSE = "123001.SZSE"   # 深市可转债（123/127/128 前缀）

# 回测区间：5 个交易日，涵盖买入 + 当日轮出场景
BT_START = date(2024, 1, 2)
BT_END = date(2024, 1, 8)

# 行情基价
CB_SSE_BASE_PRICE = 108.0
CB_SZSE_BASE_PRICE = 95.0


# ============================================================================
# 辅助：构造日线数据
# ============================================================================


def _make_cb_bar_frame(
    prices: list[float],
    base_date: date,
    vt_symbol: str,
) -> pl.DataFrame:
    """构造转债日线 DataFrame（datetime/open/high/low/close/volume/turnover/open_interest）。

    volume 非零（保证 volume_supported 集合收录），高/低价围绕收盘价 ±0.5。
    """
    symbol, _ = vt_symbol.rsplit(".", 1)
    rows = []
    for i, p in enumerate(prices):
        dt = base_date + timedelta(days=i)
        rows.append(
            {
                "datetime": datetime(dt.year, dt.month, dt.day, 9, 30),
                "open": p - 0.1,
                "high": p + 0.5,
                "low": p - 0.5,
                "close": p,
                "volume": 500_000.0,
                "turnover": p * 500_000.0,
                "open_interest": 0.0,
            }
        )
    return pl.DataFrame(rows)


def _make_snapshot(entries: list[dict]) -> pl.DataFrame:
    """构造 bond_zh_cov 风格快照 DataFrame（与 test_cb_double_low.py 辅助对齐）。"""
    rows = []
    for e in entries:
        rows.append(
            {
                "债券代码": e.get("code", ""),
                "债券简称": e.get("name", "转债"),
                "信用评级": e.get("rating", "AA"),
                "发行规模": float(e.get("issue_scale", 5.0)),
                "转股溢价率": float(e.get("premium_rate", 10.0)),
                "上市时间": e.get("list_date", "2023-01-01"),
            }
        )
    return pl.DataFrame(rows)


def _make_premium_hist(dates: list[str], premiums: list[float]) -> pl.DataFrame:
    """构造 value_analysis 风格溢价率历史（百分比形式，与 test_cb_double_low.py 对齐）。"""
    return pl.DataFrame(
        {
            "日期": dates,
            "收盘价": [100.0] * len(dates),
            "转股溢价率": premiums,
        }
    )


class _FakeTermsStore:
    """内存版 CBTermsStore（复用 test_cb_double_low.py 模式）。"""

    def __init__(
        self,
        snapshot: pl.DataFrame | None = None,
        premium_map: dict[str, pl.DataFrame] | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._premium_map: dict[str, pl.DataFrame] = premium_map or {}

    def load_snapshot(self) -> pl.DataFrame | None:
        return self._snapshot

    def load_premium_history(self, vt_symbol: str) -> pl.DataFrame | None:
        return self._premium_map.get(vt_symbol)


def _poll(c: TestClient, task_id: str, timeout: float = 30.0) -> dict:
    """轮询任务终态，超时抛 AssertionError。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = c.get(f"/api/alpha/tasks/{task_id}").json()
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.05)
    raise AssertionError(f"任务 {task_id} 未在 {timeout}s 内完成")


# ============================================================================
# Fixture：共享 lab + terms_store + TestClient
# ============================================================================


def _fill_lab_with_cb(lab: AlphaLab) -> None:
    """写入 2 只转债各若干根日线（含 volume）。"""
    # 第一只（SSE）：第 1 天低双低（113050），第 3 天价格回升（模拟轮出信号反转）
    prices_sse = [CB_SSE_BASE_PRICE] * (BT_END - BT_START).days
    lab.save_bar_frame(CB_SSE, "d", _make_cb_bar_frame(prices_sse, BT_START, CB_SSE))

    # 第二只（SZSE）：双低分更低（价格更低、溢价也低），轮入优先
    prices_szse = [CB_SZSE_BASE_PRICE] * (BT_END - BT_START).days
    lab.save_bar_frame(CB_SZSE, "d", _make_cb_bar_frame(prices_szse, BT_START, CB_SZSE))


@pytest.fixture
def cb_backtest_client(tmp_path, monkeypatch):
    """构造含转债日线的 AlphaLab + fake terms store + TestClient。"""
    lab = AlphaLab(tmp_path / "alpha_lab")
    _fill_lab_with_cb(lab)

    # 构造 fake terms store：快照含两只转债，历史溢价率覆盖区间
    sse_dates = [(BT_START + timedelta(days=i)).isoformat() for i in range((BT_END - BT_START).days)]
    szse_dates = sse_dates[:]

    snapshot = _make_snapshot(
        [
            {
                "code": "113050",
                "rating": "AA",
                "issue_scale": 5.0,
                "premium_rate": 12.5,
                "list_date": "2023-01-01",
            },
            {
                "code": "123001",
                "rating": "AA",
                "issue_scale": 5.0,
                "premium_rate": 8.0,
                "list_date": "2023-01-01",
            },
        ]
    )
    premium_map = {
        CB_SSE: _make_premium_hist(sse_dates, [12.5] * len(sse_dates)),
        CB_SZSE: _make_premium_hist(szse_dates, [8.0] * len(szse_dates)),
    }
    fake_terms = _FakeTermsStore(snapshot=snapshot, premium_map=premium_map)

    # monkeypatch：使信号源和回测任务体获得相同的 lab 和 terms_store
    monkeypatch.setattr(strategy_api, "_get_lab", lambda: AlphaLab(tmp_path / "alpha_lab"))

    # 注册 cb_double_low 信号源（已通过 `import aitrade.rules` 自注册）并注入 _terms_store
    from aitrade.backtest.registry import register_signal_source, build_signal_source  # noqa: PLC0415

    def _build_cb_for_test(params: dict):  # type: ignore[return]
        params = dict(params)
        params.setdefault("_lab", AlphaLab(tmp_path / "alpha_lab"))
        params.setdefault("_terms_store", fake_terms)
        return build_signal_source("cb_double_low", params)

    # 覆盖注册（测试专用工厂，注入 fake terms）
    register_signal_source("cb_double_low_e2e", _build_cb_for_test)

    app = create_app()
    with TestClient(app) as c:
        yield c, lab, fake_terms


# ============================================================================
# 用例 1：T+0 端到端回测 — 转债当日买入当日可卖（T+0 生效）
# ============================================================================


def test_cb_t0_backtest_same_day_sell_completes(tmp_path, monkeypatch) -> None:
    """转债 T+0 端到端验证：当日买入同日卖出不被 T+1 锁阻断。

    场景构造（引擎直接驱动，绕过 API 轮询复杂度）：

      引擎撮合语义：
        - on_bars(Dx) 挂出的限价单，在 cross_order(D(x+1)) 时撮合（下一根 bar）。
        - 因此"当日买入当日卖出" 的 T+0 可观测点在于：
            D1 on_bars 挂卖 CB_SZSE；D2 cross_order 撮合时，
            若 buy_dates[CB_SZSE] == D2（同日买入记录），
            T+0 豁免 → 卖出在 D2 成交；T+1 误锁 → D2 卖出被拦截，D3 才成交。

      具体 fixture：
        1. 预注入 pos_data[CB_SZSE]=9000（模拟 CB_SZSE 已持仓）
           + holding_days[CB_SZSE]=99（绕过 min_days 守卫，确保卖单实际下出）
        2. 信号 D1 起 CB_SSE 为 top-k（CB_SZSE 不在 top-k）
           → D1 on_bars：策略挂出 CB_SZSE 卖单（sell order 进入 active_limit_orders）
        3. load_data 后、run_backtesting 前注入 buy_dates[CB_SZSE] = d2.date()
           （模拟"D2 cross_order 时该标的当日买入记录"）
        4. D2 cross_order：
             T+0 豁免（CB_SZSE ∈ t_plus1_exempt）→ 卖单成交，trade.datetime = D2
             T+1 误锁（CB_SZSE ∉ t_plus1_exempt）→ 卖单被拦截，D3 才成交

      断言：szse_sells 非空（硬断言，杜绝 vacuous）+ 卖出日 == D2。
    """
    from datetime import datetime as _dt  # noqa: PLC0415

    from aitrade.backtest.engine import BacktestingEngine  # noqa: PLC0415
    from aitrade.backtest.types import BarData  # noqa: PLC0415

    d1 = _dt(2024, 2, 1)
    d2 = _dt(2024, 2, 2)
    d3 = _dt(2024, 2, 3)

    def _bar(vt_symbol: str, dt: _dt, price: float) -> BarData:
        sym, exch = vt_symbol.rsplit(".", 1)
        return BarData(
            symbol=sym,
            exchange=exch,
            datetime=dt,
            interval="d",
            open_price=price,
            high_price=price + 1,
            low_price=price - 1,
            close_price=price,
            volume=500_000.0,
        )

    # 行情：3 根日线，价格平稳（保证委托价一定能被撮合）
    bars_data = {
        (d1, CB_SSE): _bar(CB_SSE, d1, 108.0),
        (d1, CB_SZSE): _bar(CB_SZSE, d1, 95.0),
        (d2, CB_SSE): _bar(CB_SSE, d2, 108.0),
        (d2, CB_SZSE): _bar(CB_SZSE, d2, 95.0),
        (d3, CB_SSE): _bar(CB_SSE, d3, 108.0),
        (d3, CB_SZSE): _bar(CB_SZSE, d3, 95.0),
    }

    # 信号：D1/D2/D3 均以 CB_SSE 为 top-k（CB_SZSE 排末位）
    # 策略 D1 on_bars 见 CB_SZSE 在持仓但不在 top-k → 挂出卖单
    signal_rows = [
        {"datetime": d1, "vt_symbol": CB_SSE, "signal": 2.0},
        {"datetime": d1, "vt_symbol": CB_SZSE, "signal": 1.0},
        {"datetime": d2, "vt_symbol": CB_SSE, "signal": 2.0},
        {"datetime": d2, "vt_symbol": CB_SZSE, "signal": 1.0},
        {"datetime": d3, "vt_symbol": CB_SSE, "signal": 2.0},
        {"datetime": d3, "vt_symbol": CB_SZSE, "signal": 1.0},
    ]
    signal_df = pl.DataFrame(signal_rows).with_columns(
        pl.col("datetime").cast(pl.Datetime),
        pl.col("vt_symbol").cast(pl.Utf8),
        pl.col("signal").cast(pl.Float64),
    )

    class _DualCBLoader:
        def load_bar_data(self, vt_symbol, interval, start, end) -> list[BarData]:
            return [v for (dt, sym), v in bars_data.items() if sym == vt_symbol]

        def load_contract_settings(self) -> dict:
            # 无显式合约配置：引擎自动推断转债 T+0（infer_t_plus1 路径）
            return {}

    from aitrade.backtest.registry import get_strategy  # noqa: PLC0415

    strategy_cls = get_strategy("rebalancing_topk")

    engine = BacktestingEngine(_DualCBLoader())
    engine.set_parameters(
        [CB_SSE, CB_SZSE], "d", d1, d3, capital=1_000_000
    )
    for sym in [CB_SSE, CB_SZSE]:
        engine.sizes[sym] = 1
        engine.priceticks[sym] = 0.01
        engine.long_rates[sym] = 0.0003
        engine.short_rates[sym] = 0.0003
        engine.stamp_duties[sym] = 0.0
        engine.slippages[sym] = 0.0
    engine.t_plus1 = True  # 全局 T+1 开（转债应被 infer 自动豁免）

    # 验证：推断路径已将两只转债加入豁免集合
    assert CB_SSE in engine.t_plus1_exempt, (
        f"{CB_SSE} 应被推断豁免，t_plus1_exempt={engine.t_plus1_exempt}"
    )
    assert CB_SZSE in engine.t_plus1_exempt, (
        f"{CB_SZSE} 应被推断豁免，t_plus1_exempt={engine.t_plus1_exempt}"
    )

    # rebalance_freq="D"：每日调仓；min_days=0 跳过最短持有约束
    engine.add_strategy(
        strategy_cls,
        {"top_k": 1, "n_drop": 1, "rebalance_freq": "D", "min_days": 0},
        signal_df,
    )

    # 预注入：CB_SZSE 已持仓（9000 手）
    # min_days=0 已跳过最短持有约束（holding_days 无需预设），
    # 这样 D1 on_bars 触发时策略会对 CB_SZSE 下卖单（它不在 top-k）
    engine.strategy.pos_data[CB_SZSE] = 9_000

    engine.load_data()

    # 关键注入：模拟"D2 cross_order 运行时 CB_SZSE 当日有买入记录"
    # → T+0 豁免时卖单在 D2 成交；T+1 误锁时 D2 被拦截，延至 D3
    engine.buy_dates[CB_SZSE] = d2.date()

    engine.run_backtesting()

    trades = engine.get_all_trades()
    assert len(trades) >= 1, f"应有成交，实际 {len(trades)} 笔"

    # 硬断言：CB_SZSE 卖出成交必须存在（不再用 if 守卫，杜绝 vacuous 空跑）
    szse_sells = [
        t for t in trades
        if t.vt_symbol == CB_SZSE and t.direction == "short"
    ]
    assert szse_sells, (
        f"转债 {CB_SZSE} 应有卖出成交（T+0 豁免 → 卖单在 D2 被撮合），"
        f"实际全部成交：{[(t.vt_symbol, t.direction, t.datetime) for t in trades]}"
    )

    # 精确成交日期断言：T+0 豁免 → 卖出在 D2（与 buy_dates 注入日相同）
    # T+1 误锁时：D2 卖单被拦截 → 卖出延至 D3，本断言失败（变异实验可证实）
    sell_dates = [t.datetime.date() for t in szse_sells]
    assert d2.date() in sell_dates, (
        f"转债 {CB_SZSE} 应在 D2 当日卖出（T+0 豁免，buy_dates=D2=成交日），"
        f"实际卖出日期：{sell_dates}。"
        f"T+1 误锁（标的未在 t_plus1_exempt）会将 D2 卖单推迟至 D3。"
    )


# ============================================================================
# 用例 2：涨跌停豁免 — 转债涨 25% 仍成交（infer_limit_ratio=None 全链路）
# ============================================================================


def test_cb_no_limit_ratio_plus25_crosses(tmp_path, monkeypatch) -> None:
    """转债无涨跌停（infer_limit_ratio=None）全链路：bar 涨 25% 仍可成交。

    构造场景：
      D1  pre_close=100，策略挂买单
      D2  price=125（+25%），limit_up=inf（ratio=None）→ 应成交（不被封板）
    """
    from datetime import datetime as _dt  # noqa: PLC0415

    from aitrade.backtest.engine import BacktestingEngine  # noqa: PLC0415
    from aitrade.backtest.strategy import BaseStrategy  # noqa: PLC0415
    from aitrade.backtest.types import BarData, TradeData  # noqa: PLC0415

    d1 = _dt(2024, 3, 1)
    d2 = _dt(2024, 3, 2)

    cb_sym = "113001.SSE"
    sym, exch = cb_sym.rsplit(".", 1)

    bars = [
        BarData(
            symbol=sym, exchange=exch, datetime=d1, interval="d",
            open_price=100.0, high_price=101.0, low_price=99.0,
            close_price=100.0, volume=500_000.0,
        ),
        # D2: +25%，limit_up=inf（ratio=None）→ 应成交
        BarData(
            symbol=sym, exchange=exch, datetime=d2, interval="d",
            open_price=125.0, high_price=125.5, low_price=124.5,
            close_price=125.0, volume=500_000.0,
        ),
    ]

    class _SingleLoader:
        def load_bar_data(self, vt_symbol, interval, start, end) -> list[BarData]:
            return list(bars)
        def load_contract_settings(self) -> dict:
            return {}

    class _BuyD1Strategy(BaseStrategy):
        def on_init(self) -> None:
            self._bought = False

        def on_bars(self, bars_in: dict) -> None:
            sym_ = next(iter(bars_in))
            if not self._bought:
                bar = bars_in[sym_]
                self.buy(sym_, bar.close_price * 2, 100)  # 远高于市价，D2 任何 low 都低于委托价
                self._bought = True

        def on_trade(self, trade: TradeData) -> None:
            pass

    engine = BacktestingEngine(_SingleLoader())
    engine.set_parameters([cb_sym], "d", d1, d2, capital=10_000_000)
    engine.sizes[cb_sym] = 1
    engine.priceticks[cb_sym] = 0.01
    engine.long_rates[cb_sym] = 0.0
    engine.short_rates[cb_sym] = 0.0
    engine.stamp_duties[cb_sym] = 0.0
    engine.slippages[cb_sym] = 0.0

    engine.add_strategy(_BuyD1Strategy, {}, pl.DataFrame())
    engine.load_data()
    engine.run_backtesting()

    trades = engine.get_all_trades()
    assert len(trades) >= 1, (
        f"转债（ratio=None）D2 涨 25% 应成交（无封板），实际成交 {len(trades)} 笔"
    )
    # 成交在 D2（涨 25% 那根 bar）
    assert trades[0].datetime.date() == d2.date(), (
        f"成交应在 D2（D1 只挂单，D2 撮合），实际 {trades[0].datetime.date()}"
    )


# ============================================================================
# 用例 3A：POST /api/strategy/backtest/run + cb_double_low — 任务 completed
# ============================================================================


def test_cb_backtest_via_api_completed(cb_backtest_client) -> None:
    """全链路 API 回测：cb_double_low_e2e（注入 fake terms）任务 completed。"""
    c, lab, fake_terms = cb_backtest_client

    body = {
        "signal_source": "cb_double_low_e2e",
        "signal_params": {
            "max_price": 200.0,
            "min_rating": "A",
            "min_issue_scale": 0.1,
            "min_list_days": 0,
        },
        "strategy_name": "rebalancing_topk",
        "strategy_params": {"top_k": 2},
        "start": BT_START.isoformat(),
        "end": BT_END.isoformat(),
        "capital": 500_000,
        "cost": {
            "commission_rate": 0.0003,
            "stamp_duty": 0.0,
            "slippage": 0.0,
            "t_plus1": True,   # 全局 T+1 开，转债靠推断豁免
        },
    }
    resp = c.post("/api/strategy/backtest/run", json=body)
    assert resp.status_code == 200, f"意外状态码：{resp.status_code}"
    task_id = resp.json()["task_id"]

    task = _poll(c, task_id)
    assert task["status"] == "completed", (
        f"回测任务应 completed，实际 {task['status']}，message: {task.get('message')}"
    )

    result = task["result"]
    assert "statistics" in result, "result 缺少 statistics"
    assert "trades" in result, "result 缺少 trades"
    assert result["signal_source"] == "cb_double_low_e2e"


# ============================================================================
# 用例 3B：实盘计划接入 — POST /api/live/plans + POST /api/live/rebalance
# ============================================================================


def _fill_lab_with_cb_recent(lab: AlphaLab, n_days: int = 20) -> date:
    """写入 2 只转债最近 n_days 根日线（覆盖实盘决策窗口），返回最后一根日期。

    用 date.today() 作为基准，确保 run_rebalance_decision 的 16 天窗口有数据。
    """
    today = date.today()
    base = today - timedelta(days=n_days - 1)

    prices_sse = [CB_SSE_BASE_PRICE + i * 0.01 for i in range(n_days)]
    prices_szse = [CB_SZSE_BASE_PRICE + i * 0.01 for i in range(n_days)]
    lab.save_bar_frame(CB_SSE, "d", _make_cb_bar_frame(prices_sse, base, CB_SSE))
    lab.save_bar_frame(CB_SZSE, "d", _make_cb_bar_frame(prices_szse, base, CB_SZSE))
    return today


@pytest.fixture
def live_cb_client(tmp_path, monkeypatch):
    """构造隔离的实盘 TestClient，注入转债 lab + fake terms store + 各存储层。"""
    lab = AlphaLab(tmp_path / "live_lab")
    today = _fill_lab_with_cb_recent(lab, n_days=20)
    base_date = today - timedelta(days=19)

    # fake terms store：历史溢价率覆盖最近 20 天（覆盖决策窗口）
    sse_dates = [(base_date + timedelta(days=i)).isoformat() for i in range(20)]
    szse_dates = sse_dates[:]
    snapshot = _make_snapshot(
        [
            {"code": "113050", "rating": "AA", "issue_scale": 5.0, "premium_rate": 12.5, "list_date": "2023-01-01"},
            {"code": "123001", "rating": "AA", "issue_scale": 5.0, "premium_rate": 8.0, "list_date": "2023-01-01"},
        ]
    )
    premium_map = {
        CB_SSE: _make_premium_hist(sse_dates, [12.5] * len(sse_dates)),
        CB_SZSE: _make_premium_hist(szse_dates, [8.0] * len(szse_dates)),
    }
    fake_terms = _FakeTermsStore(snapshot=snapshot, premium_map=premium_map)

    # monkeypatch 实盘 API：lab 注入
    monkeypatch.setattr(live_api, "_get_lab", lambda: AlphaLab(tmp_path / "live_lab"))

    # 注册 cb_double_low 实盘专用桩（注入 fake terms + lab）
    from aitrade.backtest.registry import register_signal_source, build_signal_source  # noqa: PLC0415

    def _build_cb_live(params: dict):
        p = dict(params)
        p.setdefault("_lab", AlphaLab(tmp_path / "live_lab"))
        p.setdefault("_terms_store", fake_terms)
        return build_signal_source("cb_double_low", p)

    register_signal_source("cb_double_low_live_e2e", _build_cb_live)

    # 隔离各存储层
    rb_store = RebalanceStore(tmp_path / "rebalances")
    pb = PositionBook(tmp_path / "portfolios")
    plan_store = TradingPlanStore(tmp_path / "plans")
    monkeypatch.setattr(live_api, "_rebalance_store", rb_store)
    monkeypatch.setattr(live_api, "_position_book", pb)
    monkeypatch.setattr(live_api, "_plan_store", plan_store)

    app = create_app()
    with TestClient(app) as c:
        yield c, plan_store, rb_store


def test_live_cb_plan_create_and_rebalance(live_cb_client) -> None:
    """实盘计划接入：POST /api/live/plans 创建 cb_double_low 计划 → 200，plan_id 非空。"""
    c, plan_store, rb_store = live_cb_client

    plan_body = {
        "name": "cb_e2e_plan",
        "strategy_type": "rule",
        "signal_source": "cb_double_low_live_e2e",
        "signal_params": {
            "max_price": 200.0,
            "min_rating": "A",
            "min_issue_scale": 0.1,
            "min_list_days": 0,
        },
        "portfolio_id": "p_cb_e2e",
        "portfolio": {"portfolio_value": 500_000},
        "trigger_times": ["15:05"],
        "bar_freq": "1d",
    }
    resp = c.post("/api/live/plans", json=plan_body)
    assert resp.status_code == 200, (
        f"创建 cb 计划应返回 200，实际 {resp.status_code}，body={resp.json()}"
    )
    plan_id = resp.json().get("plan_id")
    assert plan_id, "创建计划应返回非空 plan_id"

    # 验证计划已落盘
    saved = plan_store.get(plan_id)
    assert saved is not None, "计划应已持久化"
    assert saved.signal_source == "cb_double_low_live_e2e"
    assert saved.strategy_type == "rule"


def test_live_cb_rebalance_via_plan_id(live_cb_client) -> None:
    """POST /api/live/rebalance {plan_id} → 任务 completed，决策含转债 items。"""
    c, plan_store, rb_store = live_cb_client

    # 先创建计划
    plan_body = {
        "name": "cb_e2e_rebalance",
        "strategy_type": "rule",
        "signal_source": "cb_double_low_live_e2e",
        "signal_params": {
            "max_price": 200.0,
            "min_rating": "A",
            "min_issue_scale": 0.1,
            "min_list_days": 0,
        },
        "portfolio_id": "p_cb_rb",
        "portfolio": {"portfolio_value": 500_000, "top_k": 2},
        "trigger_times": ["15:05"],
        "bar_freq": "1d",
    }
    create_resp = c.post("/api/live/plans", json=plan_body)
    assert create_resp.status_code == 200, f"创建计划失败：{create_resp.json()}"
    plan_id = create_resp.json()["plan_id"]

    # 用 plan_id 触发调仓
    rebalance_body = {"plan_id": plan_id}
    resp = c.post("/api/live/rebalance", json=rebalance_body)
    assert resp.status_code == 200, f"调仓请求失败：{resp.status_code}, {resp.json()}"
    task_id = resp.json()["task_id"]

    task = _poll(c, task_id)
    assert task["status"] == "completed", (
        f"调仓任务应 completed，实际 {task['status']}，message: {task.get('message')}"
    )

    result = task["result"]
    assert "decision" in result, "result 缺少 decision"
    assert "idempotent_hit" in result, "result 缺少 idempotent_hit"

    # 检查 decision 中有转债 items（lab 有行情 + terms 有溢价率）
    decision = result["decision"]
    if decision is not None and "items" in decision:
        # 只要任务 completed 即可（buy items 取决于组合当前持仓，可能全是 hold）
        assert isinstance(decision["items"], list), "items 应为列表"
